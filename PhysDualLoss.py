import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysDualLoss(nn.Module):

    def __init__(self, spb_weight: float = 2.0, corr_weight: float = 0.3,
                 trend_weight: float = 0.1, var_weight: float = 0.1,
                 huber_beta: float = 1.0):
        super().__init__()
        self.spb_weight = spb_weight
        self.corr_weight = corr_weight
        self.trend_weight = trend_weight
        self.var_weight = var_weight
        self.huber = nn.SmoothL1Loss(reduction='none', beta=huber_beta)

    def _spb_weight_tensor(self, init_month, T_out, device):
        lead_idx = torch.arange(T_out, device=device)
        target_months = (init_month.unsqueeze(1) + lead_idx.unsqueeze(0) + 1) % 12

        is_spring = (target_months >= 1) & (target_months <= 4)

        is_early_summer = (target_months >= 5) & (target_months <= 6)

        w = torch.ones_like(target_months, dtype=torch.float32)
        w = torch.where(is_spring,
                        torch.full_like(w, self.spb_weight),
                        w)
        w = torch.where(is_early_summer & ~is_spring,
                        torch.full_like(w, (self.spb_weight + 1.0) / 2.0),
                        w)
        return w

    def _lead_weighted_huber(self, pred, target, init_month, sample_weight):
        B, T = pred.shape
        device = pred.device

        leads = torch.arange(T, device=device, dtype=torch.float32)
        short_peak = 1.3 * torch.exp(-leads / 3.5)
        long_peak  = 0.85 * torch.exp(-((leads - 18.0) / 5.0) ** 2)
        lead_w = 0.5 + short_peak + long_peak
        lead_w = lead_w / lead_w.mean()

        huber_loss = self.huber(pred, target)

        if init_month is not None:
            spb_w = self._spb_weight_tensor(init_month, T, device)
        else:
            spb_w = torch.ones(B, T, device=device)

        weighted = huber_loss * lead_w.unsqueeze(0) * spb_w

        if sample_weight is not None:
            weighted = weighted * sample_weight.unsqueeze(1)

        return weighted.mean()

    def _pairwise_ranking_loss(self, pred, target):
        B, T = pred.shape
        if B < 4:
            return torch.tensor(0.0, device=pred.device)

        n_pairs = min(B * 4, B * (B - 1) // 2)
        idx_i = torch.randint(0, B, (n_pairs,), device=pred.device)
        idx_j = torch.randint(0, B, (n_pairs,), device=pred.device)
        valid = idx_i != idx_j
        idx_i, idx_j = idx_i[valid], idx_j[valid]

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_diff = pred[idx_i] - pred[idx_j]
        true_diff = target[idx_i] - target[idx_j]

        pair_loss = torch.clamp(-pred_diff * true_diff, min=0.0)

        leads = torch.arange(T, device=pred.device, dtype=torch.float32)
        lead_w = 1.0 + torch.exp(-0.02 * (leads - 16.0) ** 2)
        lead_w = lead_w / lead_w.mean()

        return (pair_loss * lead_w.unsqueeze(0)).mean()

    def _trend_consistency_loss(self, pred, target):
        B, T = pred.shape
        if T < 2:
            return torch.tensor(0.0, device=pred.device)

        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = target[:, 1:] - target[:, :-1]

        trend_loss = torch.clamp(-pred_diff * true_diff, min=0.0)

        return trend_loss.mean()

    def _variance_matching_loss(self, pred, target):
        B, T = pred.shape
        if B < 4:
            return torch.tensor(0.0, device=pred.device)

        p_std = pred.std(dim=0).clamp(min=1e-4)
        t_std = target.std(dim=0).clamp(min=1e-4)

        ratio = p_std / t_std
        var_loss = F.smooth_l1_loss(ratio, torch.ones_like(ratio), beta=0.5)

        return var_loss

    def forward(self, pred, target, init_month=None, sample_weight=None):

        main_loss = self._lead_weighted_huber(pred, target, init_month, sample_weight)

        corr_loss = self._pairwise_ranking_loss(pred, target)

        trend_loss = self._trend_consistency_loss(pred, target)

        var_loss = self._variance_matching_loss(pred, target)

        total = (main_loss
                 + self.corr_weight * corr_loss
                 + self.trend_weight * trend_loss
                 + self.var_weight * var_loss)

        return total
