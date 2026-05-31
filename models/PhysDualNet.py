import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    PBSEncoder,
    SpatialEncoder,
    SoftMaskIndexExtractor,
    IndexTemporalEncoder,
    RechargeOscillatorLoss,
)

class LeadAdaptiveCrossGatingFusion(nn.Module):

    def __init__(self, nhead: int, d_sp: int, d_ix: int, d_out: int,
                 T_out: int = 24, dropout: float = 0.1):
        super().__init__()
        self.T_out = T_out
        self.d_out = d_out

        self.sp_proj = nn.Linear(d_sp, d_out)
        self.ix_proj = nn.Linear(d_ix, d_out)

        self.gate_net = nn.Sequential(
            nn.Linear(d_out * 2, d_out),
            nn.Sigmoid(),
        )

        self.lead_bias = nn.Embedding(T_out, d_out)

        nn.init.zeros_(self.lead_bias.weight)

        self.layer_norm = nn.LayerNorm(d_out)

        self.temp_transformer = nn.TransformerEncoderLayer(
            d_model=d_out,
            nhead=nhead,
            dim_feedforward=d_out * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )

        self.temp_attn = nn.Sequential(
            nn.Linear(d_out, d_out // 4),
            nn.Tanh(),
            nn.Linear(d_out // 4, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sp_feats: torch.Tensor, ix_feats: torch.Tensor) -> tuple:
        sp = self.sp_proj(sp_feats)
        ix = self.ix_proj(ix_feats)

        gate_base = self.gate_net(torch.cat([sp, ix], dim=-1))

        fused = gate_base * sp + (1 - gate_base) * ix
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        seq_out = self.temp_transformer(fused)

        alpha = F.softmax(self.temp_attn(seq_out), dim=1)
        context = (alpha * seq_out).sum(dim=1)

        ctx_recent = seq_out[:, -3:].mean(dim=1)

        T_seq = seq_out.size(1)
        w_long = torch.linspace(1.0, 0.3, T_seq, device=seq_out.device)
        w_long = w_long / w_long.sum()
        ctx_long = (seq_out * w_long.view(1, -1, 1)).sum(dim=1)
        context = 0.5 * context + 0.5 * ctx_long

        return context, seq_out, ctx_recent, gate_base

    def get_lead_adaptive_context(self, sp_feats: torch.Tensor,
                                   ix_feats: torch.Tensor,
                                   target_lead: int) -> torch.Tensor:
        sp = self.sp_proj(sp_feats)
        ix = self.ix_proj(ix_feats)

        gate_base = self.gate_net(torch.cat([sp, ix], dim=-1))

        lead_idx = torch.tensor(target_lead - 1, device=sp.device)
        bias = self.lead_bias(lead_idx)
        bias = bias.unsqueeze(0).unsqueeze(0).expand_as(gate_base)

        gate_logit = torch.logit(gate_base.clamp(1e-7, 1 - 1e-7))
        gate = torch.sigmoid(gate_logit + bias)

        fused = gate * sp + (1 - gate) * ix
        fused = self.layer_norm(fused)

        seq_out = self.temp_transformer(fused)

        alpha = F.softmax(self.temp_attn(seq_out), dim=1)
        context = (alpha * seq_out).sum(dim=1)

        return context

class LeadAdaptivePredictionHead(nn.Module):

    def __init__(self, nhead: int, d_in: int, T_out: int, d_pbs: int,
                 T_in: int = 12, dropout: float = 0.1,
                 pbs_enc: PBSEncoder = None,
                 fusion_module = None):
        super().__init__()
        self.T_out = T_out
        self.T_in = T_in
        self.fusion_module = fusion_module

        self.pbs_enc = pbs_enc if pbs_enc is not None else PBSEncoder(d_pbs)

        self.lead_dist_proj = nn.Linear(4, d_pbs, bias=False)

        d_hidden = d_in + d_pbs

        self.lead_attn = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=nhead,
            dim_feedforward=d_hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(d_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

        self.persistence_scale = nn.Parameter(torch.tensor(0.5))
        self.raw_persistence_tau = nn.Parameter(torch.tensor(1.79))

    def forward(self, context: torch.Tensor, init_month: torch.LongTensor,
                ctx_recent: torch.Tensor = None,
                nino_init: torch.Tensor = None,
                sp_feats: torch.Tensor = None,
                ix_feats: torch.Tensor = None) -> tuple:
        B = context.size(0)
        device = context.device

        future_pbs = self.pbs_enc.generate_future(init_month, self.T_in, self.T_out)

        omega = 2.0 * np.pi / 24.0
        leads = torch.arange(1, self.T_out + 1, device=device, dtype=torch.float32)
        wt = omega * leads
        lead_basis = torch.stack([
            torch.sin(wt), torch.cos(wt),
            torch.sin(2.0 * wt), torch.cos(2.0 * wt)
        ], dim=-1)
        lead_dist = self.lead_dist_proj(lead_basis)

        pbs_with_dist = future_pbs + lead_dist.unsqueeze(0)

        if sp_feats is not None and ix_feats is not None and self.fusion_module is not None:

            contexts = []
            for lead in range(1, self.T_out + 1):
                ctx_lead = self.fusion_module.get_lead_adaptive_context(
                    sp_feats, ix_feats, lead)
                contexts.append(ctx_lead)
            context_adaptive = torch.stack(contexts, dim=1)
        else:

            if ctx_recent is not None:
                alpha = torch.linspace(0.5, 0.0, self.T_out, device=device)
                context_adaptive = (
                    alpha.view(1, -1, 1) * ctx_recent.unsqueeze(1) +
                    (1 - alpha.view(1, -1, 1)) * context.unsqueeze(1)
                )
            else:
                context_adaptive = context.unsqueeze(1).expand(-1, self.T_out, -1)

        h = torch.cat([context_adaptive, pbs_with_dist], dim=-1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            self.T_out, device=device)
        h = self.lead_attn(h, src_mask=causal_mask)

        out = self.out_proj(h)
        nino_pred = out[:, :, 0]
        wwv_pred = out[:, :, 1]

        if nino_init is not None:
            tau = F.softplus(self.raw_persistence_tau)
            pers_weight = torch.exp(-leads / tau)
            pers_baseline = (
                nino_init.view(-1, 1) *
                self.persistence_scale *
                pers_weight.unsqueeze(0)
            )
            nino_pred = nino_pred + pers_baseline

        return nino_pred, wwv_pred

class PhysDualNet(nn.Module):

    def __init__(self, args):
        super().__init__()

        C       = args.input_dim
        T_in    = args.input_len
        T_out   = args.output_len
        d_model = args.d_model
        dropout = args.dropout
        nhead   = args.heads
        d_pbs   = 16

        var_config = args.var_config
        var_names  = var_config['var_names']
        lat = np.array(args.lat_coords)
        lon = np.array(args.lon_coords)

        self.T_in  = T_in
        self.T_out = T_out

        self.spatial_enc = SpatialEncoder(
            in_channels=C, d_out=d_model,
            n_lat_bins=3, n_lon_bins=4)

        self.pbs_enc = PBSEncoder(d_pbs)

        self.idx_ext = SoftMaskIndexExtractor(
            lat=lat, lon=lon, var_names=var_names, temperature=2.0)
        n_indices = self.idx_ext.n_indices

        self.idx_temp = IndexTemporalEncoder(
            n_indices=n_indices, d_pbs=d_pbs,
            d_out=d_model, dropout=dropout)

        self.fusion = LeadAdaptiveCrossGatingFusion(
            nhead=nhead,
            d_sp=d_model, d_ix=d_model,
            d_out=d_model, T_out=T_out, dropout=dropout)

        self.pred_head = LeadAdaptivePredictionHead(
            nhead=nhead,
            d_in=d_model, T_out=T_out, d_pbs=d_pbs,
            T_in=T_in, dropout=dropout,
            pbs_enc=self.pbs_enc,
            fusion_module=self.fusion)

        self.ro_loss = RechargeOscillatorLoss(T_out=T_out, dt=1.0)

        self.wwv_loss_weight = float(getattr(args, 'wwv_loss_weight', 0.05))
        self.last_ode_loss = None
        self.last_wwv_loss = None

        self.phys_weight_min    = 0.01
        self.phys_weight_max    = 0.18
        self.phys_anneal_epochs = 30
        self._current_phys_weight = self.phys_weight_min

        self._init_weights()

    def set_phys_anneal(self, epoch: int):
        ratio = min(1.0, epoch / max(1, self.phys_anneal_epochs))
        self._current_phys_weight = (
            self.phys_weight_min
            + (self.phys_weight_max - self.phys_weight_min) * ratio
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                init_month: torch.LongTensor,
                tgt_spatial=None,
                wwv_target: torch.Tensor = None) -> tuple:

        sp_feats = self.spatial_enc(x)

        pbs_in  = self.pbs_enc(init_month, self.T_in)
        indices = self.idx_ext(x)
        ix_feats = self.idx_temp(indices, pbs_in)

        context, seq_out, ctx_recent, _ = self.fusion(sp_feats, ix_feats)

        nino_init = indices[:, -1, 0]

        nino_pred, wwv_pred = self.pred_head(
            context, init_month,
            ctx_recent=ctx_recent,
            nino_init=nino_init,
            sp_feats=sp_feats,
            ix_feats=ix_feats)

        ode_loss = self.ro_loss(nino_pred, wwv_pred, init_month)
        phys_loss = self._current_phys_weight * ode_loss

        wwv_loss = torch.zeros((), device=x.device, dtype=nino_pred.dtype)
        if wwv_target is not None and self.wwv_loss_weight > 0.0:
            wwv_target = wwv_target.to(device=wwv_pred.device, dtype=wwv_pred.dtype)
            if wwv_target.shape != wwv_pred.shape:
                raise ValueError(
                    f"wwv_target shape {tuple(wwv_target.shape)} does not match "
                    f"wwv_pred shape {tuple(wwv_pred.shape)}"
                )
            wwv_loss = F.smooth_l1_loss(wwv_pred, wwv_target, reduction='mean')
            phys_loss = phys_loss + self.wwv_loss_weight * wwv_loss

        self.last_ode_loss = ode_loss.detach()
        self.last_wwv_loss = wwv_loss.detach()

        return nino_pred, phys_loss
