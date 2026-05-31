import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PBSEncoder(nn.Module):

    def __init__(self, d_pbs: int = 16):
        super().__init__()
        self.d_pbs = d_pbs
        self.omega = 2.0 * math.pi / 12.0
        self.proj = nn.Linear(4, d_pbs)

    def forward(self, init_month: torch.LongTensor, seq_len: int) -> torch.Tensor:
        device = init_month.device

        offsets = torch.arange(
            -(seq_len - 1), 1,
            device=device,
            dtype=torch.float32
        )

        t = init_month.unsqueeze(1).float() + offsets.unsqueeze(0)
        wt = self.omega * t

        basis = torch.stack([
            torch.sin(wt), torch.cos(wt),
            torch.sin(2.0 * wt), torch.cos(2.0 * wt),
        ], dim=-1)

        return self.proj(basis)

    def generate_future(self, init_month: torch.LongTensor,
                        T_in: int, T_out: int) -> torch.Tensor:
        device = init_month.device

        offsets = torch.arange(
            1, T_out + 1,
            device=device,
            dtype=torch.float32
        )

        t = init_month.unsqueeze(1).float() + offsets.unsqueeze(0)
        wt = self.omega * t

        basis = torch.stack([
            torch.sin(wt), torch.cos(wt),
            torch.sin(2.0 * wt), torch.cos(2.0 * wt),
        ], dim=-1)

        return self.proj(basis)

class ChannelSE(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        hidden = max(2, in_channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x)
        return x * w.unsqueeze(-1).unsqueeze(-1)

class SpatialEncoder(nn.Module):

    def __init__(self, in_channels: int, d_out: int,
                 n_lat_bins: int = 3, n_lon_bins: int = 4):
        super().__init__()

        self.channel_se = ChannelSE(in_channels, reduction=2)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((n_lat_bins, n_lon_bins))
        self.proj = nn.Linear(64 * n_lat_bins * n_lon_bins, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.channel_se(x)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x.reshape(B, T, -1)

def _build_region_priors(lat, lon, var_names):
    def _var_idx(name, fallback='sst'):
        try:
            return var_names.index(name)
        except ValueError:

            return var_names.index(fallback)

    sst = _var_idx('sst')
    hc  = _var_idx('hc', 'sst')  if 'hc'  in var_names else sst
    mld = _var_idx('mld', 'sst') if 'mld' in var_names else sst

    priors = [
        {'name': 'Nino34',     'var_idx': sst, 'lat_lo': -5,  'lat_hi': 5,   'lon_lo_360': 190, 'lon_hi_360': 240},
        {'name': 'WWV',        'var_idx': hc,  'lat_lo': -5,  'lat_hi': 5,   'lon_lo_360': 120, 'lon_hi_360': 280},
        {'name': 'IOB',        'var_idx': sst, 'lat_lo': -20, 'lat_hi': 20,  'lon_lo_360': 40,  'lon_hi_360': 100},
        {'name': 'IOD_east',   'var_idx': sst, 'lat_lo': -10, 'lat_hi': 0,   'lon_lo_360': 90,  'lon_hi_360': 110},
        {'name': 'IOD_west',   'var_idx': sst, 'lat_lo': -10, 'lat_hi': 10,  'lon_lo_360': 50,  'lon_hi_360': 70},
        {'name': 'TNA',        'var_idx': sst, 'lat_lo': 5,   'lat_hi': 25,  'lon_lo_360': 305, 'lon_hi_360': 345},
        {'name': 'ATL3',       'var_idx': sst, 'lat_lo': -3,  'lat_hi': 3,   'lon_lo_360': 340, 'lon_hi_360': 360},
        {'name': 'Nino4',      'var_idx': sst, 'lat_lo': -5,  'lat_hi': 5,   'lon_lo_360': 160, 'lon_hi_360': 210},
        {'name': 'WP_WarmPool','var_idx': mld, 'lat_lo': -10, 'lat_hi': 10,  'lon_lo_360': 120, 'lon_hi_360': 160},
        {'name': 'SASD',       'var_idx': sst, 'lat_lo': -35, 'lat_hi': -20, 'lon_lo_360': 300, 'lon_hi_360': 360},
    ]
    return priors

def _make_prior_mask(lat, lon, lat_lo, lat_hi, lon_lo_360, lon_hi_360):
    lat_mask = (lat >= lat_lo) & (lat <= lat_hi)
    if np.any(lon > 180):
        lon_mask = (lon >= lon_lo_360) & (lon <= lon_hi_360)
    else:
        lo = lon_lo_360 if lon_lo_360 <= 180 else lon_lo_360 - 360
        hi = lon_hi_360 if lon_hi_360 <= 180 else lon_hi_360 - 360
        if lo <= hi:
            lon_mask = (lon >= lo) & (lon <= hi)
        else:
            lon_mask = (lon >= lo) | (lon <= hi)
    return np.outer(lat_mask, lon_mask).astype(np.float32)

class SoftMaskIndexExtractor(nn.Module):

    def __init__(self, lat: np.ndarray, lon: np.ndarray,
                 var_names: list, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature
        priors = _build_region_priors(lat, lon, var_names)
        self.n_indices = len(priors)

        self.register_buffer('var_indices',
                             torch.tensor([p['var_idx'] for p in priors], dtype=torch.long))

        cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
        self.register_buffer('cos_lat', torch.from_numpy(cos_lat))

        mask_params = []
        for p in priors:
            prior = _make_prior_mask(lat, lon,
                                     p['lat_lo'], p['lat_hi'],
                                     p['lon_lo_360'], p['lon_hi_360'])
            mask_params.append(temperature * (prior * 6.0 - 3.0))

        self.mask_params = nn.Parameter(
            torch.from_numpy(np.stack(mask_params, axis=0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        soft_masks = torch.sigmoid(self.mask_params / self.temperature)
        cos_w = self.cos_lat.unsqueeze(0).unsqueeze(-1)
        weighted_masks = soft_masks * cos_w
        mask_sums = weighted_masks.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        norm_masks = weighted_masks / mask_sums

        indices_list = []
        for i in range(self.n_indices):
            vi = self.var_indices[i]
            field = x[:, :, vi, :, :]
            idx = (field * norm_masks[i].unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))
            indices_list.append(idx)

        return torch.stack(indices_list, dim=-1)

class IndexTemporalEncoder(nn.Module):

    def __init__(self, n_indices: int, d_pbs: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        d_in = n_indices + d_pbs
        self.input_proj = nn.Linear(d_in, d_out)

        self.conv1 = nn.Conv1d(d_out, d_out, kernel_size=3, padding=0)
        self.causal_pad1 = 2
        self.norm1 = nn.GroupNorm(min(8, d_out), d_out)

        self.conv2 = nn.Conv1d(d_out, d_out, kernel_size=3, padding=0, dilation=2)
        self.causal_pad2 = 4
        self.norm2 = nn.GroupNorm(min(8, d_out), d_out)

        self.dropout = nn.Dropout(dropout)

    def forward(self, indices: torch.Tensor, pbs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([indices, pbs], dim=-1)
        x = self.input_proj(x)

        h = x.transpose(1, 2)

        h = F.pad(h, (self.causal_pad1, 0))
        h = self.dropout(F.gelu(self.norm1(self.conv1(h))))

        h = F.pad(h, (self.causal_pad2, 0))
        h = self.dropout(F.gelu(self.norm2(self.conv2(h))))

        return h.transpose(1, 2) + x

class CrossGatingFusion(nn.Module):

    def __init__(self, nhead: int, d_sp: int, d_ix: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.sp_proj = nn.Linear(d_sp, d_out)
        self.ix_proj = nn.Linear(d_ix, d_out)
        self.gate_net = nn.Sequential(
            nn.Linear(d_out * 2, d_out),
            nn.Sigmoid(),
        )
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

    def forward(self, sp_feats: torch.Tensor,
                ix_feats: torch.Tensor) -> tuple:
        sp = self.sp_proj(sp_feats)
        ix = self.ix_proj(ix_feats)

        gate = self.gate_net(torch.cat([sp, ix], dim=-1))
        fused = gate * sp + (1 - gate) * ix
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

        return context, seq_out, ctx_recent

class PBSPredictionHead(nn.Module):

    def __init__(self, nhead: int, d_in: int, T_out: int, d_pbs: int,
                 T_in: int = 12, dropout: float = 0.1,
                 pbs_enc: PBSEncoder = None):
        super().__init__()
        self.T_out = T_out
        self.T_in = T_in

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
        self.register_buffer('persistence_tau', torch.tensor(6.0))

    def _lead_dist_encoding(self, device: torch.device) -> torch.Tensor:
        leads = torch.arange(1, self.T_out + 1, device=device, dtype=torch.float32)
        omega1 = 2.0 * math.pi / self.T_out
        omega2 = 2.0 * math.pi / max(self.T_out / 2.0, 1.0)
        basis = torch.stack([
            torch.sin(omega1 * leads),
            torch.cos(omega1 * leads),
            torch.sin(omega2 * leads),
            torch.cos(omega2 * leads),
        ], dim=-1)
        return self.lead_dist_proj(basis)

    def forward(self, context: torch.Tensor,
                init_month: torch.LongTensor,
                seq_context: torch.Tensor = None,
                ctx_recent: torch.Tensor = None,
                nino_init: torch.Tensor = None):
        B = context.shape[0]
        device = context.device

        future_pbs = self.pbs_enc.generate_future(init_month, self.T_in, self.T_out)

        lead_dist = self._lead_dist_encoding(device)
        future_pbs = future_pbs + lead_dist.unsqueeze(0)

        lead_idx_for_fade = torch.arange(1.0, self.T_out + 1.0, device=device)
        pbs_gain = 1.0 - 0.30 * torch.sigmoid((lead_idx_for_fade - 17.0) / 2.0)
        future_pbs = future_pbs * pbs_gain.unsqueeze(0).unsqueeze(-1)

        lead_idx = torch.arange(1, self.T_out + 1,device=device,dtype=torch.float32)
        w_recent = (0.5 * (1.0 - lead_idx / self.T_out)).unsqueeze(0).unsqueeze(-1)
        ctx_full = context.unsqueeze(1).expand(-1, self.T_out, -1)
        if ctx_recent is not None:
            ctx_rec = ctx_recent.unsqueeze(1).expand(-1, self.T_out, -1)
            ctx = (1.0 - w_recent) * ctx_full + w_recent * ctx_rec
        else:
            ctx = ctx_full

        h = torch.cat([ctx, future_pbs], dim=-1)

        causal_mask = torch.triu(
            torch.ones(self.T_out, self.T_out, device=device), diagonal=1
        ).bool()
        h = self.lead_attn(h, src_mask=causal_mask)

        out = self.out_proj(h)
        nino_pred = out[:, :, 0]
        wwv_pred  = out[:, :, 1]

        if nino_init is not None:
            leads = torch.arange(1.0, self.T_out + 1.0, device=device)
            pers_weight = torch.exp(-leads / self.persistence_tau)
            pers_baseline = (
                nino_init.view(-1, 1) *
                self.persistence_scale *
                pers_weight.unsqueeze(0)
            )
            nino_pred = nino_pred + pers_baseline

        return nino_pred, wwv_pred

class RechargeOscillatorLoss(nn.Module):

    def __init__(self, T_out: int = 24, dt: float = 1.0):
        super().__init__()
        self.T_out = T_out
        self.dt = dt

        self.raw_a = nn.Parameter(torch.tensor(0.0))
        self.raw_b = nn.Parameter(torch.tensor(0.5))
        self.raw_c = nn.Parameter(torch.tensor(0.0))

    def forward(self, T_pred: torch.Tensor, h_pred: torch.Tensor,
                init_month: torch.LongTensor = None) -> torch.Tensor:

        a = -F.softplus(self.raw_a)
        b = F.softplus(self.raw_b)
        c = F.softplus(self.raw_c)

        dT_dt = (T_pred[:, 1:] - T_pred[:, :-1]) / self.dt
        dh_dt = (h_pred[:, 1:] - h_pred[:, :-1]) / self.dt

        T_vals = T_pred[:, :-1]
        h_vals = h_pred[:, :-1]

        T_ode_tend = a * T_vals + b * h_vals
        h_ode_tend = -c * T_vals

        lead_idx_ode = torch.arange(self.T_out - 1, device=T_pred.device, dtype=torch.float32)
        lead_weights = 0.6 + 0.6 * (lead_idx_ode / max(self.T_out - 2, 1))

        loss_T = (F.mse_loss(dT_dt, T_ode_tend, reduction='none') * lead_weights).mean()
        loss_h = (F.mse_loss(dh_dt, h_ode_tend, reduction='none') * lead_weights).mean()

        return loss_T + loss_h
