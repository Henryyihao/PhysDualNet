import torch
import numpy as np
from torch.utils.data import Dataset

from nino34_utils import find_nino34_indices

ALL_VAR_NAMES = ['sst', 'hc', 'mld', 'sss', 'slp', 'tauu', 'tauv']

_NOISE_STD = {
    'sst':  0.03,
    'hc':   0.015,
    'mld':  0.015,
    'sss':  0.015,
    'slp':  0.04,
    'tauu': 0.04,
    'tauv': 0.04,
}

class ENSODataset(Dataset):

    CLIP_SIGMA = 5.0

    def __init__(self,
                 arrays: list,
                 input_len: int,
                 output_len: int,
                 lat: np.ndarray,
                 lon: np.ndarray,
                 is_train: bool = True,
                 stats: dict = None,
                 segments: list = None,
                 var_names: list = None,
                 start_month: int = 0,
                 return_spatial_target: bool = False):
        super().__init__()

        if not arrays:
            raise ValueError("arrays is empty")

        self.input_len   = int(input_len)
        self.output_len  = int(output_len)
        self.window      = self.input_len + self.output_len
        self.n_vars      = len(arrays)
        self.is_train    = bool(is_train)
        self.start_month = int(start_month) % 12
        self.return_spatial_target = bool(return_spatial_target)

        if var_names is not None:
            if len(var_names) != len(arrays):
                raise ValueError(
                    f"var_names length ({len(var_names)}) != arrays length ({len(arrays)})"
                )
            self.var_names = list(var_names)
        else:
            self.var_names = ALL_VAR_NAMES[:self.n_vars]

        if 'sst' not in self.var_names:
            raise ValueError("SST must be included because Nino3.4 target is extracted from SST.")

        arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
        shapes = [a.shape for a in arrays]
        if len(set(shapes)) != 1:
            raise ValueError(f"All variable arrays must have identical shape, got: {shapes}")
        if arrays[0].ndim != 3:
            raise ValueError(f"Each variable array must be (T,H,W), got {arrays[0].shape}")

        self.sst_idx = self.var_names.index('sst')
        self.lat = np.asarray(lat, dtype=np.float32)
        self.lon = np.asarray(lon, dtype=np.float32)

        lat_s, lat_e, lon_s, lon_e = find_nino34_indices(self.lat, self.lon)
        self._nino_slices = (lat_s, lat_e, lon_s, lon_e)

        lat_region = self.lat[lat_s:lat_e]
        self._weights = np.cos(np.deg2rad(lat_region)).astype(np.float32)
        self._weights_2d = self._weights[:, np.newaxis]
        self._w_sum = float(self._weights_2d.sum() * (lon_e - lon_s))

        T_total = arrays[0].shape[0]
        if T_total == 0:
            raise ValueError("Data time dimension is 0")

        if segments is None:
            segments = [(0, T_total)]
        self.segments = self._sanitize_segments(segments, T_total)

        if is_train:
            self.stats = stats if stats is not None else self._compute_stats(arrays, self.segments)
            if 'nino34_mean' not in self.stats:
                self._compute_nino34_stats(arrays, self.segments)
        else:
            if stats is None:
                raise ValueError("Val/test dataset requires training stats dict")
            self.stats = stats

        self._arrays = self._clip(arrays)

        self._valid_starts = []
        self._sample_model_ids = []
        for seg_id, (s, e) in enumerate(self.segments):
            for idx in range(s, e - self.window + 1):
                self._valid_starts.append(idx)
                self._sample_model_ids.append(seg_id)

        if len(self._valid_starts) == 0:
            raise ValueError(
                f"No valid samples: T={T_total}, window={self.window}, segments={self.segments}."
            )

        print(
            f"  [Dataset] samples={len(self._valid_starts)} segments={len(self.segments)} "
            f"vars={self.var_names} spatial={self.spatial_shape} start_month={self.start_month+1}"
        )

    @staticmethod
    def _sanitize_segments(segments, T_total: int):
        clean = []
        for s, e in segments:
            s = max(0, int(s))
            e = min(int(e), T_total)
            if e > s:
                clean.append((s, e))
        if not clean:
            raise ValueError(f"No non-empty segments after sanitization. T_total={T_total}, segments={segments}")
        return clean

    @staticmethod
    def _concat_segment_values(arr: np.ndarray, segments: list):
        if len(segments) == 1 and segments[0] == (0, arr.shape[0]):
            return arr
        return np.concatenate([arr[s:e] for s, e in segments], axis=0)

    def _compute_stats(self, arrays: list, segments: list) -> dict:
        stats = {}
        for i, arr in enumerate(arrays):
            vn = self.var_names[i]
            vals = self._concat_segment_values(arr, segments)
            mu = float(np.nanmean(vals))
            sig = max(float(np.nanstd(vals)), 1e-6)
            stats[f'{vn}_mean'] = mu
            stats[f'{vn}_std'] = sig
            print(f"  [Stats] {vn:8s}  mean={mu:+.6f}  std={sig:.6f}")
        return stats

    def _compute_nino34_stats(self, arrays, segments):
        sst = arrays[self.sst_idx]
        lat_s, lat_e, lon_s, lon_e = self._nino_slices
        vals = []
        for s, e in segments:
            for t in range(s, e):
                region = sst[t, lat_s:lat_e, lon_s:lon_e]
                vals.append(np.nansum(region * self._weights_2d) / self._w_sum)
        nino_vals = np.asarray(vals, dtype=np.float32)
        self.stats['nino34_mean'] = float(np.nanmean(nino_vals))
        self.stats['nino34_std'] = max(float(np.nanstd(nino_vals)), 1e-6)
        print(f"  [Stats] {'nino34':8s}  mean={self.stats['nino34_mean']:+.6f}  std={self.stats['nino34_std']:.6f}")

    def _clip(self, arrays: list) -> list:
        clipped = []
        for i, arr in enumerate(arrays):
            vn  = self.var_names[i]
            mu  = float(self.stats[f'{vn}_mean'])
            sig = max(float(self.stats[f'{vn}_std']), 1e-6)
            arr = np.clip(arr, mu - self.CLIP_SIGMA * sig,
                          mu + self.CLIP_SIGMA * sig)
            arr = np.nan_to_num(arr, nan=mu, posinf=mu + self.CLIP_SIGMA * sig,
                                neginf=mu - self.CLIP_SIGMA * sig)
            clipped.append(arr.astype(np.float32, copy=False))
        return clipped

    def _calc_nino34(self, sst_slice):
        lat_s, lat_e, lon_s, lon_e = self._nino_slices
        T = sst_slice.shape[0]
        nino = np.zeros(T, dtype=np.float32)
        for t in range(T):
            region = sst_slice[t, lat_s:lat_e, lon_s:lon_e]
            nino[t] = np.nansum(region * self._weights_2d) / self._w_sum
        return nino

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(self, idx: int):
        start = self._valid_starts[idx]
        mid   = start + self.input_len
        end   = mid + self.output_len

        in_list = []
        for i, arr in enumerate(self._arrays):
            vn  = self.var_names[i]
            mu  = float(self.stats[f'{vn}_mean'])
            sig = float(self.stats[f'{vn}_std']) + 1e-6
            x_val = (arr[start:mid] - mu) / sig

            if self.is_train:
                noise_std = _NOISE_STD.get(vn, 0.03)
                noise = np.random.normal(0.0, noise_std, size=x_val.shape).astype(np.float32)
                x_val = x_val + noise

            in_list.append(x_val.astype(np.float32, copy=False))

        x = torch.from_numpy(np.stack(in_list, axis=1)).float()

        sst_target = self._arrays[self.sst_idx][mid:end]
        nino_raw = self._calc_nino34(sst_target)
        nino_mean = float(self.stats['nino34_mean'])
        nino_std  = float(self.stats['nino34_std']) + 1e-6
        nino_norm = (nino_raw - nino_mean) / nino_std
        y = torch.from_numpy(nino_norm.astype(np.float32)).float()

        init_month = (self.start_month + start + self.input_len - 1) % 12
        init_month_tensor = torch.tensor(init_month, dtype=torch.long)

        model_id = self._sample_model_ids[idx]
        model_id_tensor = torch.tensor(model_id, dtype=torch.long)

        if self.return_spatial_target:
            tgt_list = []
            for i, arr in enumerate(self._arrays):
                vn  = self.var_names[i]
                mu  = float(self.stats[f'{vn}_mean'])
                sig = float(self.stats[f'{vn}_std']) + 1e-6
                tgt_val = (arr[mid:end] - mu) / sig
                tgt_list.append(tgt_val.astype(np.float32, copy=False))
            y_spatial = torch.from_numpy(np.stack(tgt_list, axis=1)).float()
            return x, y, init_month_tensor, model_id_tensor, y_spatial

        return x, y, init_month_tensor, model_id_tensor

    def get_stats(self) -> dict:
        return self.stats

    @property
    def spatial_shape(self):
        return self._arrays[0].shape[1:]
