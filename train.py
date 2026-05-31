import torch
from torch.utils.data import DataLoader
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
import random
from models import get_model_dict
from dataset import ENSODataset
from PhysDualLoss import PhysDualLoss
from predict_utils import run_all_predict_plots
from utils import (EarlyStopping, evaluate_nino34_skill_decay,
                   save_nino34_to_csv_and_plot, plot_extreme_event_case_study,
                   plot_power_spectrum, plot_nino34_lead_correlation,
                   plot_seasonal_lead_heatmap, plot_init_month_lead_heatmap,
                   compute_acc_per_lead, compute_acc_3ma, compute_effective_lead)
from nino34_utils import find_nino34_indices

SPATIAL_TARGET_MODELS = set()
PHYS_MODELS = {'PhysDualNet'}
INIT_MONTH = {'PhysDualNet'}

def evaluate_on_obs(model, te_loader, model_name, stats, device):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in te_loader:
            x, y, init_month, _ = batch[:4]
            x, y, init_month = x.to(device), y.to(device), init_month.to(device)
            preds, _ = model_forward(model, x, model_name, init_month)
            all_preds.append(preds.cpu())
            all_trues.append(y.cpu())

    all_preds = torch.cat(all_preds, 0).numpy()
    all_trues = torch.cat(all_trues, 0).numpy()

    nino_mean = stats.get('nino34_mean', 0.0)
    nino_std  = stats.get('nino34_std',  1.0)
    preds_phys = all_preds * nino_std + nino_mean
    trues_phys = all_trues * nino_std + nino_mean

    accs_raw = compute_acc_per_lead(preds_phys, trues_phys)
    accs_3ma  = compute_acc_3ma(preds_phys, trues_phys)
    eff_lead  = compute_effective_lead(accs_3ma, threshold=0.5)
    mean_3ma_acc = float(np.nanmean(accs_3ma))

    return mean_3ma_acc, list(accs_raw), accs_3ma, eff_lead

def append_epoch_skill_log(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([row_dict])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)

def add_per_lead_metrics(row_dict, prefix, accs_raw, accs_3ma):
    for i, v in enumerate(accs_raw, start=1):
        row_dict[f'{prefix}_raw_acc_lead_{i:02d}'] = float(v) if not np.isnan(v) else np.nan
    for i, v in enumerate(accs_3ma, start=1):
        row_dict[f'{prefix}_3ma_acc_lead_{i:02d}'] = float(v) if not np.isnan(v) else np.nan
    return row_dict

ALL_VAR_NAMES = ['sst', 'hc', 'mld', 'sss', 'slp', 'tauu', 'tauv']
CANONICAL_VAR_ORDER = ['sst', 'hc', 'mld', 'sss', 'slp', 'tauu', 'tauv']

def fix_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_variables(var_str: str) -> list:
    raw_list = [v.strip().lower() for v in var_str.split(',')]

    for v in raw_list:
        if v not in ALL_VAR_NAMES:
            raise ValueError(
                f"Unknown variable '{v}'. Valid options: {ALL_VAR_NAMES}"
            )
    if 'sst' not in raw_list:
        raise ValueError("SST must always be included in variables.")

    canonical = [v for v in CANONICAL_VAR_ORDER if v in raw_list]

    if set(canonical) != set(raw_list):
        missing   = set(raw_list) - set(canonical)
        unexpected = set(canonical) - set(raw_list)
        raise ValueError(
            f"Variable canonicalization failed. "
            f"Missing: {missing}, Unexpected: {unexpected}"
        )

    print(f"  [Canonical Order] Input: {raw_list} -> Normalized: {canonical}")
    return canonical

def build_var_config(var_names: list) -> dict:
    return {
        'var_names': var_names,
        'n_vars': len(var_names),
        'sst_idx': var_names.index('sst'),
    }

def validate_preprocessed_grid(lat, lon, name="dataset"):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    print(f"  [Grid] {name}: lat {lat.min():.1f}~{lat.max():.1f} n={len(lat)}; "
          f"lon {lon.min():.1f}~{lon.max():.1f} n={len(lon)}")
    if lat.min() > -59.9 or lat.max() < 59.9:
        print(f"  [Warning] {name}: latitude range is not full 60S-60N.")
    if not np.any((lon >= 190) & (lon <= 240)) and not np.any((lon >= -170) & (lon <= -120)):
        raise ValueError(f"{name}: longitude coordinate does not cover Nino3.4 region.")

def maybe_crop_lat(arrays, lat, lat_south=None, lat_north=None):
    if lat_south is None or lat_north is None:
        return arrays, lat
    mask = (lat >= lat_south) & (lat <= lat_north)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(
            f"No latitude points in [{lat_south}, {lat_north}]. "
            f"Lat range: [{lat.min():.1f}, {lat.max():.1f}]")
    lat_s = int(indices[0])
    lat_e = int(indices[-1]) + 1
    cropped_lat = lat[lat_s:lat_e]
    cropped_arrays = [arr[:, lat_s:lat_e, :] for arr in arrays]
    print(f"  [Optional Crop] Lat {lat.min():.1f}~{lat.max():.1f} -> "
          f"{cropped_lat.min():.1f}~{cropped_lat.max():.1f} "
          f"(H: {len(lat)} -> {len(cropped_lat)})")
    return cropped_arrays, cropped_lat

def da_to_numpy(da) -> np.ndarray:
    dims = list(da.dims)
    if all(d in dims for d in ['time', 'lat', 'lon']):
        da = da.transpose('time', 'lat', 'lon')
    arr = np.asarray(da.values, dtype=np.float32)
    if arr.ndim == 4:

        squeeze_axes = [i for i, s in enumerate(arr.shape) if s == 1]
        if squeeze_axes:
            arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"da_to_numpy: cannot handle dims={da.dims}, shape={arr.shape}")
    return arr

def model_forward(model, inputs, model_name,
                  init_month=None, tgt_spatial=None, wwv_gt=None):
    if model_name in PHYS_MODELS:
        if model_name in SPATIAL_TARGET_MODELS:
            preds, phys_loss = model(inputs, init_month=init_month,
                                     tgt_spatial=tgt_spatial)
        elif model_name in INIT_MONTH:
            preds, phys_loss = model(inputs, init_month)
        else:
            preds, phys_loss = model(inputs)
        return preds, phys_loss

    if model_name in INIT_MONTH:
        preds = model(inputs, init_month)
        return preds, None
    preds = model(inputs)
    return preds, None

def build_criterion(args):
    if getattr(args, 'model_name', '') not in {'PhysDualNet', 'PhysDualNet'}:
        raise ValueError('This clean package includes PhysDualNet only. Set --model_name PhysDualNet.')
    return PhysDualLoss(
        spb_weight=getattr(args, 'spb_weight', 2.0),
        corr_weight=getattr(args, 'corr_weight', 0.3),
        trend_weight=getattr(args, 'trend_weight', 0.1),
        var_weight=getattr(args, 'var_weight', 0.1),
    ).to(args.device)

def compute_loss(crit, preds, y, phys_loss, init_month, sample_weight):
    loss = crit(preds, y, init_month=init_month, sample_weight=sample_weight)
    if phys_loss is not None:
        loss = loss + phys_loss
    return loss

def _load_vars_from_ds(ds, var_names):
    loaded  = [ds[v] for v in var_names if v in ds]
    missing = [v for v in var_names if v not in ds]
    if missing:
        print(f"  [Warning] Missing variables in nc: {missing}")
    names = [v.name for v in loaded]
    shape = loaded[0].shape if loaded else "N/A"
    print(f"  -> Loaded variables: {names}, shape: {shape}")
    return loaded

def _require_vars(ds, var_names, dataset_name):
    missing = [v for v in var_names if v not in ds.data_vars]
    if missing:
        raise ValueError(f"{dataset_name}: missing variables {missing}; available={list(ds.data_vars)}")

def load_cmip6(path, var_names, select_models=None, lat_south=None, lat_north=None):
    ds = xr.open_dataset(path)
    print(f"  CMIP6 dims: {dict(ds.sizes)}")
    _require_vars(ds, var_names, 'CMIP6')

    model_dim = next((d for d in ds.dims if d in ('model', 'source_id', 'member_id')), None)
    if model_dim is None:
        raise ValueError(
            "Current pipeline expects CMIP6 file with a model/source_id/member_id dimension. "
            f"Got dims={dict(ds.sizes)}"
        )

    all_models = [str(m) for m in ds[model_dim].values]
    if select_models:
        valid_selected = [m for m in select_models if m in all_models]
        invalid = [m for m in select_models if m not in all_models]
        if invalid:
            print(f"  [Warning] 数据集中未找到以下模式，已忽略: {invalid}")
        if not valid_selected:
            raise ValueError(f"指定的模式 {select_models} 均不存在，请检查拼写！")
        ds = ds.sel({model_dim: valid_selected})
        model_names = valid_selected
        print(f"  [Models] selected {len(model_names)} models: {model_names}")
    else:
        model_names = all_models
        print(f"  [Models] using all {len(model_names)} models in file")

    n_models = ds.sizes[model_dim]
    T_per_model = ds.sizes['time']
    lat = ds['lat'].values.astype(np.float32)
    lon = ds['lon'].values.astype(np.float32)
    validate_preprocessed_grid(lat, lon, name='CMIP6')
    print(f"  Model dim '{model_dim}': {n_models} models x {T_per_model} months")

    arrays = []
    for v in var_names:
        da = ds[v].transpose(model_dim, 'time', 'lat', 'lon')
        vals = np.asarray(da.values, dtype=np.float32)
        vals = vals.reshape(n_models * T_per_model, vals.shape[-2], vals.shape[-1])
        arrays.append(vals)

    segments = [(i * T_per_model, (i + 1) * T_per_model) for i in range(n_models)]

    arrays, lat = maybe_crop_lat(arrays, lat, lat_south, lat_north)

    print(f"  CMIP6 arrays: {len(arrays)} vars, total_time={arrays[0].shape[0]}, "
          f"spatial={arrays[0].shape[1]}x{arrays[0].shape[2]}")
    ds.close()
    return arrays, segments, lat, lon, model_names

def load_obs(path, var_names, lat_south=None, lat_north=None, name='OBS'):
    ds = xr.open_dataset(path)
    print(f"  {name} dims: {dict(ds.sizes)}")
    _require_vars(ds, var_names, name)
    da_list = [ds[v].transpose('time', 'lat', 'lon') for v in var_names]
    arrays = [da_to_numpy(da) for da in da_list]
    lat = ds['lat'].values.astype(np.float32)
    lon = ds['lon'].values.astype(np.float32)
    validate_preprocessed_grid(lat, lon, name=name)
    time_da = da_list[0]
    print(f"  {name} time range: {str(time_da.time.values[0])[:7]} ~ "
          f"{str(time_da.time.values[-1])[:7]}  ({len(time_da.time)} months)")
    arrays, lat = maybe_crop_lat(arrays, lat, lat_south, lat_north)
    return arrays, lat, lon, time_da

def slice_obs(arrays, time_da, start_str, end_str):
    times   = pd.to_datetime([str(t)[:10] for t in time_da.time.values])
    t_start = pd.Timestamp(start_str) if start_str else times[0]
    t_end   = pd.Timestamp(end_str)   if end_str   else times[-1]
    mask    = (times >= t_start) & (times <= t_end)
    idxs    = np.where(mask)[0]
    if len(idxs) == 0:
        raise ValueError(f"Time slice [{start_str}, {end_str}] has no matches")
    s, e = idxs[0], idxs[-1] + 1
    return [arr[s:e] for arr in arrays], times[s:e]

def split_cmip6_segments(segments, val_months_per_model=24, window=36):
    min_val = max(val_months_per_model, window)
    if min_val != val_months_per_model:
        print(f"  [Info] val_months raised from {val_months_per_model} to {min_val}")
        val_months_per_model = min_val

    train_segs, val_segs = [], []
    n_tr, n_vl = 0, 0

    for (s, e) in segments:
        length = e - s
        if length <= val_months_per_model + window:
            train_segs.append((s, e))
            n_tr += length
        else:
            split_point = e - val_months_per_model
            train_segs.append((s, split_point))
            val_segs.append((split_point, e))
            n_tr += split_point - s
            n_vl += val_months_per_model

    if not val_segs:
        print("  [Warning] No model long enough for val split")
        longest = max(range(len(segments)), key=lambda i: segments[i][1] - segments[i][0])
        s, e    = segments[longest]
        split_point = e - val_months_per_model
        train_segs[longest] = (s, split_point)
        val_segs.append((split_point, e))
        n_tr = sum(te - ts for ts, te in train_segs)
        n_vl = val_months_per_model

    print(f"  CMIP6 train/val split: train {n_tr} months, val {n_vl} months")
    return train_segs, val_segs

def _parse_model_list(model_str: str) -> list:
    return [m.strip() for m in model_str.split(',') if m.strip()]

def _start_month_from_times(times) -> int:
    if times is None or len(times) == 0:
        return 0
    return int(pd.Timestamp(times[0]).month) - 1

def prepare_data(args):
    var_names = parse_variables(args.variables)
    result = {}

    if args.stage == 'train':
        print(f"[Data] Loading CMIP6 train data: {args.cmip_path}")
        select_models_list = _parse_model_list(args.select_models)

        cmip_arr, cmip_segs, lat, lon, model_names = load_cmip6(
            args.cmip_path, var_names,
            select_models_list if select_models_list else None,
            lat_south=args.lat_south, lat_north=args.lat_north)

        window = args.input_len + args.output_len
        if args.obs_val_path:
            train_segs = cmip_segs
            val_segs = None
        else:
            train_segs, val_segs = split_cmip6_segments(
                cmip_segs, val_months_per_model=args.cmip_val_months_per_model, window=window)

        val_arr = val_times = lat_val = lon_val = None
        if args.obs_val_path:
            print(f"\n[Data] Loading OBS validation data: {args.obs_val_path}")
            obs_val_arr, lat_val, lon_val, time_val_da = load_obs(
                args.obs_val_path, var_names,
                lat_south=args.lat_south, lat_north=args.lat_north,
                name='OBS-VAL')
            val_arr, val_times = slice_obs(obs_val_arr, time_val_da, args.obs_val_start, args.obs_val_end)
            print(f"  Validation set (OBS): {val_arr[0].shape[0]} months")

        print(f"\n[Data] Loading OBS test data: {args.obs_path}")
        obs_arr, lat_obs, lon_obs, time_da = load_obs(
            args.obs_path, var_names,
            lat_south=args.lat_south, lat_north=args.lat_north,
            name='OBS-TEST')
        test_arr, test_times = slice_obs(obs_arr, time_da, args.obs_start, args.obs_end)
        print(f"  Test set (OBS): {test_arr[0].shape[0]} months")

        result.update({
            'cmip_arr': cmip_arr,
            'cmip_segs': cmip_segs,
            'train_segs': train_segs,
            'val_segs': val_segs,
            'lat': lat,
            'lon': lon,
            'model_names': model_names,
            'val_arr': val_arr,
            'val_times': val_times,
            'lat_val': lat_val,
            'lon_val': lon_val,
            'test_arr': test_arr,
            'test_times': test_times,
            'lat_obs': lat_obs,
            'lon_obs': lon_obs,
        })

    elif args.stage == 'test':
        print(f"[Data] Test mode — loading OBS test data only: {args.obs_path}")
        obs_arr, lat_obs, lon_obs, time_da = load_obs(
            args.obs_path, var_names,
            lat_south=args.lat_south, lat_north=args.lat_north,
            name='OBS-TEST')
        test_arr, test_times = slice_obs(obs_arr, time_da, args.obs_start, args.obs_end)
        print(f"  Test set (OBS): {test_arr[0].shape[0]} months")
        result.update({
            'test_arr': test_arr,
            'test_times': test_times,
            'lat_obs': lat_obs,
            'lon_obs': lon_obs,
        })

    elif args.stage == 'predict':
        print(f"[Data] Predict mode — loading OBS data: {args.obs_path}")
        all_arr, lat, lon, time_da = load_obs(
            args.obs_path, var_names,
            lat_south=args.lat_south, lat_north=args.lat_north,
            name='OBS-PREDICT')
        if args.predict_input_end:
            input_end = pd.Timestamp(args.predict_input_end)
        else:
            times_pd = pd.to_datetime([str(t)[:10] for t in time_da.time.values])
            input_end = times_pd[-1]
        input_start = input_end - pd.DateOffset(months=args.input_len - 1)
        predict_arr, _ = slice_obs(
            all_arr, time_da,
            input_start.strftime('%Y-%m-%d'),
            input_end.strftime('%Y-%m-%d'))
        result.update({
            'all_arr': all_arr,
            'lat': lat,
            'lon': lon,
            'time_da': time_da,
            'predict_arr': predict_arr,
            'input_end': input_end,
        })

    return result

def main(args):
    fix_seed(args.seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.save_dir,   exist_ok=True)
    os.makedirs(args.visual_dir, exist_ok=True)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    var_names  = parse_variables(args.variables)
    var_config = build_var_config(var_names)
    args.var_config = var_config

    print(f"\n{'='*60}")
    print(f"  Stage: {args.stage.upper()}  |  Model: {args.model_name}  |  Device: {device}")
    print(f"  Variables ({len(var_names)}): {var_names}")
    print(f"  Input: {args.input_len}m -> Output: {args.output_len} Nino3.4 scalars")
    print(f"  OBS range: {args.obs_start} ~ {args.obs_end}")
    print(f"  Loss: lead_decay={args.lead_decay} spb_weight={args.spb_weight} "
          f"corr_weight={args.corr_weight} var_weight={getattr(args, 'var_weight', 0.1)} "
          f"trend_weight={getattr(args, 'trend_weight', 0.1)}")
    print(f"{'='*60}\n")

    exp_name  = args.model_name
    best_path = os.path.join(args.save_dir, f"zeroshot_{exp_name}.pth")

    data = prepare_data(args)

    if args.stage == 'train':
        cmip_arr           = data['cmip_arr']
        train_segs         = data['train_segs']
        val_segs           = data.get('val_segs')
        lat                = data['lat']
        lon                = data['lon']
        model_names        = data['model_names']
        val_arr            = data.get('val_arr')
        val_times          = data.get('val_times')
        lat_val            = data.get('lat_val')
        lon_val            = data.get('lon_val')
        test_arr           = data['test_arr']
        test_times         = data['test_times']
        lat_obs            = data['lat_obs']
        lon_obs            = data['lon_obs']

        args.lat_coords = lat
        args.lon_coords = lon

        need_spatial = args.model_name in SPATIAL_TARGET_MODELS

        cmip_start_month = args.cmip_start_month - 1

        tr_ds = ENSODataset(
            cmip_arr, args.input_len, args.output_len,
            lat=lat, lon=lon,
            is_train=True, stats=None, segments=train_segs,
            var_names=var_names, start_month=cmip_start_month,
            return_spatial_target=need_spatial)
        stats = tr_ds.get_stats()

        if val_arr is not None:
            val_start_month = _start_month_from_times(val_times)
            vl_ds = ENSODataset(
                val_arr, args.input_len, args.output_len,
                lat=lat_val, lon=lon_val,
                is_train=False, stats=stats, segments=None,
                var_names=var_names, start_month=val_start_month,
                model_weights=None,
                return_spatial_target=False)
            print(f"  [Validation] Using OBS validation file: {args.obs_val_path}")
        else:
            vl_ds = ENSODataset(
                cmip_arr, args.input_len, args.output_len,
                lat=lat, lon=lon,
                is_train=False, stats=stats, segments=val_segs,
                var_names=var_names, start_month=cmip_start_month,
                return_spatial_target=False)
            print("  [Validation] Using CMIP6 tail split")

        obs_start_month = _start_month_from_times(test_times)

        te_ds = ENSODataset(
            test_arr, args.input_len, args.output_len,
            lat=lat_obs, lon=lon_obs,
            is_train=False, stats=stats, segments=None,
            var_names=var_names, start_month=obs_start_month,
            model_weights=None,
            return_spatial_target=False)

        tr_loader = DataLoader(
            tr_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        vl_loader = DataLoader(
            vl_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        te_loader = DataLoader(
            te_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

        args.input_dim = tr_ds.n_vars
        args.img_height, args.img_width = tr_ds.spatial_shape

        print(f"  Train: {len(tr_ds)}, Val: {len(vl_ds)}, Test: {len(te_ds)}")
        print(f"  Variables: {tr_ds.n_vars}, Spatial: {args.img_height}x{args.img_width}")

    elif args.stage == 'test':
        test_arr   = data['test_arr']
        test_times = data['test_times']
        lat_obs    = data['lat_obs']
        lon_obs    = data['lon_obs']

        args.lat_coords = lat_obs
        args.lon_coords = lon_obs
        args.input_dim  = len(var_names)
        args.img_height = test_arr[0].shape[1]
        args.img_width  = test_arr[0].shape[2]
        print(f"  Variables: {args.input_dim}, Spatial: {args.img_height}x{args.img_width}")

    elif args.stage == 'predict':
        all_arr     = data['all_arr']
        lat         = data['lat']
        lon         = data['lon']
        time_da     = data['time_da']
        predict_arr = data['predict_arr']
        input_end   = data['input_end']

        args.lat_coords = lat
        args.lon_coords = lon
        args.input_dim  = len(predict_arr)
        args.img_height = predict_arr[0].shape[1]
        args.img_width  = predict_arr[0].shape[2]

    print(f"\n[Model] Building {args.model_name}...")
    model_dict = get_model_dict()
    model = model_dict[args.model_name](args).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.2f} M")

    if args.stage == 'train':
        crit     = build_criterion(args)
        val_crit = build_criterion(args)

        n_models = len(model_names)

        if args.model_name in SPATIAL_TARGET_MODELS:
            params_reg   = [p for n, p in model.named_parameters() if 'input_reg' in n]
            params_other = [p for n, p in model.named_parameters() if 'input_reg' not in n]
            opt = torch.optim.Adam([
                {"params": params_reg,   "weight_decay": 0},
                {"params": params_other, "weight_decay": 0.0001},
            ], lr=args.learning_rate)
        else:
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )

        warmup_epochs = 5
        def lr_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / warmup_epochs
            pct = (ep - warmup_epochs) / max(1, args.epochs - warmup_epochs)
            return 0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * pct))

        sched   = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        stopper = EarlyStopping(patience=args.patience, verbose=True, path=best_path)

        obs_best_path     = os.path.join(args.save_dir, f"zeroshot_{exp_name}_realval_best.pth")
        obs_eval_interval = getattr(args, 'obs_eval_interval', 5)
        test_eval_interval = getattr(args, 'test_eval_interval', 1)
        best_obs_acc      = -1.0
        epoch_skill_log_path = os.path.join(args.save_dir, f"epoch_skill_monitor_{exp_name}.csv")

        print(f"\n{'~'*55}")
        print(f"  [TRAIN]  lr={args.learning_rate:.1e}  epochs={args.epochs}"
              f"  patience={args.patience}")
        print(f"  [REAL-VAL] 每 {obs_eval_interval} 个 epoch 在验证集上评估一次")
        print(f"  [TEST-MONITOR] 每 {test_eval_interval} 个 epoch 在测试集 "
              f"({args.obs_start} ~ {args.obs_end}) 上计算 3MA-ACC / Effective Lead")
        print(f"  [TEST-MONITOR] 仅记录，不用于 early stopping 或 best checkpoint 选择")
        print(f"  [TEST-MONITOR] CSV -> {epoch_skill_log_path}")
        print(f"  [LearnableWeights] n_models={n_models}, prior from similarity scores")
        print(f"  [Models] {model_names}")
        print(f"{'~'*55}")

        global_step = 0
        for epoch in range(args.epochs):

            lr_now = opt.param_groups[0]['lr']

            if hasattr(model, 'set_phys_anneal'):
                model.set_phys_anneal(epoch)

            model.train()
            tr_loss = 0.0

            for batch in tqdm(tr_loader, desc=f"  E{epoch+1:03d} Train", leave=False):

                if len(batch) == 5:
                    if args.model_name in SPATIAL_TARGET_MODELS:
                        x, y, init_month, model_ids, tgt_spatial = batch
                        tgt_spatial = tgt_spatial.to(device)
                    else:
                        x, y, init_month, model_ids = batch
                        tgt_spatial = None
                else:
                    x, y, init_month, model_ids = batch
                    tgt_spatial = None

                x, y         = x.to(device), y.to(device)
                init_month   = init_month.to(device)
                model_ids    = model_ids.to(device)

                sample_weight = None

                opt.zero_grad()
                preds, phys = model_forward(
                    model, x, args.model_name, init_month,
                    tgt_spatial=tgt_spatial)

                loss = compute_loss(crit, preds, y, phys, init_month,
                                    sample_weight=sample_weight)

                total_loss = loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                tr_loss += total_loss.item()
                global_step += 1

            avg_tr = tr_loss / len(tr_loader)

            model.eval()
            vl_loss = 0.0
            with torch.no_grad():
                for x, y, init_month, _ in vl_loader:
                    x, y, init_month = x.to(device), y.to(device), init_month.to(device)
                    preds, _ = model_forward(model, x, args.model_name, init_month)
                    loss = val_crit(preds, y, init_month=init_month)
                    vl_loss += loss.item()
            avg_vl = vl_loss / len(vl_loader)

            obs_acc_str = ""
            test_acc_str = ""
            current_obs_3ma_acc = None
            realval_eff_lead = np.nan
            realval_accs_raw = None
            realval_accs_3ma = None

            if (epoch + 1) % obs_eval_interval == 0 or epoch == 0:
                obs_3ma_acc, accs_raw, accs_3ma, eff_lead = evaluate_on_obs(
                    model, vl_loader, args.model_name, stats, device)
                current_obs_3ma_acc = obs_3ma_acc
                realval_eff_lead = eff_lead
                realval_accs_raw = accs_raw
                realval_accs_3ma = accs_3ma
                obs_acc_str = f"  REALVAL_3MA_ACC={obs_3ma_acc:.4f} | REALVAL_Eff.Lead={eff_lead}m"

                if obs_3ma_acc > best_obs_acc:
                    best_obs_acc = obs_3ma_acc
                    torch.save({
                        'args': args,
                        'model_state_dict': model.state_dict(),
                        'stats': stats,
                        'realval_acc': obs_3ma_acc,
                        'realval_per_lead_acc': accs_raw,
                        'realval_per_lead_3ma': accs_3ma,
                        'eff_lead': eff_lead,
                        'epoch': epoch + 1,
                    }, obs_best_path)
                    obs_acc_str += f" (*NEW BEST*)"

            if test_eval_interval > 0 and ((epoch + 1) % test_eval_interval == 0 or epoch == 0):
                test_3ma_acc, test_accs_raw, test_accs_3ma, test_eff_lead = evaluate_on_obs(
                    model, te_loader, args.model_name, stats, device)
                test_acc_str = f"  TEST_3MA_ACC={test_3ma_acc:.4f} | TEST_Eff.Lead={test_eff_lead}m"

                log_row = {
                    'epoch': epoch + 1,
                    'lr': lr_now,
                    'train_loss': avg_tr,
                    'val_loss': avg_vl,
                    'realval_3ma_acc': float(current_obs_3ma_acc) if current_obs_3ma_acc is not None else np.nan,
                    'realval_eff_lead': realval_eff_lead,
                    'test_3ma_acc': float(test_3ma_acc),
                    'test_eff_lead': test_eff_lead,
                }
                if realval_accs_raw is not None and realval_accs_3ma is not None:
                    log_row = add_per_lead_metrics(log_row, 'realval', realval_accs_raw, realval_accs_3ma)
                else:
                    nan_leads = np.full(args.output_len, np.nan, dtype=np.float32)
                    log_row = add_per_lead_metrics(log_row, 'realval', nan_leads, nan_leads)
                log_row = add_per_lead_metrics(log_row, 'test', test_accs_raw, test_accs_3ma)
                append_epoch_skill_log(epoch_skill_log_path, log_row)

            sched.step()
            print(f"  E{epoch+1:03d} [lr={lr_now:.2e}]  "
                  f"Train={avg_tr:.4f}  Val={avg_vl:.4f}{obs_acc_str}{test_acc_str}")

            if current_obs_3ma_acc is not None:
                stopper(-current_obs_3ma_acc, model, args, stats=stats)

            if stopper.early_stop:
                print("  Early stopping triggered.")
                break

        print(f"\n  Val-best model saved: {best_path}")
        print(f"  Real-validation-best model saved: {obs_best_path}  (ACC={best_obs_acc:.4f})")

    if args.stage in ('train', 'test'):
        print(f"\n[Test] OBS blind evaluation ({args.obs_start} ~ {args.obs_end})...")

        if args.load_model_path:
            load_path = args.load_model_path
        elif args.stage == 'train' and os.path.exists(obs_best_path):
            load_path = obs_best_path
            print(f"  Using real-validation-best model")
        else:
            obs_best_fallback = os.path.join(
                args.save_dir, f"zeroshot_{exp_name}_realval_best.pth")
            if os.path.exists(obs_best_fallback):
                load_path = obs_best_fallback
                print(f"  Using real-validation-best model")
            else:
                load_path = best_path
                print(f"  Real-validation-best not found, falling back to val-best model")

        ckpt = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        stats = ckpt.get('stats', {})
        print(f"  Loaded model from: {load_path}")

        if args.stage == 'test':
            if not stats:
                raise RuntimeError(
                    "Checkpoint 中没有 stats，请确认使用了正确的 checkpoint 文件。")
            print(f"  [Test] Building te_ds from checkpoint stats...")
            te_ds = ENSODataset(
                test_arr, args.input_len, args.output_len,
                lat=lat_obs, lon=lon_obs,
                is_train=False, stats=stats, segments=None,
                var_names=var_names, start_month=_start_month_from_times(test_times), model_weights=None,
                return_spatial_target=False)
            te_loader = DataLoader(
                te_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            print(f"  Test samples: {len(te_ds)}")

        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch in tqdm(te_loader, desc="  Inference"):
                x, y, init_month = batch[0], batch[1], batch[2]
                x, y, init_month = x.to(device), y.to(device), init_month.to(device)
                preds, _ = model_forward(model, x, args.model_name, init_month)
                all_preds.append(preds.cpu())
                all_trues.append(y.cpu())

        all_preds = torch.cat(all_preds, 0).numpy()
        all_trues = torch.cat(all_trues, 0).numpy()

        vis_dir = os.path.join(args.visual_dir, f"zeroshot_{exp_name}_{timestamp}")
        os.makedirs(vis_dir, exist_ok=True)

        evaluate_nino34_skill_decay(all_preds, all_trues, stats, vis_dir, args)
        save_nino34_to_csv_and_plot(all_preds, all_trues, stats, vis_dir,
                                     args.output_len, test_times, args)
        plot_nino34_lead_correlation(
            all_preds_dict={args.model_name: all_preds},
            all_trues=all_trues, stats=stats,
            save_dir=vis_dir, args=args, test_times=test_times)
        plot_seasonal_lead_heatmap(all_preds, all_trues, stats, vis_dir, args, test_times)
        plot_init_month_lead_heatmap(all_preds, all_trues, stats, vis_dir, args, test_times)
        plot_extreme_event_case_study(all_preds, all_trues, stats, vis_dir, test_times, args)
        plot_power_spectrum(all_preds, all_trues, stats, vis_dir)

        print(f"\n  Results saved to: {vis_dir}")

    if args.stage == 'predict':
        print("\n[Predict] Rolling future prediction...")

        load_path = args.load_model_path or os.path.join(
            args.save_dir, f"zeroshot_{exp_name}.pth")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found: {load_path}")

        ckpt = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        stats = ckpt.get('stats', {})

        C  = len(predict_arr)
        H, W = predict_arr[0].shape[1], predict_arr[0].shape[2]
        inp_np = np.zeros((args.input_len, C, H, W), dtype=np.float32)
        for c, (vn, arr) in enumerate(zip(var_names, predict_arr)):
            mu  = float(stats.get(f'{vn}_mean', 0.0))
            sig = max(float(stats.get(f'{vn}_std', 1.0)), 1e-6)
            inp_np[:, c] = np.nan_to_num((arr - mu) / sig, nan=0.0)

        inp_tensor = torch.from_numpy(inp_np).unsqueeze(0).to(device)
        n_pred     = min(args.predict_months, args.output_len)
        base_date = (pd.Timestamp(args.predict_input_end)
                     if args.predict_input_end
                     else pd.Timestamp(str(time_da.time.values[-1])[:10]))
        init_month_tensor = torch.tensor([base_date.month - 1], dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            preds_out, _ = model_forward(model, inp_tensor, args.model_name, init_month_tensor)
        pred_norm     = preds_out[0, :n_pred].cpu().numpy()
        nino_mean     = stats.get('nino34_mean', 0.0)
        nino_std      = stats.get('nino34_std', 1.0)
        pred_physical = pred_norm * nino_std + nino_mean

        future_dates = [base_date + pd.DateOffset(months=m + 1) for m in range(n_pred)]
        print(f"  Predict range: {future_dates[0].strftime('%Y-%m')} -> "
              f"{future_dates[-1].strftime('%Y-%m')}")

        obs_nino34 = None
        obs_dates_plot = None
        try:
            times_all = pd.to_datetime([str(t)[:10] for t in time_da.time.values])
            last_t    = times_all[-1]
            if future_dates[0] <= last_t:
                ov_end = min(future_dates[-1], last_t)
                mask   = (times_all >= future_dates[0]) & (times_all <= ov_end)
                idxs   = np.where(mask)[0]
                if len(idxs) > 0:
                    sst_idx = var_names.index('sst')
                    sst_obs = all_arr[sst_idx][idxs]
                    lat_s, lat_e, lon_s, lon_e = find_nino34_indices(lat, lon)
                    lat_region = lat[lat_s:lat_e]
                    weights    = np.cos(np.deg2rad(lat_region))[:, np.newaxis]
                    w_sum      = float(weights.sum() * (lon_e - lon_s))
                    obs_nino34 = np.array([
                        np.nansum(sst_obs[t, lat_s:lat_e, lon_s:lon_e] * weights) / w_sum
                        for t in range(len(idxs))])
                    obs_dates_plot = [times_all[i] for i in idxs]
        except Exception as e:
            print(f"  [Warning] Failed to extract obs Nino3.4: {e}")

        dir_name = f"predict-{exp_name}-init{base_date.strftime('%Y-%m')}-lead{n_pred}m"
        vis_dir  = os.path.join(args.visual_dir, dir_name)
        if os.path.exists(vis_dir):
            vis_dir = f"{vis_dir}_{timestamp}"
        os.makedirs(vis_dir, exist_ok=True)

        run_all_predict_plots(
            nino_pred=pred_physical,
            future_dates=future_dates,
            save_dir=vis_dir,
            title_suffix=f"init {base_date.strftime('%Y-%m')}, lead 1-{n_pred}m",
            obs_nino34=obs_nino34,
            obs_dates=obs_dates_plot)

        print(f"\n  Prediction results saved to: {vis_dir}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Nino3.4 Scalar Prediction (12 months input -> 24 scalars output)')

    p.add_argument('--stage', required=True, choices=['train', 'test', 'predict'])

    p.add_argument('--cmip_path', default='../processed_ssta_data/cmip6_all_models_processed.nc')
    p.add_argument('--obs_val_path', default='../processed_ssta_data/obs_1958_1978_processed.nc',
                   help='真实验证集文件；设为空字符串则回退到 CMIP6 尾部分割验证')
    p.add_argument('--obs_path',  default='../processed_ssta_data/obs_1980_2021_processed.nc',
                   help='真实测试/预测数据文件')

    p.add_argument('--variables',  default='sst,hc,mld,slp,tauu,tauv',
                   help='默认使用统一预处理后的 7 个变量：sst,hc,mld,sss,slp,tauu,tauv')
    p.add_argument('--obs_val_start', default='1958-01-01')
    p.add_argument('--obs_val_end',   default='1978-12-31')
    p.add_argument('--obs_start',  default='1980-01-01')
    p.add_argument('--obs_end',    default='2021-12-31')
    p.add_argument('--cmip_val_months_per_model', type=int, default=0,
                   help='仅当 --obs_val_path 为空时生效；从每个 CMIP6 模式尾部划分多少个月作为验证')
    p.add_argument('--select_models', type=str, default='',
                   help='逗号分隔的 CMIP6 模式名；默认空字符串表示使用文件中的全部模式')

    p.add_argument('--save_dir',        default='./checkpoints/')
    p.add_argument('--visual_dir',      default='./results/')
    p.add_argument('--load_model_path', default=None)

    p.add_argument('--predict_input_end', default=None)
    p.add_argument('--predict_months',    type=int, default=24)

    p.add_argument('--model_name',  default='PhysDualNet')
    p.add_argument('--input_len',   type=int, default=12)
    p.add_argument('--output_len',  type=int, default=24)
    p.add_argument('--epochs',      type=int, default=200)
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--patience',    type=int, default=50)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed',        type=int, default=2025, help="271828,2025")
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--d_model',     type=int, default=96)
    p.add_argument('--heads',       type=int, default=4)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--d_var',       type=int, default=48)

    p.add_argument('--geo_patch_size', type=str, default=None,
                   help="Patch size for GeoformerIndex, e.g. '12,12'. Larger patches reduce memory.")
    p.add_argument('--geo_d_size', type=int, default=None,
                   help='Hidden size for GeoformerIndex. Default: use --d_model.')
    p.add_argument('--geo_heads', type=int, default=None,
                   help='Attention heads for GeoformerIndex. Default: use --heads.')
    p.add_argument('--geo_dim_feedforward', type=int, default=None,
                   help='Feed-forward dimension for GeoformerIndex. Default: use --dim_feedforward.')
    p.add_argument('--geo_dropout', type=float, default=None,
                   help='Dropout for GeoformerIndex. Default: use --dropout.')
    p.add_argument('--geo_num_encoder_layers', type=int, default=None,
                   help='Number of Geoformer encoder layers. Default: 4.')
    p.add_argument('--geo_num_decoder_layers', type=int, default=None,
                   help='Number of Geoformer decoder layers. Default: 4.')

    p.add_argument('--lead_decay',   type=float, default=0.0)
    p.add_argument('--spb_weight',   type=float, default=2.0)
    p.add_argument('--corr_weight',  type=float, default=0.3)
    p.add_argument('--var_weight',   type=float, default=0.1)
    p.add_argument('--trend_weight', type=float, default=0.1)

    p.add_argument('--lat_south', type=float, default=None,
                   help='可选纬度裁剪；统一预处理数据默认不裁剪')
    p.add_argument('--lat_north', type=float, default=None,
                   help='可选纬度裁剪；统一预处理数据默认不裁剪')

    p.add_argument('--obs_eval_interval', type=int, default=1,
                   help='训练中每隔多少个 epoch 在 1958-1978 OBS 验证集上计算 3MA-ACC；该指标用于保存 best/early stopping')
    p.add_argument('--test_eval_interval', type=int, default=1,
                   help='训练中每隔多少个 epoch 在 1980-2021 OBS 测试集上计算 3MA-ACC/有效预测月数；只监控记录，不参与保存和早停；设为0可关闭')

    p.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128, 64])
    p.add_argument('--kernel_size', type=int, nargs='+', default=[5, 5])
    p.add_argument('--n_layers',    type=int, default=4)

    p.add_argument('--l0_penalty',     type=float, default=0.1)
    p.add_argument('--emb_mse_weight', type=float, default=1.0)
    p.add_argument('--dim_feedforward', type=int, default=None)
    p.add_argument('--cmip_start_month', type=int, default=1,
                    help='CMIP6 第一条时间对应的月份，1=Jan, ..., 12=Dec')

    args = p.parse_args()
    if args.dim_feedforward is None:
        args.dim_feedforward = args.d_model
    if args.obs_val_path == '':
        args.obs_val_path = None
    main(args)
