import numpy as np
import torch

def _resolve_lon_mask(lon: np.ndarray, lon_lo_360: float, lon_hi_360: float):
    if np.any(lon > 180):

        return (lon >= lon_lo_360) & (lon <= lon_hi_360)
    else:

        lo = lon_lo_360 if lon_lo_360 <= 180 else lon_lo_360 - 360
        hi = lon_hi_360 if lon_hi_360 <= 180 else lon_hi_360 - 360
        if lo <= hi:
            return (lon >= lo) & (lon <= hi)
        else:

            return (lon >= lo) | (lon <= hi)

def find_region_indices(lat: np.ndarray, lon: np.ndarray,
                        lat_lo: float, lat_hi: float,
                        lon_lo_360: float, lon_hi_360: float):
    lat_mask = (lat >= lat_lo) & (lat <= lat_hi)
    lat_indices = np.where(lat_mask)[0]
    if len(lat_indices) == 0:
        raise ValueError(
            f"No latitude points in [{lat_lo}, {lat_hi}]. "
            f"Lat range: [{lat.min():.1f}, {lat.max():.1f}]")
    lat_s = int(lat_indices[0])
    lat_e = int(lat_indices[-1]) + 1

    lon_mask = _resolve_lon_mask(lon, lon_lo_360, lon_hi_360)
    lon_indices = np.where(lon_mask)[0]
    if len(lon_indices) == 0:
        raise ValueError(
            f"No longitude points in [{lon_lo_360}, {lon_hi_360}] (0-360). "
            f"Lon range: [{lon.min():.1f}, {lon.max():.1f}]")
    lon_s = int(lon_indices[0])
    lon_e = int(lon_indices[-1]) + 1

    return lat_s, lat_e, lon_s, lon_e

def find_nino34_indices(lat: np.ndarray, lon: np.ndarray):
    return find_region_indices(lat, lon, -5, 5, 190, 240)

def make_region_mask(lat: np.ndarray, lon: np.ndarray,
                     lat_lo: float, lat_hi: float,
                     lon_lo_360: float, lon_hi_360: float) -> torch.Tensor:
    lat_m = (lat >= lat_lo) & (lat <= lat_hi)
    lon_m = _resolve_lon_mask(lon, lon_lo_360, lon_hi_360)
    return torch.from_numpy(np.outer(lat_m, lon_m).astype(np.float32))

def nino34_mask_bool(lat: np.ndarray, lon: np.ndarray):
    lat_mask = (lat >= -5) & (lat <= 5)
    if np.any(lon > 180):
        lon_mask = (lon >= 190) & (lon <= 240)
    else:
        lon_mask = (lon >= -170) & (lon <= -120)
    return lat_mask, lon_mask

KEY_REGIONS = {
    'Nino1+2':        {'lat': (-10, 0),    'lon_360': (270, 280)},
    'Nino3':          {'lat': (-5, 5),     'lon_360': (210, 270)},
    'Nino3.4':        {'lat': (-5, 5),     'lon_360': (190, 240)},
    'Nino4':          {'lat': (-5, 5),     'lon_360': (160, 210)},
    'WP_WarmPool':    {'lat': (-10, 10),   'lon_360': (120, 160)},
    'SP_SPCZ':        {'lat': (-30, -5),   'lon_360': (150, 220)},
    'IO_East':        {'lat': (-10, 10),   'lon_360': (80, 100)},
    'IO_West':        {'lat': (-10, 10),   'lon_360': (40, 70)},
    'NP_PMM':         {'lat': (15, 30),    'lon_360': (170, 230)},
    'SouthernOcean':  {'lat': (-60, -40),  'lon_360': (0, 360)},
    'TA_Atlantic':    {'lat': (0, 15),     'lon_360': (320, 360)},
    'WP_BarrierLay':  {'lat': (-5, 15),    'lon_360': (140, 180)},
}

def get_region_mask(lat: np.ndarray, lon: np.ndarray, region_name: str):
    if region_name not in KEY_REGIONS:
        raise ValueError(f"Unknown region '{region_name}'. Valid: {list(KEY_REGIONS.keys())}")
    r = KEY_REGIONS[region_name]
    lat_lo, lat_hi = r['lat']
    lon_lo, lon_hi = r['lon_360']
    lat_mask = (lat >= lat_lo) & (lat <= lat_hi)
    lon_mask = _resolve_lon_mask(lon, lon_lo, lon_hi)
    return lat_mask[:, np.newaxis] & lon_mask[np.newaxis, :]
