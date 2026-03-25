import numpy as np
import pandas as pd

from .config import (
    GEO_MIN_DAY_RATIO,
    NIGHT_ELEV_THRESH_DEG,
    POST_DAY_SMOOTH,
    POST_DAY_SMOOTH_WIN,
    POST_NIGHT_ZERO,
    PROXY_DAY_THR,
)


def apply_postprocessing(test_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    y = preds.copy().astype(np.float32)

    if POST_NIGHT_ZERO:
        if 'solar_elev_deg' in test_df.columns:
            elev = test_df['solar_elev_deg'].astype(float).values
            geo_day = elev > float(NIGHT_ELEV_THRESH_DEG)
            proxy_day = test_df['sun_elevation_proxy'].astype(float).values > float(PROXY_DAY_THR)
            print(f'[Solar] elev_deg stats: min={np.nanmin(elev):.2f}, max={np.nanmax(elev):.2f}, geo-day={geo_day.mean():.2%}')
            use_geo = geo_day.mean() >= float(GEO_MIN_DAY_RATIO)
            day_mask = geo_day if use_geo else proxy_day
            print(f"[NightZero] method={'geo' if use_geo else 'proxy'}, day_ratio={day_mask.mean():.2%}")
            y[~day_mask] = 0.0
        else:
            proxy_day = test_df['sun_elevation_proxy'].astype(float).values > float(PROXY_DAY_THR)
            print('[NightZero] solar_elev_deg 없음 → proxy 사용')
            y[~proxy_day] = 0.0

    if POST_DAY_SMOOTH and 'pv_id' in test_df.columns:
        if 'solar_elev_deg' in test_df.columns:
            elev = test_df['solar_elev_deg'].astype(float).values
            geo_day = elev > float(NIGHT_ELEV_THRESH_DEG)
            proxy_day = test_df['sun_elevation_proxy'].astype(float).values > float(PROXY_DAY_THR)
            use_geo = geo_day.mean() >= float(GEO_MIN_DAY_RATIO)
            day_mask_all = geo_day if use_geo else proxy_day
        else:
            day_mask_all = test_df['sun_elevation_proxy'].astype(float).values > float(PROXY_DAY_THR)

        y_series = pd.Series(y)
        pv_series = test_df['pv_id'].astype(str).values
        idx = np.arange(len(y))

        for _, group_idx in pd.Series(idx).groupby(pv_series):
            group_positions = group_idx.values
            tmp = y_series.iloc[group_positions].copy()
            day_mask = day_mask_all[group_positions]
            tmp[~day_mask] = np.nan
            smoothed = tmp.rolling(POST_DAY_SMOOTH_WIN, center=True, min_periods=1).mean()
            y_series.iloc[group_positions] = np.where(day_mask, smoothed.fillna(tmp).values, y_series.iloc[group_positions].values)

        y = y_series.values

    return np.maximum(0.0, y)
