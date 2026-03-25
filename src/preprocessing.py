import os
import psutil
import warnings

import numpy as np
import pandas as pd

from .config import OUTDIR

warnings.filterwarnings('ignore')


def ensure_output_dir() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def print_header() -> None:
    print('=' * 70)
    print('🚀 GC_final_v2 — 45.88 Base + Temp×UV Features')
    print('=' * 70)
    print('  ✅ temp_avg 사용 (temp_a + temp_b 평균)')
    print('  ✅ UV 중간 1칸 보간 추가')
    print('  ✅ fillna 최소화')
    print('  ✅ nins 산점도 저장')
    print('=' * 70)


def print_memory_usage(tag: str = '') -> None:
    mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f'📊 MEM[{tag}]: {mem_gb:.2f} GB')


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col == 'time' or df[col].dtype == object:
            continue

        cmin, cmax = df[col].min(), df[col].max()
        dtype_name = str(df[col].dtype)

        if dtype_name.startswith('int'):
            if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df


def normalize_missing(df: pd.DataFrame, exclude=('time', 'pv_id', 'type')) -> pd.DataFrame:
    df = df.copy()
    null_tokens = {'', ' ', 'NA', 'NaN', 'NULL', 'None', 'null', 'nan'}

    for col in df.columns:
        if col in exclude:
            continue

        if df[col].dtype == object:
            df[col] = df[col].replace(list(null_tokens), np.nan)
            df[col] = pd.to_numeric(df[col], errors='ignore')

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def midpoint_only_fill(series: pd.Series) -> pd.Series:
    """각 결측 구간의 중앙 1칸만 양쪽 평균으로 채운다."""
    s = series.copy()
    isna = s.isna().to_numpy()
    if not isna.any():
        return s

    diff = np.diff(np.pad(isna.astype(int), (1, 1)))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    for start, end in zip(starts, ends):
        left_idx = start - 1
        right_idx = end + 1
        if left_idx < 0 or right_idx >= len(s):
            continue

        left_val = s.iloc[left_idx]
        right_val = s.iloc[right_idx]
        if pd.isna(left_val) or pd.isna(right_val):
            continue

        mid = (start + end) // 2
        s.iloc[mid] = (float(left_val) + float(right_val)) / 2.0

    return s


def apply_midpoint_fill(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    for col in ['temp_a', 'temp_b']:
        if col in train.columns:
            train[f'{col}_original'] = train[col]
            train[col] = train.groupby('pv_id')[col].transform(midpoint_only_fill)
            before = train[f'{col}_original'].isna().sum()
            after = train[col].isna().sum()
            print(f'🛟 train {col} NaN: before={before:,} → after={after:,} (filled={before - after:,})')

        if col in test.columns:
            test[f'{col}_original'] = test[col]
            test[col] = test.groupby('pv_id')[col].transform(midpoint_only_fill)
            before = test[f'{col}_original'].isna().sum()
            after = test[col].isna().sum()
            print(f'🛟 test  {col} NaN: before={before:,} → after={after:,} (filled={before - after:,})')

    if 'uv_idx' in train.columns:
        train['uv_idx_original'] = train['uv_idx']
        train['uv_idx'] = train.groupby('pv_id')['uv_idx'].transform(midpoint_only_fill)
        before = train['uv_idx_original'].isna().sum()
        after = train['uv_idx'].isna().sum()
        print(f'🛟 train uv_idx NaN: before={before:,} → after={after:,} (filled={before - after:,})')

    if 'uv_idx' in test.columns:
        test['uv_idx_original'] = test['uv_idx']
        test['uv_idx'] = test.groupby('pv_id')['uv_idx'].transform(midpoint_only_fill)
        before = test['uv_idx_original'].isna().sum()
        after = test['uv_idx'].isna().sum()
        print(f'🛟 test  uv_idx NaN: before={before:,} → after={after:,} (filled={before - after:,})')

    return train, test
