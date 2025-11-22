from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


CONTINUOUS_OBS = [
    "muc_thuong_luu","luu_luong_den","tong_luong_xa","xa_tran","xa_nha_may",
    "so_cua_xa_sau","so_cua_xa_mat","was_imputed"
]
KNOWN_FUTURE = ["rain_forecast_mm", "planned_release_m3s", "hour_sin","hour_cos","dow_sin","dow_cos","doy_sin","doy_cos","is_weekend"]
STATIC_REAL = ["muc_dang_binh_thuong", "muc_chet"]


def train_val_test_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    for sid, g in df.groupby("site_id", sort=False):
        n = len(g)
        i1 = int(n * train_ratio)
        i2 = int(n * (train_ratio + val_ratio))
        parts.append((g.iloc[:i1], g.iloc[i1:i2], g.iloc[i2:]))
    train = pd.concat([p[0] for p in parts])
    val   = pd.concat([p[1] for p in parts])
    test  = pd.concat([p[2] for p in parts])
    return train, val, test


def fit_scalers(train_df: pd.DataFrame) -> Dict[str, StandardScaler]:
    scalers = {}
    for name, cols in {
        "obs": CONTINUOUS_OBS,
        "static": STATIC_REAL,
        "known": [c for c in KNOWN_FUTURE if c in train_df.columns]
    }.items():
        cols = [c for c in cols if c in train_df.columns]
        if not cols:
            continue
        sc = StandardScaler()
        sc.fit(train_df[cols].astype(float))
        scalers[name] = sc
    return scalers


def apply_scalers(df: pd.DataFrame, scalers: Dict[str, StandardScaler]) -> pd.DataFrame:
    out = df.copy()
    if "obs" in scalers:
        cols = [c for c in CONTINUOUS_OBS if c in out.columns]
        out[cols] = scalers["obs"].transform(out[cols].astype(float))
    if "static" in scalers:
        cols = [c for c in STATIC_REAL if c in out.columns]
        out[cols] = scalers["static"].transform(out[cols].astype(float))
    if "known" in scalers:
        cols = [c for c in KNOWN_FUTURE if c in out.columns]
        out[cols] = scalers["known"].transform(out[cols].astype(float))
    return out


def inverse_transform_target(values, scalers: Dict[str, StandardScaler], target_col: str = "muc_thuong_luu"):
    """
    Chuyển giá trị dự đoán từ dạng chuẩn hóa về thang đo gốc.
    
    Args:
        values: numpy array hoặc pandas Series - giá trị đã chuẩn hóa
        scalers: dict của scalers
        target_col: tên cột target cần inverse transform
    
    Returns:
        numpy array - giá trị đã inverse transform về thang đo gốc
    """
    import numpy as np
    
    if "obs" not in scalers:
        return values
    
    # Lấy scaler và tìm index của target column
    obs_scaler = scalers["obs"]
    obs_cols = [c for c in CONTINUOUS_OBS if c in obs_scaler.feature_names_in_]
    
    if target_col not in obs_cols:
        return values
    
    target_idx = obs_cols.index(target_col)
    
    # Lấy mean và scale của target column
    mean = obs_scaler.mean_[target_idx]
    scale = obs_scaler.scale_[target_idx]
    
    # Inverse transform: X_original = X_scaled * scale + mean
    return values * scale + mean