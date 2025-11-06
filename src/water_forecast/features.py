from __future__ import annotations
import numpy as np, pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"].dt
    df["hour"] = ts.hour
    df["dow"] = ts.dayofweek
    df["doy"] = ts.dayofyear
    # cyclical encodings
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*df["hour"]/24), np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"], df["dow_cos"]   = np.sin(2*np.pi*df["dow"]/7),   np.cos(2*np.pi*df["dow"]/7)
    df["doy_sin"], df["doy_cos"]   = np.sin(2*np.pi*df["doy"]/366), np.cos(2*np.pi*df["doy"]/366)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def add_lags_rollings(df: pd.DataFrame, target: str, lags: list[int], roll_windows: list[int], roll_stats: list[str]) -> pd.DataFrame:
    by = ["site_id"]
    for L in lags:
        df[f"{target}_lag_{L}"] = df.groupby(by)[target].shift(L)
    for W in roll_windows:
        g = df.groupby(by)[target]
        if "mean" in roll_stats:
            df[f"{target}_rolling_mean_{W}"] = g.shift(1).rolling(W, min_periods=max(1, W//3)).mean()
        if "std" in roll_stats:
            df[f"{target}_rolling_std_{W}"] = g.shift(1).rolling(W, min_periods=max(1, W//3)).std()
    return df