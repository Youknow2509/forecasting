from __future__ import annotations
import pandas as pd, numpy as np
from dateutil import tz


def load_csv(path: str, timezone: str = "Asia/Bangkok") -> pd.DataFrame:
    df = pd.read_csv(path)
    # required columns
    assert "timestamp" in df.columns, "missing timestamp"
    if "site_id" not in df.columns:
        df["site_id"] = "SITE_001"
    # parse time
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)
    df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    return df


def resample_fill(df: pd.DataFrame, freq: str = "1H", tz_name: str = "Asia/Bangkok") -> pd.DataFrame:
    out = []
    for sid, g in df.groupby("site_id", sort=False):
        g = g.set_index("timestamp").asfreq(freq.lower())
        # track missing
        was_nan = g.isna()
        # interpolate numeric columns linearly
        num_cols = g.select_dtypes(include=["float", "int"]).columns
        g[num_cols] = g[num_cols].interpolate(method="time", limit=48, limit_direction="both")
        # g[num_cols] = g[num_cols].fillna(method="ffill").fillna(method="bfill")
        g[num_cols] = g[num_cols].ffill().bfill()
        g["was_imputed"] = was_nan.any(axis=1).astype(int)
        g["site_id"] = sid
        g = g.reset_index()
        out.append(g)
    out = pd.concat(out, ignore_index=True)
    # ensure tz
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(tz_name)

    return out.sort_values(["site_id", "timestamp"]).reset_index(drop=True)