from __future__ import annotations
import argparse, os
import pandas as pd

from .config import load_config
from .utils import ensure_dir, save_json
from .dataio import load_csv, resample_fill
from .features import add_time_features, add_lags_rollings
from .preprocessing import (
    train_val_test_split, fit_scalers, apply_scalers,
    CONTINUOUS_OBS, KNOWN_FUTURE, STATIC_REAL,
)


def _write_df(df: pd.DataFrame, path: str, fmt: str):
    ensure_dir(os.path.dirname(path) or ".")
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--in_csv", default=None, help="Nếu None dùng cfg.paths.data_csv")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--format", choices=["csv","parquet"], default="parquet")
    ap.add_argument("--scaled", action="store_true", help="Lưu thêm bản đã scale")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    in_csv = args.in_csv or cfg.paths.data_csv

    # 1) Load & resample
    df = load_csv(in_csv, timezone=cfg.timezone)
    df = resample_fill(df, cfg.frequency, cfg.timezone)

    # 2) Feature engineering (đồng nhất với train.py)
    df = add_time_features(df)
    df = add_lags_rollings(
        df,
        target="muc_thuong_luu",
        lags=cfg.lags_hours,
        roll_windows=cfg.roll_windows_hours,
        roll_stats=cfg.roll_stats,
    )

    # 3) Split theo thời gian (per-site)
    tr, va, te = train_val_test_split(df, cfg.split.train_ratio, cfg.split.val_ratio)

    # 4) Ghi RAW (đã FE) để audit
    _write_df(tr, f"{args.out_dir}/train.{args.format}", args.format)
    _write_df(va, f"{args.out_dir}/val.{args.format}", args.format)
    _write_df(te, f"{args.out_dir}/test.{args.format}", args.format)

    # 5) Fit scaler trên train và lưu metadata
    scalers = fit_scalers(tr)
    meta = {
        "enc_len": cfg.enc_len,
        "dec_len": cfg.dec_len,
        "frequency": cfg.frequency,
        "timezone": cfg.timezone,
        "quantiles": cfg.quantiles,
        "features": {
            "continuous_obs": [c for c in CONTINUOUS_OBS if c in df.columns],
            "known_future": [c for c in KNOWN_FUTURE if c in df.columns],
            "static_real": [c for c in STATIC_REAL if c in df.columns],
        },
    }
    ensure_dir(cfg.paths.models_dir)
    save_json(meta, os.path.join(cfg.paths.models_dir, "data_meta.json"))

    # 6) (Tùy chọn) Lưu bản đã scale để dùng trực tiếp khi cần
    if args.scaled:
        tr_s = apply_scalers(tr, scalers)
        va_s = apply_scalers(va, scalers)
        te_s = apply_scalers(te, scalers)
        _write_df(tr_s, f"{args.out_dir}/train_scaled.{args.format}", args.format)
        _write_df(va_s, f"{args.out_dir}/val_scaled.{args.format}", args.format)
        _write_df(te_s, f"{args.out_dir}/test_scaled.{args.format}", args.format)

    print("Saved splits to:", args.out_dir)


if __name__ == "__main__":
    main()