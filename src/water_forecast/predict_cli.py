from __future__ import annotations
import argparse, pandas as pd
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet

from .config import load_config
from .dataio import load_csv, resample_fill
from .features import add_time_features, add_lags_rollings
from .preprocessing import fit_scalers, apply_scalers, train_val_test_split
from .dataset import build_timeseries_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="models/tft-best.ckpt")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    df = load_csv(cfg.paths.data_csv, timezone=cfg.timezone)
    df = resample_fill(df, cfg.frequency, cfg.timezone)
    df = add_time_features(df)
    df = add_lags_rollings(df, "muc_thuong_luu", cfg.lags_hours, cfg.roll_windows_hours, cfg.roll_stats)

    tr, va, te = train_val_test_split(df, cfg.split.train_ratio, cfg.split.val_ratio)
    scalers = fit_scalers(tr)
    full = pd.concat([apply_scalers(x, scalers) for x in [tr, va, te]])

    training, _ = build_timeseries_datasets(full, cfg.enc_len, cfg.dec_len)
    model = TemporalFusionTransformer.load_from_checkpoint(args.ckpt)

    to_pred = TimeSeriesDataSet.from_dataset(training, full, predict=True, stop_randomization=True)
    dl = to_pred.to_dataloader(train=False, batch_size=cfg.batch_size)
    preds = model.predict(dl, mode="quantiles")  # [N,H,Q]
    print("Predictions shape:", preds.shape)

if __name__ == "__main__":
    main()