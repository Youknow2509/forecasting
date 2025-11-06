from __future__ import annotations
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from .config import load_config
from .utils import ensure_dir, save_json
from .dataio import load_csv, resample_fill
from .features import add_time_features, add_lags_rollings
from .preprocessing import train_val_test_split, fit_scalers, apply_scalers
from .dataset import build_timeseries_datasets
from .tft_model import make_tft


def main(cfg_path: str = "configs/default.yaml"):
    cfg = load_config(cfg_path)
    os.makedirs(cfg.paths.models_dir, exist_ok=True)
    os.makedirs(cfg.paths.artifacts_dir, exist_ok=True)

    # 1) Load & resample
    df = load_csv(cfg.paths.data_csv, timezone=cfg.timezone)
    df = resample_fill(df, freq=cfg.frequency, tz_name=cfg.timezone)

    # 2) Feature Engineering
    df = add_time_features(df)
    df = add_lags_rollings(
        df,
        target="muc_thuong_luu",
        lags=cfg.lags_hours,
        roll_windows=cfg.roll_windows_hours,
        roll_stats=cfg.roll_stats,
    )

    # 3) Split dataset
    train_df, val_df, test_df = train_val_test_split(df, cfg.split.train_ratio, cfg.split.val_ratio)

    # 4) Fit scalers trên train và apply cho tất cả
    scalers = fit_scalers(train_df)
    train_df_s = apply_scalers(train_df, scalers)
    val_df_s   = apply_scalers(val_df, scalers)
    test_df_s  = apply_scalers(test_df, scalers)
    full_df_s  = pd.concat([train_df_s, val_df_s, test_df_s], ignore_index=True)

    # 5) Build TimeSeries datasets
    training, validation = build_timeseries_datasets(full_df_s, cfg.enc_len, cfg.dec_len)

    # 6) DataLoaders
    train_loader = training.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=4)
    val_loader   = validation.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=4)

    # 7) Model (LightningModule hợp lệ)
    model = make_tft(
        training,
        hidden_size=cfg.hidden_size,
        attention_heads=cfg.attention_heads,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
        quantiles=tuple(cfg.quantiles)
    )

    # 8) Trainer
    early_stop = EarlyStopping(monitor="val_loss", patience=cfg.patience, mode="min")
    ckpt = ModelCheckpoint(
        dirpath=cfg.paths.models_dir,
        filename="tft-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop, ckpt, lrmon],
        gradient_clip_val=0.1,
        deterministic=True,
        log_every_n_steps=50,
    )

    # 9) Fit model
    trainer.fit(model, train_loader, val_loader)

    # 10) Save metadata
    meta = {
        "enc_len": cfg.enc_len,
        "dec_len": cfg.dec_len,
        "quantiles": cfg.quantiles,
        "frequency": cfg.frequency,
        "timezone": cfg.timezone,
    }
    save_json(meta, cfg.paths.metadata_json)
    print("Best checkpoint saved at:", ckpt.best_model_path)


if __name__ == "__main__":
    main()
