from __future__ import annotations
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch
try:
    from lightning.pytorch import seed_everything
except ImportError:
    from pytorch_lightning import seed_everything
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer

from .config import load_config
from .utils import ensure_dir, load_json
from .dataio import load_csv, resample_fill
from .features import add_time_features, add_lags_rollings
from .preprocessing import train_val_test_split, fit_scalers, apply_scalers
from .dataset import build_timeseries_datasets

try:
    import properscoring as ps
except Exception:
    ps = None


def pinball(y, q_pred, q):
    e = y - q_pred
    return np.maximum(q*e, (q-1)*e)


def main(cfg_path: str = "configs/default.yaml", ckpt_path: str | None = None):
    cfg = load_config(cfg_path)
    seed_everything(cfg.seed)

    meta = load_json(cfg.paths.metadata_json)
    qtiles = np.array(meta["quantiles"]) if "quantiles" in meta else np.array(cfg.quantiles)

    # Load and preprocess exactly like training
    df = load_csv(cfg.paths.data_csv, timezone=cfg.timezone)
    df = resample_fill(df, freq=cfg.frequency, tz_name=cfg.timezone)
    df = add_time_features(df)
    df = add_lags_rollings(df, target="muc_thuong_luu", lags=cfg.lags_hours, roll_windows=cfg.roll_windows_hours, roll_stats=cfg.roll_stats)

    tr, va, te = train_val_test_split(df, cfg.split.train_ratio, cfg.split.val_ratio)
    scalers = fit_scalers(tr)
    tr_s, va_s, te_s = apply_scalers(tr, scalers), apply_scalers(va, scalers), apply_scalers(te, scalers)
    full = pd.concat([tr_s, va_s, te_s], ignore_index=True)

    training, validation = build_timeseries_datasets(full, cfg.enc_len, cfg.dec_len)
    if ckpt_path is None:
        ckpt_path = cfg.paths.best_ckpt

    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)

    # Build test set aligned to validation creation
    test = TimeSeriesDataSet.from_dataset(training, full, predict=True, stop_randomization=True)
    test_loader = test.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=4)

    preds, idx = model.predict(test_loader, return_index=True, mode="quantiles")
    # preds shape: [N, horizon, n_quantiles]
    q10, q50, q90 = np.moveaxis(preds, -1, 0)  # [N, H]

    # Collect actuals aligned (use from_dataset reconstruction)
    y = model.to_prediction(test_loader)["target"]  # fallback: may be None on some versions
    if y is None:
        # reconstruct actuals from underlying dataset
        y = test.get_target()
    y = y.numpy()

    # Compute metrics
    mae = np.mean(np.abs(y - q50))
    rmse = np.sqrt(np.mean((y - q50)**2))

    # Coverage of 80% interval
    inside = (y >= q10) & (y <= q90)
    coverage = inside.mean()

    # Horizon-wise MAE (first 24h, 7d, 14d)
    H = q50.shape[1]
    def horizon_slice(h):
        h = min(h, H)
        return np.mean(np.abs(y[:, :h] - q50[:, :h]))
    mae_24 = horizon_slice(24)
    mae_7d = horizon_slice(24*7)
    mae_14d = horizon_slice(24*14)

    # CRPS if available (approx via quantiles, or proper via properscoring if full samples existed)
    if ps is not None:
        crps = np.mean(ps.crps_ensemble(y, np.stack([q10, q50, q90], axis=-1)))
    else:
        # pinball average as proxy
        crps = np.mean([pinball(y, q10, 0.1), pinball(y, q50, 0.5), pinball(y, q90, 0.9)])

    os.makedirs(cfg.paths.artifacts_dir, exist_ok=True)
    with open(os.path.join(cfg.paths.artifacts_dir, "metrics.txt"), "w") as f:
        f.write(f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nCoverage80={coverage:.3f}\nMAE@24h={mae_24:.4f}\nMAE@7d={mae_7d:.4f}\nMAE@14d={mae_14d:.4f}\nCRPS~={crps:.4f}\n")

    # Plot 3 example windows
    import matplotlib.pyplot as plt
    for i in range(3):
        plt.figure()
        plt.title(f"Sample {i} forecast (median & 80% band)")
        plt.plot(y[i], label="actual")
        plt.plot(q50[i], label="q50")
        plt.fill_between(range(H), q10[i], q90[i], alpha=0.3, label="[q10,q90]")
        plt.legend(); plt.xlabel("horizon step"); plt.ylabel("scaled level")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.paths.artifacts_dir, f"forecast_{i}.png"))
        plt.close()

    print("Saved metrics and plots to:", cfg.paths.artifacts_dir)


if __name__ == "__main__":
    main()