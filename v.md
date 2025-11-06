# Temporal Fusion Transformer (TFT) — Water Level Forecasting (Multi‑Horizon, Quantiles)

Below is a **ready‑to‑run Python project** implementing your design doc: preprocessing, dataset builder, TFT training with quantile loss, evaluation (MAE/RMSE/coverage/CRPS*), inference API (FastAPI), and deployment (Dockerfile).

> Assumptions: time zone = `Asia/Bangkok`, frequency = hourly (resampling supported), group id = `site_id` (single site also OK). Quantiles = `[0.1, 0.5, 0.9]`. Horizon H = 336–672 configurable.

---

## Project layout

```
TFT-WaterLevel/
├─ README.md
├─ requirements.txt
├─ Makefile
├─ Dockerfile
├─ configs/
│  └─ default.yaml
├─ data/
│  ├─ sample.csv                # optional tiny sample to validate pipeline shape
│  └─ schema.json               # column specs & units
├─ artifacts/                   # saved figures, reports, predictions
├─ models/                      # checkpoints, scalers, metadata
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ dataio.py
│  ├─ preprocessing.py
│  ├─ features.py
│  ├─ dataset.py
│  ├─ tft_model.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ predict_cli.py
│  ├─ export_onnx.py
│  └─ utils.py
├─ service/
│  ├─ app.py                    # FastAPI
│  └─ schemas.py
└─ tests/
   ├─ __init__.py
   └─ test_preprocessing.py
```

---

## requirements.txt

```txt
# Core
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
pyyaml>=6.0.1
pytz
python-dateutil

# Torch stack (CPU by default; see Dockerfile for CUDA alternative)
torch>=2.3
pytorch-lightning>=2.4
pytorch-forecasting>=1.0.0  

# Viz & reports
matplotlib>=3.9
seaborn>=0.13

# API & monitoring
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.8
prometheus-client>=0.20

# Metrics (optional CRPS)
properscoring>=0.1.3

# Logging & tracking (optional)
mlflow>=2.16
```

---

## configs/default.yaml

```yaml
# ===== Data =====
seed: 3407
timezone: "Asia/Bangkok"
frequency: "1H"

# Required columns (see data/schema.json)
columns:
  timestamp: timestamp
  site_id: category
  muc_thuong_luu: float
  muc_dang_binh_thuong: float
  muc_chet: float
  luu_luong_den: float
  tong_luong_xa: float
  xa_tran: float
  xa_nha_may: float
  so_cua_xa_sau: int
  so_cua_xa_mat: int
  # optional/known-future
  rain_forecast_mm: float
  planned_release_m3s: float

# Split (time-ordered)
split:
  train_ratio: 0.7
  val_ratio: 0.15
  # test = remainder

# Windowing
enc_len: 168           # 7 days history
dec_len: 336           # 14 days horizon (set 672 for 4 weeks)

# Quantiles & loss
quantiles: [0.1, 0.5, 0.9]

# Feature engineering
lags_hours: [1, 3, 6, 24, 48, 168]
roll_windows_hours: [3, 12, 24, 168]
roll_stats: ["mean", "std"]

# Training
batch_size: 64
max_epochs: 150
hidden_size: 160
attention_heads: 4
dropout: 0.1
learning_rate: 0.001
patience: 10

# Logging
mlflow:
  enabled: false
  tracking_uri: "file:./mlruns"
  experiment: "tft-waterlevel"

# Paths
paths:
  data_csv: "data/sample.csv"
  artifacts_dir: "artifacts"
  models_dir: "models"
  best_ckpt: "models/best.ckpt"
  metadata_json: "models/metadata.json"
```

---

## data/schema.json

```json
{
  "timestamp": {"type": "datetime", "tz": "Asia/Bangkok", "freq": "1H"},
  "site_id": {"type": "category", "required": true},
  "muc_thuong_luu": {"type": "float", "unit": "m", "role": "target"},
  "muc_dang_binh_thuong": {"type": "float", "unit": "m", "role": "static"},
  "muc_chet": {"type": "float", "unit": "m", "role": "static"},
  "luu_luong_den": {"type": "float", "unit": "m3/s", "role": "observed"},
  "tong_luong_xa": {"type": "float", "unit": "m3/s", "role": "observed"},
  "xa_tran": {"type": "float", "unit": "m3/s", "role": "observed"},
  "xa_nha_may": {"type": "float", "unit": "m3/s", "role": "observed"},
  "so_cua_xa_sau": {"type": "int", "role": "observed"},
  "so_cua_xa_mat": {"type": "int", "role": "observed"},
  "rain_forecast_mm": {"type": "float", "role": "known_future", "optional": true},
  "planned_release_m3s": {"type": "float", "role": "known_future", "optional": true}
}
```

---

## src/config.py

```python
from __future__ import annotations
import os, yaml, random, numpy as np, torch
from dataclasses import dataclass

@dataclass
class CFG:
    d: dict
    def __getattr__(self, item):
        v = self.d.get(item)
        if isinstance(v, dict):
            return CFG(v)
        return v


def load_config(path: str = "configs/default.yaml") -> CFG:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # set seeds
    seed = d.get("seed", 3407)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return CFG(d)
```

---

## src/utils.py

```python
from __future__ import annotations
import json, os
from pathlib import Path

def ensure_dir(p: str | Path) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
```

---

## src/dataio.py

```python
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
        g = g.set_index("timestamp").asfreq(freq)
        # track missing
        was_nan = g.isna()
        # interpolate numeric columns linearly
        num_cols = g.select_dtypes(include=["float", "int"]).columns
        g[num_cols] = g[num_cols].interpolate(method="time", limit=48, limit_direction="both")
        g[num_cols] = g[num_cols].fillna(method="ffill").fillna(method="bfill")
        g["was_imputed"] = was_nan.any(axis=1).astype(int)
        g["site_id"] = sid
        g = g.reset_index()
        out.append(g)
    out = pd.concat(out, ignore_index=True)
    # ensure tz
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(tz_name, nonexistent="shift_forward", ambiguous="NaT")
    return out.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
```

---

## src/features.py

```python
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
```

---

## src/preprocessing.py

```python
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
        "known": [c for c in KNOWN_FUTURE if train_df.columns.contains(c)]
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
```

---

## src/dataset.py

```python
from __future__ import annotations
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


def build_timeseries_datasets(
    df: pd.DataFrame,
    enc_len: int,
    dec_len: int,
    target: str = "muc_thuong_luu",
):
    # integer time index per group
    df = df.sort_values(["site_id", "timestamp"]).copy()
    df["time_idx"] = df.groupby("site_id").cumcount()

    # Identify feature groups
    static_categoricals = ["site_id"]
    static_reals = [c for c in ["muc_dang_binh_thuong", "muc_chet"] if c in df.columns]

    time_varying_known_reals = [c for c in [
        "rain_forecast_mm", "planned_release_m3s",
        "hour_sin","hour_cos","dow_sin","dow_cos","doy_sin","doy_cos","is_weekend",
    ] if c in df.columns]

    # observed/unknown future (used in encoder only)
    tv_unknown_reals = [c for c in df.columns if c.startswith(target+"_lag_") or c.startswith(target+"_rolling_")]
    base_obs = [c for c in [
        "muc_thuong_luu","luu_luong_den","tong_luong_xa","xa_tran","xa_nha_may",
        "so_cua_xa_sau","so_cua_xa_mat","was_imputed"
    ] if c in df.columns]
    time_varying_unknown_reals = sorted(set([target] + base_obs + tv_unknown_reals))

    max_encoder_length = enc_len
    max_prediction_length = dec_len

    training_cutoff = df["time_idx"].max() - max_prediction_length

    common_kwargs = dict(
        time_idx="time_idx",
        target=target,
        group_ids=["site_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["site_id"], transformation="standard"),
        add_relative_time_idx=True,
        allow_missing_timesteps=True,
    )

    training = TimeSeriesDataSet(df[df.time_idx <= training_cutoff], **common_kwargs)
    validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff + 1)
    return training, validation
```

---

## src/tft_model.py

```python
from __future__ import annotations
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


def make_tft(training_dataset,
             hidden_size=160,
             attention_heads=4,
             dropout=0.1,
             learning_rate=1e-3,
             quantiles=(0.1, 0.5, 0.9)):
    loss = QuantileLoss(quantiles=quantiles)
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        dropout=dropout,
        loss=loss,
        log_interval=50,
        output_size=len(quantiles),
        reduce_on_plateau_patience=4,
    )
    return tft
```

---

## src/train.py

```python
from __future__ import annotations
import os, json
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

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

    # 2) FE
    df = add_time_features(df)
    df = add_lags_rollings(
        df,
        target="muc_thuong_luu",
        lags=cfg.lags_hours,
        roll_windows=cfg.roll_windows_hours,
        roll_stats=cfg.roll_stats,
    )

    # 3) Split (time-ordered per site)
    train_df, val_df, test_df = train_val_test_split(df, cfg.split.train_ratio, cfg.split.val_ratio)

    # 4) Fit scalers ONLY on train, then transform all
    scalers = fit_scalers(train_df)
    train_df_s = apply_scalers(train_df, scalers)
    val_df_s   = apply_scalers(val_df, scalers)
    full_df_s  = pd.concat([train_df_s, val_df_s, apply_scalers(test_df, scalers)], ignore_index=True)

    # 5) Build TimeSeries datasets (training set decides encoding)
    training, validation = build_timeseries_datasets(full_df_s, cfg.enc_len, cfg.dec_len)

    # 6) DataLoaders
    train_loader = training.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=4)
    val_loader   = validation.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=4)

    # 7) Model
    model = make_tft(training, cfg.hidden_size, cfg.attention_heads, cfg.dropout, cfg.learning_rate, tuple(cfg.quantiles))

    # 8) Trainer
    early_stop = EarlyStopping(monitor="val_loss", patience=cfg.patience, mode="min")
    ckpt = ModelCheckpoint(dirpath=cfg.paths.models_dir, filename="tft-best", monitor="val_loss", mode="min", save_top_k=1)
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

    trainer.fit(model, train_loader, val_loader)

    # Save artifacts
    meta = {
        "enc_len": cfg.enc_len,
        "dec_len": cfg.dec_len,
        "quantiles": cfg.quantiles,
        "frequency": cfg.frequency,
        "timezone": cfg.timezone,
    }
    save_json(meta, cfg.paths.metadata_json)
    print("Best ckpt:", ckpt.best_model_path)


if __name__ == "__main__":
    main()
```

---

## src/evaluate.py

```python
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer

from .config import load_config
from .utils import ensure_dir, load_json
from .dataio import load_csv, resample_fill
from .features import add_time_features, add_lags_rollings
from .preprocessing import train_val_test_split, fit_scalers, apply_scalers
from .dataset import build_timeseries_datasets

# --- CRPS: optional properscoring, else quantile-based fallback (Cách 1) ---
try:
    import properscoring as ps  # sẽ không có nếu đã bỏ khỏi requirements
except Exception:
    ps = None


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> np.ndarray:
    e = y - q_pred
    return np.maximum(q * e, (q - 1.0) * e)


def crps_from_quantiles(y: np.ndarray, q_preds: np.ndarray, q_levels) -> float:
    """
    Approximate CRPS = ∫_0^1 pinball_τ(y, q_τ) dτ using midpoint weights.
    y       : (N, H)
    q_preds : (K, N, H) with K = len(q_levels), in ascending q order
    q_levels: iterable of K quantiles, ascending (e.g., [0.1, 0.5, 0.9])
    """
    q_levels = np.asarray(q_levels, dtype=float)
    edges = np.concatenate([[0.0], (q_levels[1:] + q_levels[:-1]) / 2.0, [1.0]])
    w = np.diff(edges)  # (K,) weights, sum to 1
    loss = 0.0
    for k, tau in enumerate(q_levels):
        e = y - q_preds[k]
        rho = np.maximum(tau * e, (tau - 1.0) * e)
        loss += w[k] * rho
    return float(np.mean(loss))


def main(cfg_path: str = "configs/default.yaml", ckpt_path: str | None = None):
    cfg = load_config(cfg_path)
    pl.seed_everything(cfg.seed)

    # Load and preprocess exactly like training
    df = load_csv(cfg.paths.data_csv, timezone=cfg.timezone)
    df = resample_fill(df, freq=cfg.frequency, tz_name=cfg.timezone)
    df = add_time_features(df)
    df = add_lags_rollings(
        df,
        target="muc_thuong_luu",
        lags=cfg.lags_hours,
        roll_windows=cfg.roll_windows_hours,
        roll_stats=cfg.roll_stats,
    )

    # Split and scale (fit on train only)
    tr, va, te = train_val_test_split(df, cfg.split.train_ratio, cfg.split.val_ratio)
    scalers = fit_scalers(tr)
    tr_s, va_s, te_s = apply_scalers(tr, scalers), apply_scalers(va, scalers), apply_scalers(te, scalers)
    full = pd.concat([tr_s, va_s, te_s], ignore_index=True)

    training, _ = build_timeseries_datasets(full, cfg.enc_len, cfg.dec_len)

    if ckpt_path is None:
        ckpt_path = cfg.paths.best_ckpt
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)

    # Build test set aligned to training dataset encoding
    test = TimeSeriesDataSet.from_dataset(training, full, predict=True, stop_randomization=True)
    test_loader = test.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=4)

    # Predictions (quantiles) + collect actuals from loader
    preds, index = model.predict(test_loader, return_index=True, mode="quantiles")  # (N, H, Q)
    preds = np.asarray(preds)
    q10, q50, q90 = np.moveaxis(preds, -1, 0)  # each (N, H)

    ys = []
    for bx, by in iter(test_loader):
        # by is a tuple: (target, weight) usually; take target
        ys.append(by[0].detach().cpu())
    y = torch.cat(ys, dim=0).numpy()  # (N, H)

    # Metrics
    mae = float(np.mean(np.abs(y - q50)))
    rmse = float(np.sqrt(np.mean((y - q50) ** 2)))
    coverage = float(((y >= q10) & (y <= q90)).mean())

    H = q50.shape[1]
    def h_mae(h):
        h = min(h, H)
        return float(np.mean(np.abs(y[:, :h] - q50[:, :h])))

    mae_24 = h_mae(24)
    mae_7d = h_mae(24 * 7)
    mae_14d = h_mae(24 * 14)

    qtiles = np.array(cfg.quantiles, dtype=float)
    if ps is not None:
        # Use properscoring if installed (optional)
        crps = float(np.mean(ps.crps_ensemble(y, np.stack([q10, q50, q90], axis=-1))))
    else:
        # Fallback (Cách 1): quantile-based approximation
        q_stack = np.stack([q10, q50, q90], axis=0)
        crps = crps_from_quantiles(y, q_stack, q_levels=qtiles)

    # Save artifacts
    os.makedirs(cfg.paths.artifacts_dir, exist_ok=True)
    with open(os.path.join(cfg.paths.artifacts_dir, "metrics.txt"), "w") as f:
        f.write(
            f"MAE={mae:.6f}
RMSE={rmse:.6f}
Coverage80={coverage:.6f}
"
            f"MAE@24h={mae_24:.6f}
MAE@7d={mae_7d:.6f}
MAE@14d={mae_14d:.6f}
CRPS~={crps:.6f}
"
        )

    # Plots for 3 samples
    for i in range(min(3, y.shape[0])):
        plt.figure()
        plt.title(f"Sample {i} forecast (median & 80% band)")
        plt.plot(y[i], label="actual")
        plt.plot(q50[i], label="q50")
        plt.fill_between(np.arange(H), q10[i], q90[i], alpha=0.3, label="[q10,q90]")
        plt.legend(); plt.xlabel("horizon step"); plt.ylabel("scaled level")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.paths.artifacts_dir, f"forecast_{i}.png"))
        plt.close()

    print("Saved metrics and plots to:", cfg.paths.artifacts_dir)


if __name__ == "__main__":
    main()
```

---

## src/predict_cli.py

```python
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
```

---

## service/schemas.py

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Row(BaseModel):
    timestamp: str
    site_id: str = "SITE_001"
    muc_thuong_luu: float
    muc_dang_binh_thuong: float
    muc_chet: float
    luu_luong_den: float
    tong_luong_xa: float
    xa_tran: float
    xa_nha_may: float
    so_cua_xa_sau: int
    so_cua_xa_mat: int
    rain_forecast_mm: Optional[float] = None
    planned_release_m3s: Optional[float] = None

class InferenceRequest(BaseModel):
    history: List[Row]  # past encoder window
    known_future: List[Row]  # future decoder window (length = dec_len)

class InferenceResponse(BaseModel):
    quantiles: List[float]
    forecasts: List[List[float]]  # per-quantile series for horizon (Q x H)
```

---

## service/app.py

```python
from __future__ import annotations
import os, pandas as pd, numpy as np
from fastapi import FastAPI
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet

from .schemas import InferenceRequest, InferenceResponse
from ..src.config import load_config
from ..src.dataio import resample_fill
from ..src.features import add_time_features, add_lags_rollings
from ..src.preprocessing import fit_scalers, apply_scalers
from ..src.dataset import build_timeseries_datasets

app = FastAPI(title="TFT WaterLevel Forecaster")

CFG = load_config("configs/default.yaml")
MODEL = TemporalFusionTransformer.load_from_checkpoint(CFG.paths.best_ckpt)

@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    # Build DF from history + future (known_future may have NaNs for unknown observed)
    hist = pd.DataFrame([r.model_dump() for r in req.history])
    fut  = pd.DataFrame([r.model_dump() for r in req.known_future])
    df = pd.concat([hist, fut], ignore_index=True)
    # Resample/align (assumes already at proper freq; asfreq to be safe)
    df = df.sort_values(["site_id", "timestamp"])  # strings ok; convert later
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = resample_fill(df, CFG.frequency, CFG.timezone)

    # FE
    df = add_time_features(df)
    df = add_lags_rollings(df, target="muc_thuong_luu", lags=CFG.lags_hours, roll_windows=CFG.roll_windows_hours, roll_stats=CFG.roll_stats)

    # Scale using history portion only (simple approach)
    hlen = len(hist)
    scalers = fit_scalers(df.iloc[:hlen])
    df_s = apply_scalers(df, scalers)

    # Dataset from training template
    training, _ = build_timeseries_datasets(df_s, CFG.enc_len, CFG.dec_len)
    new_ds = TimeSeriesDataSet.from_dataset(training, df_s, predict=True, stop_randomization=True)
    dl = new_ds.to_dataloader(train=False, batch_size=1)
    qpred = MODEL.predict(dl, mode="quantiles")  # [N,H,Q]
    qpred = np.moveaxis(qpred[0], -1, 0).tolist()  # Q x H for first item

    return InferenceResponse(
        quantiles=list(CFG.quantiles),
        forecasts=qpred,
    )
```

---

## src/export_onnx.py

```python
from __future__ import annotations
import torch
from pytorch_forecasting.models import TemporalFusionTransformer

# NOTE: TFT is complex; ONNX export may fail depending on ops. This script
# attempts TorchScript as a portable fallback.

def main(ckpt="models/tft-best.ckpt"):
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    ts_model = model.to_torchscript(method="script")
    torch.jit.save(ts_model, "models/tft-best.ts")
    print("Saved TorchScript to models/tft-best.ts")

if __name__ == "__main__":
    main()
```

---

## tests/test_preprocessing.py

```python
from __future__ import annotations
import pandas as pd
from src.preprocessing import train_val_test_split, fit_scalers, apply_scalers


def test_split_and_scale():
    # minimal synthetic
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="H", tz="Asia/Bangkok"),
        "site_id": ["S1"]*n,
        "muc_thuong_luu": range(n),
        "muc_dang_binh_thuong": 10.0,
        "muc_chet": 1.0,
        "luu_luong_den": 5.0,
        "tong_luong_xa": 1.0,
        "xa_tran": 0.0,
        "xa_nha_may": 1.0,
        "so_cua_xa_sau": 2,
        "so_cua_xa_mat": 1,
        "was_imputed": 0,
    })
    tr, va, te = train_val_test_split(df, 0.7, 0.15)
    sc = fit_scalers(tr)
    tr2 = apply_scalers(tr, sc)
    assert abs(tr2["muc_thuong_luu"].mean()) < 1.0  # roughly standardized
```

---

## README.md

````md
# TFT Water Level Forecasting (Multi-Horizon with Quantiles)

Implements your design doc using **PyTorch Forecasting** (TFT), delivering:
- Data prep & feature engineering (lags, rolling stats, cyclical time)
- TimeSeriesDataSet for encoder/decoder windows
- Temporal Fusion Transformer with QuantileLoss `[0.1, 0.5, 0.9]`
- Evaluation: MAE/RMSE, horizon slices, coverage, CRPS*
- Serving API (FastAPI) returning multi-horizon quantile forecasts
- Dockerfile for deployment

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --cfg configs/default.yaml
python -m src.evaluate --cfg configs/default.yaml --ckpt models/tft-best.ckpt
uvicorn service.app:app --host 0.0.0.0 --port 8000
````

### Data

Provide a CSV with columns defined in `data/schema.json`. If `site_id` is missing, a default `SITE_001` is used. Time must be parseable to UTC; it will be converted to `Asia/Bangkok`.

### Config

Adjust `configs/default.yaml`:

* `enc_len`, `dec_len`: history and horizon
* `lags_hours`, `roll_windows_hours`: lag/rolling windows
* `quantiles`: output distribution quantiles

### API

POST `/predict` with JSON:

```json
{
  "history": [ {"timestamp": "2025-01-01T00:00:00Z", "site_id": "S1", "muc_thuong_luu": 50.1, "muc_dang_binh_thuong": 55, "muc_chet": 40, "luu_luong_den": 120, "tong_luong_xa": 60, "xa_tran": 0, "xa_nha_may": 60, "so_cua_xa_sau": 2, "so_cua_xa_mat": 1 } ],
  "known_future": [ {"timestamp": "2025-01-08T00:00:00Z", "site_id": "S1", "muc_thuong_luu": 0, "muc_dang_binh_thuong": 55, "muc_chet": 40, "luu_luong_den": 0, "tong_luong_xa": 0, "xa_tran": 0, "xa_nha_may": 0, "so_cua_xa_sau": 2, "so_cua_xa_mat": 1, "rain_forecast_mm": 5, "planned_release_m3s": 50 } ]
}
```

Returns quantiles and horizon arrays.

### Notes

* Scaling is fit on train history only (production: persist scalers per site/version).
* CRPS via `properscoring` uses ensemble approximation from quantiles.
* For multi-site, keep `site_id` per row; GroupNormalizer scales per site.

```
make train     # run training
make eval      # evaluate & plots
make serve     # run API
```

````

---

## Makefile
```makefile
.PHONY: train eval serve
train:
	python -m src.train --cfg configs/default.yaml

eval:
	python -m src.evaluate --cfg configs/default.yaml --ckpt models/tft-best.ckpt

serve:
	uvicorn service.app:app --host 0.0.0.0 --port 8000
````

---

## Dockerfile

```dockerfile
# CPU build (simple). For CUDA, see comments below.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose API
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ===== CUDA (optional) =====
# Use an NVIDIA CUDA base image and install torch with matching cu* version, e.g.:
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
# RUN pip install --no-cache-dir torch==2.3.* --index-url https://download.pytorch.org/whl/cu121
# (Then install the rest of requirements)
```

---

## data/sample.csv (toy shape — replace with real)

```csv
timestamp,site_id,muc_thuong_luu,muc_dang_binh_thuong,muc_chet,luu_luong_den,tong_luong_xa,xa_tran,xa_nha_may,so_cua_xa_sau,so_cua_xa_mat,rain_forecast_mm,planned_release_m3s
2025-01-01T00:00:00Z,S1,50.2,55,40,120,60,0,60,2,1,1,50
2025-01-01T01:00:00Z,S1,50.1,55,40,118,60,0,60,2,1,1,50
2025-01-01T02:00:00Z,S1,50.3,55,40,119,60,0,60,2,1,1,50
```

---

### Implementation Notes mapping to your doc

* **Quantile loss & outputs**: `QuantileLoss([0.1,0.5,0.9])`, TFT `output_size=3`.
* **Encoder/decoder**: `enc_len`, `dec_len` configurable; unknown reals used in encoder only.
* **Feature groups**: static reals, known‑future reals (rain forecast, planned release, time encodings), observed reals (flows, gates) + lags/rolling.
* **Walk‑forward CV**: add a second script if needed; current split is time‑ordered + validation; extend to rolling splits easily by iterating cutoffs and reusing `build_timeseries_datasets`.
* **Monitoring**: expose Prometheus in `service/app.py` if desired (simple to add counters/gauges).
* **Retraining**: schedule a cron to rerun `src/train.py` and rotate `models/tft-best.ckpt`.

---

### Next steps

1. Replace `data/sample.csv` with your historical dataset (2–3 years).
2. Tune `enc_len/dec_len`, `hidden_size`, and lags/rollings.
3. If you have external hydrologic forecasts, populate `rain_forecast_mm`/`planned_release_m3s` for the decoder window.
4. (Optional) enable MLflow in config for experiment tracking.

---

# NEW: ETL service — crawl → training CSV (theo config hiện tại)

Các file crawl theo tháng như `2020-01.csv, 2020-02.csv, ...` có cột tiếng Việt sẽ được chuyển thành một CSV duy nhất (chuẩn schema của dự án) để train. Dịch vụ dưới đây xử lý: gộp file, parse thời gian `dd/MM HH:mm` (**không có năm** → lấy năm từ tên file), chuẩn hóa timezone (Asia/Bangkok → UTC), đổi tên cột, ép kiểu, khử trùng lặp `(site_id, timestamp)` và lưu ra `data/sample.csv`.

## src/ingest.py

```python
from __future__ import annotations
import argparse, os
from pathlib import Path
import pandas as pd
import re

VN2EN = {
    "ten_ho": "site_id",
    "muc_nuoc_thuong_luu": "muc_thuong_luu",
    "muc_nuoc_dang_binh_thuong": "muc_dang_binh_thuong",
    "muc_nuoc_chet": "muc_chet",
    "luu_luong_den_ho": "luu_luong_den",
    "tong_luong_xa": "tong_luong_xa",
    "tong_luong_xa_qua_dap_tran": "xa_tran",
    "tong_luong_xa_qua_nha_may": "xa_nha_may",
    "so_cua_xa_sau": "so_cua_xa_sau",
    "so_cua_xa_mat": "so_cua_xa_mat",
}

KEEP_COLS = [
    "timestamp","site_id","muc_thuong_luu","muc_dang_binh_thuong","muc_chet",
    "luu_luong_den","tong_luong_xa","xa_tran","xa_nha_may","so_cua_xa_sau","so_cua_xa_mat"
]


def _infer_year_from_name(name: str) -> int:
    m = re.search(r"(20[0-9]{2})[-_]?[0-9]{2}", name)
    if not m:
        m = re.search(r"(20[0-9]{2})", name)
    if not m:
        raise ValueError(f"Cannot infer year from filename: {name}")
    return int(m.group(1))


def _parse_timestamp_series(raw_ts: pd.Series, year: int, tz: str) -> pd.Series:
    # raw like '01/01 00:00' meaning dd/MM HH:mm (VN)
    ts = pd.to_datetime(raw_ts, format="%d/%m %H:%M", errors="coerce")
    if ts.isna().any():
        ts2 = pd.to_datetime(raw_ts.str.replace(" ", "", regex=False), format="%d/%m%H:%M", errors="coerce")
        ts = ts.fillna(ts2)
    ts = ts.map(lambda x: x.replace(year=year) if pd.notna(x) else x)
    ts = ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    return ts


def read_one_csv(path: str, tz: str) -> pd.DataFrame:
    year = _infer_year_from_name(os.path.basename(path))
    df = pd.read_csv(path)
    # Standardize headers
    cols = {c: VN2EN.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    assert "timestamp" in df.columns, "missing 'timestamp' column in crawl file"
    assert "site_id" in df.columns, "missing 'ten_ho' column in crawl file"

    # Build UTC timestamp from local representation
    df["timestamp"] = _parse_timestamp_series(df["timestamp"].astype(str), year, tz)

    # Force dtypes and create missing numeric columns as zeros
    num_map = {
        "muc_thuong_luu": float,
        "muc_dang_binh_thuong": float,
        "muc_chet": float,
        "luu_luong_den": float,
        "tong_luong_xa": float,
        "xa_tran": float,
        "xa_nha_may": float,
        "so_cua_xa_sau": "Int64",
        "so_cua_xa_mat": "Int64",
    }
    for c, dt in num_map.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(dt)
        else:
            df[c] = pd.Series([0]*len(df), dtype=dt if isinstance(dt, str) else float)

    # Select & order, dedup
    df = df[KEEP_COLS]
    df = df.drop_duplicates(subset=["site_id", "timestamp"], keep="last").sort_values(["site_id","timestamp"]).reset_index(drop=True)
    return df


def ingest_crawl_dir(crawl_dir: str, out_path: str, tz: str = "Asia/Bangkok") -> str:
    paths = sorted([str(p) for p in Path(crawl_dir).glob("*.csv")])
    if not paths:
        raise FileNotFoundError(f"No CSV files in {crawl_dir}")
    frames = [read_one_csv(p, tz) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["site_id","timestamp"], keep="last").sort_values(["site_id","timestamp"]).reset_index(drop=True)
    # ISO UTC strings
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl_dir", required=True)
    ap.add_argument("--out", default="data/sample.csv")
    ap.add_argument("--tz", default="Asia/Bangkok")
    args = ap.parse_args()

    p = ingest_crawl_dir(args.crawl_dir, args.out, args.tz)
    print("Wrote:", p)

if __name__ == "__main__":
    main()
```

## service/ingest_app.py (FastAPI, tùy chọn)

```python
from __future__ import annotations
import os
from typing import List
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from ..src.ingest import read_one_csv, ingest_crawl_dir

app = FastAPI(title="TFT ETL Service (crawl → training CSV)")

@app.post("/ingest/dir")
def ingest_dir(crawl_dir: str = Form(...), out: str = Form("data/sample.csv"), tz: str = Form("Asia/Bangkok")):
    try:
        p = ingest_crawl_dir(crawl_dir, out, tz)
        return {"ok": True, "output": p}
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

@app.post("/ingest/upload")
async def ingest_upload(files: List[UploadFile] = File(...), tz: str = Form("Asia/Bangkok")):
    frames = []
    for f in files:
        content = await f.read()
        tmp_path = Path("/tmp") / f.filename
        with open(tmp_path, "wb") as w:
            w.write(content)
        frames.append(read_one_csv(str(tmp_path), tz))
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if not frames:
        return JSONResponse(status_code=400, content={"ok": False, "error": "no files"})
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["site_id","timestamp"], keep="last").sort_values(["site_id","timestamp"]).reset_index(drop=True)
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    tmp = "/tmp/processed.csv"
    out.to_csv(tmp, index=False)
    return FileResponse(tmp, filename="processed.csv", media_type="text/csv")
```

### Cách dùng nhanh

* CLI: `python -m src.ingest --crawl_dir ./crawl --out data/sample.csv --tz Asia/Bangkok`
* API ETL: `uvicorn service.ingest_app:app --host 0.0.0.0 --port 9000`

  * `POST /ingest/dir` với JSON form: `crawl_dir`, `out` (tùy chọn), `tz` (tùy chọn)
  * `POST /ingest/upload` (multipart) gửi nhiều file CSV → trả về `processed.csv`

**Sau khi sinh `data/sample.csv`, chạy train bình thường:**

```
python -m src.train --cfg configs/default.yaml
```

---

## NEW: Tạo bộ **training/validation/test** sẵn (giữ đúng kiến trúc dữ liệu hiện tại)

Script này đọc `data/sample.csv` (tạo từ ETL), thực hiện **resample + FE + lags/rolling** đúng pipeline, **chia time-ordered per site** (train/val/test), **fit scaler trên train** rồi **ghi ra đĩa** theo định dạng bạn chọn. Mục tiêu: tạo ra artefacts dữ liệu cố định để kiểm định/lưu version.

### src/make_splits.py

```python
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
```

### Makefile bổ sung

```makefile
splits:
	python -m src.make_splits --cfg configs/default.yaml --out_dir data/processed --format parquet --scaled
```

### Cách dùng

```bash
# 1) ETL từ crawl → một CSV huấn luyện chuẩn
python -m src.ingest --crawl_dir ./crawl --out data/sample.csv --tz Asia/Bangkok

# 2) Sinh bộ train/val/test cố định (có cả bản scaled nếu cần)
python -m src.make_splits --cfg configs/default.yaml --out_dir data/processed --format parquet --scaled

# 3) Huấn luyện như cũ (train.py vẫn có thể tự split nội bộ)
python -m src.train --cfg configs/default.yaml
```

**Ghi chú:** Script này **giữ nguyên kiến trúc dữ liệu hiện tại** (resample + FE + lags/rolling + scaler fit on train). Việc lưu thêm `data_meta.json` giúp đảm bảo bạn dùng đúng `enc_len/dec_len` và danh mục features ở các bước sau (đánh giá, serving).
