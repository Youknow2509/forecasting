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