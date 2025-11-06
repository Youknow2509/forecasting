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