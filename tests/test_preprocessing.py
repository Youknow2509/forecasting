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