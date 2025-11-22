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
    """
    Xây dựng TimeSeriesDataSet cho PyTorch Forecasting.

    - Điền NaN trong các cột lag/rolling.
    - Sắp xếp theo site và timestamp.
    - Thêm time_idx nếu chưa có.
    """
    # 1) Sắp xếp và tạo time_idx nếu chưa có
    if "time_idx" not in df.columns:
        df = df.sort_values(["site_id", "timestamp"]).copy()
        df["time_idx"] = df.groupby("site_id").cumcount()
    else:
        df = df.copy()

    # 2) Xác định các feature groups
    static_categoricals = ["site_id"]
    static_reals = [c for c in ["muc_dang_binh_thuong", "muc_chet"] if c in df.columns]

    time_varying_known_reals = [c for c in [
        "rain_forecast_mm", "planned_release_m3s",
        "hour_sin","hour_cos","dow_sin","dow_cos","doy_sin","doy_cos","is_weekend",
    ] if c in df.columns]

    # observed/unknown future
    tv_unknown_reals = [c for c in df.columns if c.startswith(target+"_lag_") or c.startswith(target+"_rolling_")]
    base_obs = [c for c in [
        "muc_thuong_luu","luu_luong_den","tong_luong_xa","xa_tran","xa_nha_may",
        "so_cua_xa_sau","so_cua_xa_mat","was_imputed"
    ] if c in df.columns]
    time_varying_unknown_reals = sorted(set([target] + base_obs + tv_unknown_reals))

    # 3) Điền NaN trong các cột lag/rolling
    for col in tv_unknown_reals:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    max_encoder_length = enc_len
    max_prediction_length = dec_len

    training_cutoff = df["time_idx"].max() - max_prediction_length

    # 4) Cấu hình TimeSeriesDataSet
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
        target_normalizer=GroupNormalizer(groups=["site_id"], transformation=None),  # giữ nguyên, tránh lỗi KeyError
        add_relative_time_idx=True,
        allow_missing_timesteps=True,
    )

    # 5) Tạo dataset
    training = TimeSeriesDataSet(df[df.time_idx <= training_cutoff], **common_kwargs)
    validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff + 1)

    return training, validation
