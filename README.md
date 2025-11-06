# TFT Water Level Forecasting (Multi-Horizon with Quantiles)

Implements your design doc using **PyTorch Forecasting** (TFT), delivering:
- Data prep & feature engineering (lags, rolling stats, cyclical time)
- TimeSeriesDataSet for encoder/decoder windows
- Temporal Fusion Transformer with QuantileLoss `[0.1, 0.5, 0.9]`
- Evaluation: MAE/RMSE, horizon slices, coverage, CRPS*
- Serving API (FastAPI) returning multi-horizon quantile forecasts
- Dockerfile for deployment
- Use python 3.11

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --cfg configs/default.yaml
python -m src.evaluate --cfg configs/default.yaml --ckpt models/tft-best.ckpt
uvicorn service.app:app --host 0.0.0.0 --port 8000
```


### API

POST `/predict` with JSON:

```json
{
  "history": [ {"timestamp": "2025-01-01T00:00:00Z", "site_id": "S1", "muc_thuong_luu": 50.1, "muc_dang_binh_thuong": 55, "muc_chet": 40, "luu_luong_den": 120, "tong_luong_xa": 60, "xa_tran": 0, "xa_nha_may": 60, "so_cua_xa_sau": 2, "so_cua_xa_mat": 1 } ],
  "known_future": [ {"timestamp": "2025-01-08T00:00:00Z", "site_id": "S1", "muc_thuong_luu": 0, "muc_dang_binh_thuong": 55, "muc_chet": 40, "luu_luong_den": 0, "tong_luong_xa": 0, "xa_tran": 0, "xa_nha_may": 0, "so_cua_xa_sau": 2, "so_cua_xa_mat": 1, "rain_forecast_mm": 5, "planned_release_m3s": 50 } ]
}
```

Returns quantiles and horizon arrays.

### Full ETL + Training + Serving Pipeline
```bash
# 1) ETL từ crawl → một CSV huấn luyện chuẩn
python -m src.water_forecast.ingest --crawl_dir data/crawl --out data/sample.csv --tz Asia/Bangkok

# 2) Sinh bộ train/val/test cố định (có cả bản scaled nếu cần)
python -m src.water_forecast.make_splits --cfg configs/default.yaml --out_dir data/processed --format parquet --scaled

# 3) Huấn luyện như cũ (train.py vẫn có thể tự split nội bộ)
python -m src.water_forecast.train --cfg configs/default.yaml
```
