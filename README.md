# TFT Water Level Forecasting (Multi-Horizon with Quantiles)

Implements your design doc using **PyTorch Forecasting** (TFT), delivering:

-   Data prep & feature engineering (lags, rolling stats, cyclical time)
-   TimeSeriesDataSet for encoder/decoder windows
-   Temporal Fusion Transformer with QuantileLoss `[0.1, 0.5, 0.9]`
-   Evaluation: MAE/RMSE, horizon slices, coverage, CRPS\*
-   Serving API (FastAPI) returning multi-horizon quantile forecasts
-   Dockerfile for deployment
-   Use python 3.11

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
    "history": [
        {
            "timestamp": "2025-01-01T00:00:00Z",
            "site_id": "S1",
            "muc_thuong_luu": 50.1,
            "muc_dang_binh_thuong": 55,
            "muc_chet": 40,
            "luu_luong_den": 120,
            "tong_luong_xa": 60,
            "xa_tran": 0,
            "xa_nha_may": 60,
            "so_cua_xa_sau": 2,
            "so_cua_xa_mat": 1
        }
    ],
    "known_future": [
        {
            "timestamp": "2025-01-08T00:00:00Z",
            "site_id": "S1",
            "muc_thuong_luu": 0,
            "muc_dang_binh_thuong": 55,
            "muc_chet": 40,
            "luu_luong_den": 0,
            "tong_luong_xa": 0,
            "xa_tran": 0,
            "xa_nha_may": 0,
            "so_cua_xa_sau": 2,
            "so_cua_xa_mat": 1,
            "rain_forecast_mm": 5,
            "planned_release_m3s": 50
        }
    ]
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

## Testing & Evaluation

### Comprehensive Model Testing

Sau khi huấn luyện, sử dụng test service để đánh giá toàn diện mô hình:

```bash
# Test mô hình với đầy đủ metrics và visualizations
python -m src.water_forecast.test_model --cfg configs/default.yaml

# Hoặc test với checkpoint cụ thể
python -m src.water_forecast.test_model --cfg configs/default.yaml --ckpt models/tft-best.ckpt

# Chỉ định số lượng sample predictions để plot
python -m src.water_forecast.test_model --cfg configs/default.yaml --samples 10
```

Test service sẽ tạo ra:

-   📊 **metrics.json**: Các metrics đánh giá chi tiết
-   📄 **metrics.txt**: Báo cáo metrics dạng text dễ đọc
-   📈 **forecast_samples.png**: Dự đoán mẫu với uncertainty bands
-   📉 **error_analysis.png**: Phân tích phân bố lỗi
-   📊 **coverage_analysis.png**: Phân tích độ bao phủ của prediction intervals

### Metrics được tính toán

#### Overall Performance

-   **MAE**: Mean Absolute Error
-   **RMSE**: Root Mean Squared Error
-   **MAPE**: Mean Absolute Percentage Error
-   **Coverage_80%**: Tỷ lệ giá trị thực nằm trong [Q10, Q90]
-   **Interval_Width**: Độ rộng trung bình của prediction interval
-   **Sharpness**: Độ biến thiên của interval width

#### Horizon-wise Performance

-   **MAE_24h**: MAE cho 24 giờ đầu
-   **RMSE_24h**: RMSE cho 24 giờ đầu
-   **MAE_7d**: MAE cho 7 ngày đầu
-   **RMSE_7d**: RMSE cho 7 ngày đầu
-   **MAE_14d**: MAE cho 14 ngày đầu
-   **RMSE_14d**: RMSE cho 14 ngày đầu

#### Quantile Loss

-   **Pinball_Q10**: Pinball loss cho quantile 0.1
-   **Pinball_Q50**: Pinball loss cho quantile 0.5 (median)
-   **Pinball_Q90**: Pinball loss cho quantile 0.9
-   **Avg_Pinball**: Trung bình pinball loss

### Evaluate Script (Legacy)

Script evaluate cũ vẫn có sẵn:

```bash
python -m src.water_forecast.evaluate --cfg configs/default.yaml
```

### Unit Tests

Chạy unit tests để kiểm tra các thành phần riêng lẻ:

```bash
# Chạy tất cả tests
pytest tests/

# Chạy test cụ thể
pytest tests/test_model.py -v
pytest tests/test_ingest.py -v
pytest tests/test_preprocessing.py -v

# Chạy với coverage
pytest tests/ --cov=src/water_forecast --cov-report=html
```

### Prediction CLI

Dự đoán nhanh từ command line:

```bash
python -m src.water_forecast.predict_cli --cfg configs/default.yaml --ckpt models/tft-best.ckpt
```

## Mô hình Temporal Fusion Transformer (TFT)

### Tổng quan

Dự án sử dụng mô hình **Temporal Fusion Transformer (TFT)** - một kiến trúc deep learning tiên tiến được thiết kế đặc biệt cho bài toán dự báo chuỗi thời gian đa biến. TFT kết hợp các cơ chế attention với khả năng xử lý các biến động thời gian phức tạp.

### Kiến trúc mô hình

#### 1. **Các thành phần chính**

-   **Variable Selection Networks (VSN)**: Tự động chọn lọc các biến đầu vào quan trọng nhất
-   **Gated Residual Networks (GRN)**: Xử lý thông tin với khả năng học phi tuyến tính
-   **Multi-head Attention**: Học được mối quan hệ phụ thuộc thời gian dài hạn
-   **Quantile Regression**: Dự đoán nhiều quantiles để ước lượng độ không chắc chắn

#### 2. **Cấu hình mô hình**

```python
# Các tham số chính của TFT
hidden_size = 160              # Kích thước hidden layer
attention_heads = 4            # Số lượng attention heads
dropout = 0.1                  # Tỷ lệ dropout để tránh overfitting
learning_rate = 1e-3           # Tốc độ học
quantiles = (0.1, 0.5, 0.9)   # Các quantile để dự đoán
```

#### 3. **Các lớp xử lý**

##### a. **Input Layer**

-   Nhận đầu vào là chuỗi thời gian với nhiều biến (multivariate time series)
-   Xử lý các loại biến:
    -   **Static covariates**: Các biến không đổi theo thời gian (vd: ID trạm đo)
    -   **Time-varying known**: Các biến biết trước (vd: thời gian trong ngày, tháng)
    -   **Time-varying unknown**: Các biến chỉ biết trong quá khứ (vd: lượng mưa thực tế)

##### b. **Variable Selection Network**

-   Tự động đánh trọng số cho từng biến đầu vào
-   Giúp mô hình tập trung vào các biến quan trọng nhất
-   Cải thiện khả năng giải thích của mô hình

##### c. **LSTM Encoder-Decoder**

-   **Encoder**: Xử lý dữ liệu quá khứ (historical data)
-   **Decoder**: Xử lý các biến known trong tương lai
-   Kết hợp thông tin từ cả quá khứ và tương lai đã biết

##### d. **Multi-head Attention Layer**

-   Học được mối quan hệ phụ thuộc giữa các bước thời gian
-   Cho phép mô hình "chú ý" đến các thời điểm quan trọng trong quá khứ
-   Sử dụng 4 attention heads để học nhiều pattern khác nhau

##### e. **Gated Residual Network (GRN)**

-   Xử lý thông tin phi tuyến tính
-   Có cơ chế gating để kiểm soát luồng thông tin
-   Kết nối residual giúp huấn luyện mô hình sâu hơn

##### f. **Output Layer - Quantile Regression**

-   Dự đoán 3 quantiles: 10%, 50% (median), 90%
-   **Q10 (0.1)**: Giới hạn dưới - kịch bản lạc quan
-   **Q50 (0.5)**: Dự đoán trung bình - kịch bản có khả năng nhất
-   **Q90 (0.9)**: Giới hạn trên - kịch bản bi quan
-   Cho phép đánh giá độ không chắc chắn của dự đoán

### Cách thức dự đoán

#### 1. **Quá trình dự đoán**

```python
# Bước 1: Chuẩn bị dữ liệu đầu vào
# - Context window: N bước thời gian trong quá khứ
# - Prediction horizon: M bước thời gian cần dự đoán

# Bước 2: Variable Selection
# Mô hình đánh giá tầm quan trọng của từng biến

# Bước 3: LSTM Encoding
# Mã hóa thông tin từ chuỗi quá khứ

# Bước 4: Temporal Fusion
# Kết hợp thông tin quá khứ với các biến known tương lai

# Bước 5: Multi-head Attention
# Tập trung vào các thời điểm quan trọng

# Bước 6: Quantile Prediction
# Dự đoán 3 giá trị cho mỗi bước thời gian
```

#### 2. **Ví dụ sử dụng**

```python
from water_forecast.tft_model import make_tft

# Tạo mô hình từ dataset
model = make_tft(
    training_dataset,
    hidden_size=160,
    attention_heads=4,
    dropout=0.1,
    learning_rate=1e-3,
    quantiles=(0.1, 0.5, 0.9)
)

# Huấn luyện
trainer.fit(model, train_loader, val_loader)

# Dự đoán
predictions = model.predict(test_data)

# Kết quả: dictionary chứa các quantiles
# predictions["prediction"]: shape (batch_size, prediction_length, 3)
# [:, :, 0]: quantile 0.1 (giới hạn dưới)
# [:, :, 1]: quantile 0.5 (dự đoán trung bình)
# [:, :, 2]: quantile 0.9 (giới hạn trên)
```

### Loss Function - Quantile Loss

Mô hình sử dụng **Quantile Loss** để tối ưu hóa:

```python
QuantileLoss = Σ max[q(y - ŷ), (q-1)(y - ŷ)]
```

Trong đó:

-   `y`: Giá trị thực tế
-   `ŷ`: Giá trị dự đoán
-   `q`: Quantile (0.1, 0.5, 0.9)

**Ưu điểm:**

-   Cho phép dự đoán khoảng tin cậy, không chỉ giá trị điểm
-   Phù hợp với các tình huống cần đánh giá rủi ro
-   Xử lý tốt với dữ liệu có outliers

### Ưu điểm của TFT

1. **Khả năng giải thích cao**

    - Variable importance: Biết biến nào quan trọng
    - Attention weights: Hiểu mô hình tập trung vào thời điểm nào

2. **Xử lý dữ liệu phức tạp**

    - Nhiều biến đầu vào (multivariate)
    - Các biến với tính chất khác nhau (static, time-varying)
    - Missing data handling

3. **Dự đoán khoảng tin cậy**

    - Quantile prediction cho phép đánh giá độ không chắc chắn
    - Hữu ích cho ra quyết định trong điều kiện bất định

4. **Hiệu suất cao**
    - Kết hợp LSTM và Attention hiệu quả
    - Học được cả phụ thuộc ngắn hạn và dài hạn

### Hyperparameters chính

| Tham số                      | Giá trị mặc định | Mô tả                                      |
| ---------------------------- | ---------------- | ------------------------------------------ |
| `hidden_size`                | 160              | Kích thước của các hidden layers           |
| `attention_heads`            | 4                | Số lượng attention heads                   |
| `dropout`                    | 0.1              | Tỷ lệ dropout (0-1)                        |
| `learning_rate`              | 1e-3             | Tốc độ học ban đầu                         |
| `quantiles`                  | (0.1, 0.5, 0.9)  | Các quantile để dự đoán                    |
| `log_interval`               | 50               | Tần suất log metrics                       |
| `reduce_on_plateau_patience` | 4                | Số epochs chờ trước khi giảm learning rate |

### Monitoring & Callbacks

Mô hình sử dụng các callbacks của PyTorch Lightning:

-   **EarlyStopping**: Dừng training khi validation loss không cải thiện
-   **ModelCheckpoint**: Lưu best model theo validation loss
-   **LearningRateMonitor**: Theo dõi learning rate schedule
-   **ReduceLROnPlateau**: Tự động giảm learning rate khi loss plateau

### Tài liệu tham khảo

-   [PyTorch Forecasting Documentation](https://pytorch-forecasting.readthedocs.io/)
-   [Temporal Fusion Transformers Paper](https://arxiv.org/abs/1912.09363)
-   [Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
