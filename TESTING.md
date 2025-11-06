# Testing Guide - Hướng dẫn Test Mô hình

## Tổng quan

Dự án cung cấp hệ thống test toàn diện để đánh giá mô hình sau khi training, bao gồm:

1. **Comprehensive Model Testing** (`test_model.py`) - Test tổng thể với metrics và visualizations
2. **Unit Tests** (`tests/`) - Test các thành phần riêng lẻ
3. **Evaluation Script** (`evaluate.py`) - Script đánh giá legacy

## 1. Comprehensive Model Testing

### Chức năng

Test service (`test_model.py`) cung c **Metrics toàn diện:**

-   Overall: MAE, RMSE, MAPE, Coverage, Interval Width
-   Horizon-wise: Metrics theo từng khoảng thời gian (24h, 7d, 14d)
-   Quantile Loss: Pinball loss cho từng quantil  **Visualizations:**

-   Forecast samples với uncertainty bands
-   Error distribution analysis
-   Coverage analysis by horizon
-   Interval width analys **Output files:**

-   `metrics.json` - Metrics dạng JSON
-   `metrics.txt` - Metrics dạng text dễ đọc
-   `forecast_samples.png` - Sample predictions
-   `error_analysis.png` - Error distribution plots
-   `coverage_analysis.png` - Coverage plots

### Sử dụng

```bash
# Test cơ bản (sử dụng best checkpoint từ config)
python -m src.water_forecast.test_model --cfg configs/default.yaml

# Test với checkpoint cụ thể
python -m src.water_forecast.test_model \
    --cfg configs/default.yaml \
    --ckpt models/tft-best.ckpt

# Chỉ định số lượng samples để plot
python -m src.water_forecast.test_model \
    --cfg configs/default.yaml \
    --samples 10

# Sử dụng Makefile
make test-model
```

### Output

Kết quả được lưu trong `artifacts/test_results/`:

```
artifacts/test_results/
├── metrics.json              # Metrics chi tiết
├── metrics.txt               # Báo cáo text
├── forecast_samples.png      # Dự đoán mẫu
├── error_analysis.png        # Phân tích lỗi
└── coverage_analysis.png     # Phân tích coverage
```

### Ví dụ output metrics

```
============================================================
📊 MODEL EVALUATION RESULTS
============================================================

🎯 Overall Performance:
  MAE                 :   0.1234
  RMSE                :   0.2456
  MAPE                :   5.6789
  Coverage_80%        :  78.9012
  Interval_Width      :   0.5678
  Sharpness           :   0.1234

⏱️  Horizon-wise Performance:
  MAE_24h             :   0.1111
  RMSE_24h            :   0.2222
  MAE_7d              :   0.1333
  RMSE_7d             :   0.2444
  MAE_14d             :   0.1456
  RMSE_14d            :   0.2567

📉 Quantile Loss:
  Pinball_Q10         :   0.0123
  Pinball_Q50         :   0.0456
  Pinball_Q90         :   0.0789
  Avg_Pinball         :   0.0456
============================================================
```

## 2. Unit Tests

### Các test có sẵn

#### `tests/test_model.py`

Test các chức năng của model:

- Model creation from dataset
- Forward pass
- Prediction generation
- Training step
- Save/load checkpoint
- Metrics calculation

#### `tests/test_ingest.py`

Test data ingestion:

- CSV reading
- Column validation
- Timestamp parsing
- Data cleaning

#### `tests/test_preprocessing.py`

Test preprocessing:

- Train/val/test split
- Scaling/normalization
- Feature engineering

### Chạy unit tests

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy test cụ thể
pytest tests/test_model.py -v
pytest tests/test_model.py::TestModelTraining::test_model_creation -v

# Chạy với coverage
pytest tests/ --cov=src/water_forecast --cov-report=html

# Chạy parallel (nhanh hơn)
pytest tests/ -n auto

# Sử dụng Makefile
make test-unit
```

### Ví dụ output

```
tests/test_model.py::TestModelTraining::test_model_creation PASSED      [ 10%]
tests/test_model.py::TestModelTraining::test_model_forward_pass PASSED  [ 20%]
tests/test_model.py::TestModelTraining::test_model_prediction PASSED    [ 30%]
tests/test_model.py::TestModelTraining::test_model_training_step PASSED [ 40%]
tests/test_model.py::TestModelTraining::test_model_load_checkpoint PASSED [ 50%]
tests/test_model.py::TestModelMetrics::test_mae_calculation PASSED      [ 60%]
tests/test_model.py::TestModelMetrics::test_coverage_calculation PASSED [ 70%]
tests/test_model.py::TestModelMetrics::test_pinball_loss PASSED         [ 80%]

========================== 8 passed in 45.23s ===========================
```

## 3. Evaluation Script (Legacy)

Script evaluate.py cũ vẫn có sẵn:

```bash
python -m src.water_forecast.evaluate --cfg configs/default.yaml
```

Output:

-   `artifacts/metrics.txt` - Metrics cơ bản
-   `artifacts/forecast_0.png` - Sample forecasts

## 4. Workflow Testing Hoàn chỉnh

### Development Workflow

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate  # hoặc `.venv\Scripts\activate` trên Windows
pip install -r requirements-dev.txt

# 2. Run unit tests during development
pytest tests/test_model.py -v --tb=short

# 3. Train model
python -m src.water_forecast.train --cfg configs/default.yaml

# 4. Comprehensive evaluation
python -m src.water_forecast.test_model --cfg configs/default.yaml

# 5. Review results
ls -la artifacts/test_results/
```

### CI/CD Workflow

```bash
# Run all tests
make test

# Hoặc
pytest tests/ -v && python -m src.water_forecast.test_model --cfg configs/default.yaml
```

## 5. Interpreting Results

### MAE & RMSE

-   **MAE**: Lỗi tuyệt đối trung bình, đơn vị giống với target
-   **RMSE**: Nhấn mạnh lỗi lớn hơn, đơn vị giống với target
-   **Best**: Càng thấp càng tốt
-   **Typical**: MAE < RMSE (do RMSE penalize outliers)

### Coverage

-   **Target**: 80% (cho [Q10, Q90] interval)
-   **Good**: 75-85%
-   **Too narrow**: >85% → Model quá tự tin
-   **Too wide**: <75% → Model không tự tin

### Pinball Loss

-   **Lower is better**
-   **Q50** (median): Thường thấp nhất
-   **Q10/Q90**: Thường cao hơn do extreme quantiles

### Horizon-wise Metrics

-   **Expected**: Error tăng theo horizon (xa hơn → khó đoán hơn)
-   **Issue**: Nếu error tăng đột ngột → cần review model

## 6. Troubleshooting

### Lỗi thường gặp

#### 1. Import Error

```
ImportError: cannot import name 'seed_everything'
```

**Fix**: Cài đúng version Lightning:

```bash
pip install 'lightning>=2.0,<3.0'
```

#### 2. Checkpoint not found

```
FileNotFoundError: Checkpoint not found
```

**Fix**: Chỉ định đường dẫn checkpoint:

```bash
python -m src.water_forecast.test_model --ckpt models/tft-best.ckpt
```

#### 3. Out of memory

```
RuntimeError: CUDA out of memory
```

**Fix**: Giảm batch size trong config hoặc dùng CPU:

```yaml
batch_size: 16 # giảm từ 32
```

#### 4. No test data

```
AssertionError: Test set is empty
```

**Fix**: Kiểm tra split ratio trong config:

```yaml
split:
    train_ratio: 0.7
    val_ratio: 0.15
    # test_ratio: 0.15 (tự động)
```

## 7. Best Practices

# DO

-   Chạy unit tests trước khi training
-   Test với nhiều random seeds để kiểm tra stability
-   Lưu metrics cho mỗi experiment
-   So sánh metrics giữa các versions
-   Review visualizations để hiểu model behavior

### DON'T

-   Không test trên training data
-   Không bỏ qua warnings
-   Không chỉ nhìn vào 1 metric duy nhất
-   Không ignore outliers trong error analysis

## 8. Advanced Testing

### A/B Testing

```bash
# Test model A
python -m src.water_forecast.test_model \
    --cfg configs/default.yaml \
    --ckpt models/model_a.ckpt

# Test model B
python -m src.water_forecast.test_model \
    --cfg configs/default.yaml \
    --ckpt models/model_b.ckpt

# Compare results
diff artifacts/test_results_a/metrics.json artifacts/test_results_b/metrics.json
```

### Statistical Significance Testing

Thêm vào test script (tùy chọn):

```python
from scipy import stats

# Compare predictions from 2 models
errors_a = actuals - preds_a
errors_b = actuals - preds_b

# Paired t-test
t_stat, p_value = stats.ttest_rel(np.abs(errors_a), np.abs(errors_b))

if p_value < 0.05:
    print("Difference is statistically significant!")
```

## 9. Continuous Integration Example

`.github/workflows/test.yml`:

```yaml
name: Test

on: [push, pull_request]

jobs:
    test:
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v2
            - uses: actions/setup-python@v2
              with:
                  python-version: '3.11'
            - name: Install dependencies
              run: pip install -r requirements-dev.txt
            - name: Run unit tests
              run: pytest tests/ -v --cov=src
            - name: Upload coverage
              uses: codecov/codecov-action@v2
```

## 10. Tài liệu tham khảo

-   [PyTest Documentation](https://docs.pytest.org/)
-   [PyTorch Lightning Testing](https://lightning.ai/docs/pytorch/stable/common/evaluation.html)
-   [Model Evaluation Best Practices](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Câu hỏi?** Mở issue trên GitHub hoặc liên hệ team.
