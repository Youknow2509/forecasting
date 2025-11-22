from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet

from .config import load_config
from .dataio import load_csv, resample_fill
from .features import add_time_features, add_lags_rollings
from .preprocessing import fit_scalers, apply_scalers, train_val_test_split, inverse_transform_target
from .dataset import build_timeseries_datasets


def predict_water_level(cfg_path: str, ckpt_path: str):
    """
    Thực hiện dự đoán mực nước và trả về kết quả.
    
    Returns:
        predictions: numpy array shape [N, H, Q] - dự đoán quantiles
        timestamps: list of timestamps tương ứng
        actual_values: numpy array - giá trị thực tế (nếu có)
        full_df: DataFrame đầy đủ với các features
    """
    cfg = load_config(cfg_path)
    
    # Load và tiền xử lý dữ liệu
    df = load_csv(cfg.paths.data_csv, timezone=cfg.timezone)
    df = resample_fill(df, cfg.frequency, cfg.timezone)
    df = add_time_features(df)
    df = add_lags_rollings(df, "muc_thuong_luu", cfg.lags_hours, cfg.roll_windows_hours, cfg.roll_stats)

    # Chia train/val/test và chuẩn hóa
    tr, va, te = train_val_test_split(df, cfg.split.train_ratio, cfg.split.val_ratio)
    scalers = fit_scalers(tr)
    full = pd.concat([apply_scalers(x, scalers) for x in [tr, va, te]])

    # Sort và thêm time_idx
    full = full.sort_values(["site_id", "timestamp"]).copy()
    full["time_idx"] = full.groupby("site_id").cumcount()
    
    # Fill NaN values trong lag/rolling features
    lag_roll_cols = [c for c in full.columns if c.startswith("muc_thuong_luu_lag_") or c.startswith("muc_thuong_luu_rolling_")]
    for col in lag_roll_cols:
        full[col] = full.groupby("site_id")[col].ffill().bfill()

    # Tạo dataset và load model
    training, _ = build_timeseries_datasets(full, cfg.enc_len, cfg.dec_len)
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)

    # Dự đoán
    to_pred = TimeSeriesDataSet.from_dataset(training, full, predict=True, stop_randomization=True)
    dl = to_pred.to_dataloader(train=False, batch_size=cfg.batch_size)
    predictions = model.predict(dl, mode="quantiles")  # [N, H, Q]
    
    # Chuyển về numpy
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    
    # INVERSE TRANSFORM: Chuyển predictions từ dạng chuẩn hóa về thang đo gốc
    if predictions.ndim == 3:
        # predictions shape: [N, H, Q]
        for i in range(predictions.shape[0]):
            for q in range(predictions.shape[2]):
                predictions[i, :, q] = inverse_transform_target(predictions[i, :, q], scalers, "muc_thuong_luu")
    elif predictions.ndim == 2:
        # predictions shape: [H, Q]
        for q in range(predictions.shape[1]):
            predictions[:, q] = inverse_transform_target(predictions[:, q], scalers, "muc_thuong_luu")
    
    # Lấy timestamps cho prediction
    # Giả sử dự đoán từ thời điểm cuối cùng của dữ liệu
    last_timestamp = full['timestamp'].max()
    freq = pd.Timedelta(cfg.frequency)
    pred_timestamps = [last_timestamp + freq * (i + 1) for i in range(predictions.shape[1])]
    
    # Lấy actual values từ test set gốc (CHƯA scale)
    # te là từ df gốc nên không cần inverse transform
    actual_values = None
    if len(te) > 0 and 'muc_thuong_luu' in te.columns:
        actual_values = te['muc_thuong_luu'].values[:predictions.shape[1]]
        if len(actual_values) < predictions.shape[1]:
            # Pad với NaN nếu không đủ
            actual_values = np.concatenate([actual_values, np.full(predictions.shape[1] - len(actual_values), np.nan)])
    
    return predictions, pred_timestamps, actual_values, full, scalers, cfg


def plot_predictions(predictions, timestamps, actual_values=None, save_path="predictions_plot.png"):
    """
    Vẽ biểu đồ dự đoán với các quantiles.
    
    Args:
        predictions: numpy array shape [N, H, Q] hoặc [H, Q]
        timestamps: list of timestamps
        actual_values: numpy array - giá trị thực tế (optional)
        save_path: đường dẫn lưu biểu đồ
    """
    # Nếu predictions có batch dimension, lấy batch đầu tiên
    if predictions.ndim == 3:
        predictions = predictions[0]  # shape [H, Q]
    
    # Giả sử quantiles là [0.1, 0.5, 0.9]
    # ensure lower/upper bounds even if ordering is unexpected
    q = predictions
    if q.ndim == 3:
        q = q[0]
    # q shape [H, Q]
    if q.shape[1] >= 3:
        q10 = q[:, 0]
        q50 = q[:, 1]
        q90 = q[:, 2]
    else:
        # fallback: use min/median/max across axis
        q10 = np.min(q, axis=1)
        q50 = np.median(q, axis=1)
        q90 = np.max(q, axis=1)

    # ensure lower <= median <= upper for plotting
    lower = np.minimum.reduce([q10, q50, q90])
    upper = np.maximum.reduce([q10, q50, q90])
    median = q50
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Vẽ dự đoán
    ax.plot(timestamps, median, label='Dự đoán (median)', color='tab:blue', linewidth=2.5)
    ax.fill_between(timestamps, lower, upper, alpha=0.3, color='tab:blue', label='Khoảng tin cậy (10%-90%)')
    
    # Vẽ giá trị thực tế nếu có
    if actual_values is not None:
        actual_len = min(len(actual_values), len(timestamps))
        ax.plot(timestamps[:actual_len], actual_values[:actual_len], 
                label='Giá trị thực tế', color='tab:red', linewidth=2.5, linestyle='--', marker='o', markersize=5)
    
    # Tự động điều chỉnh y-axis để zoom vào vùng dữ liệu
    all_values = list(median) + list(lower) + list(upper)
    if actual_values is not None:
        all_values.extend([v for v in actual_values if not np.isnan(v)])
    y_min = np.nanmin(all_values)
    y_max = np.nanmax(all_values)
    y_range = y_max - y_min
    margin = max(y_range * 0.15, 2)  # margin 15% hoặc tối thiểu 2m
    ax.set_ylim(y_min - margin, y_max + margin)
    
    # Định dạng
    ax.set_xlabel('Thời gian', fontsize=12)
    ax.set_ylabel('Mực nước thượng lưu (m)', fontsize=12)
    ax.set_title('Dự đoán Mực Nước Thượng Lưu', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Biểu đồ đã được lưu tại: {save_path}")
    plt.show()
    
    return fig


def plot_detailed_analysis(predictions, timestamps, actual_values, full_df, save_dir="artifacts"):
    """
    Tạo nhiều biểu đồ phân tích chi tiết.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if predictions.ndim == 3:
        predictions = predictions[0]

    q = predictions
    if q.shape[1] >= 3:
        q10 = q[:, 0]
        q50 = q[:, 1]
        q90 = q[:, 2]
    else:
        q10 = np.min(q, axis=1)
        q50 = np.median(q, axis=1)
        q90 = np.max(q, axis=1)

    lower = np.minimum.reduce([q10, q50, q90])
    upper = np.maximum.reduce([q10, q50, q90])
    median = q50
    
    # 1. Biểu đồ chính với historical data + subplot zoom
    fig1 = plt.figure(figsize=(18, 10))
    gs = fig1.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1], hspace=0.3, wspace=0.3)
    ax1 = fig1.add_subplot(gs[0, :])
    
    # Lấy 7 ngày gần nhất từ historical data
    hist_hours = 168  # 7 days
    hist_data = full_df.tail(hist_hours)
    hist_timestamps = pd.to_datetime(hist_data['timestamp']).values
    hist_values = hist_data['muc_thuong_luu'].values
    
    # Vẽ historical
    ax1.plot(hist_timestamps, hist_values, label='Dữ liệu lịch sử', 
             color='gray', linewidth=2, alpha=0.8)

    # Vẽ predictions
    ax1.plot(timestamps, median, label='Dự đoán (median)', color='tab:blue', linewidth=2.5)
    ax1.fill_between(timestamps, lower, upper, alpha=0.25, color='tab:blue', 
                     label='Khoảng tin cậy (10%-90%)')
    
    if actual_values is not None:
        actual_len = min(len(actual_values), len(timestamps))
        ax1.plot(timestamps[:actual_len], actual_values[:actual_len],
                label='Giá trị thực tế', color='tab:red', linewidth=2, linestyle='--', marker='o', markersize=3)
    
    ax1.axvline(x=timestamps[0], color='green', linestyle=':', linewidth=2, label='Điểm dự đoán')
    ax1.set_xlabel('Thời gian', fontsize=13)
    ax1.set_ylabel('Mực nước thượng lưu (m)', fontsize=13)
    ax1.set_title('Dự đoán Mực Nước Thượng Lưu - Phân tích Chi tiết', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    
    # Subplot 1: Zoom vào vùng dự đoán (72h đầu)
    ax2 = fig1.add_subplot(gs[1, 0])
    zoom_hours = min(72, len(timestamps))
    ax2.plot(timestamps[:zoom_hours], median[:zoom_hours], label='Dự đoán (median)', 
             color='tab:blue', linewidth=2.5, marker='o', markersize=4)
    ax2.fill_between(timestamps[:zoom_hours], lower[:zoom_hours], upper[:zoom_hours], 
                     alpha=0.3, color='tab:blue', label='Khoảng tin cậy')
    if actual_values is not None:
        actual_len = min(len(actual_values), zoom_hours)
        ax2.plot(timestamps[:actual_len], actual_values[:actual_len],
                label='Giá trị thực tế', color='tab:red', linewidth=2.5, 
                linestyle='--', marker='s', markersize=4)
    
    # Zoom y-axis
    zoom_values = list(median[:zoom_hours]) + list(lower[:zoom_hours]) + list(upper[:zoom_hours])
    if actual_values is not None:
        zoom_values.extend([v for v in actual_values[:zoom_hours] if not np.isnan(v)])
    y_min = np.nanmin(zoom_values)
    y_max = np.nanmax(zoom_values)
    y_range = y_max - y_min
    margin = max(y_range * 0.2, 1)
    ax2.set_ylim(y_min - margin, y_max + margin)
    
    ax2.set_xlabel('Thời gian (72h đầu)', fontsize=11)
    ax2.set_ylabel('Mực nước (m)', fontsize=11)
    ax2.set_title('Chi tiết 3 ngày đầu', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Subplot 2: Hiển thị sai số theo thời gian
    ax3 = fig1.add_subplot(gs[1, 1])
    if actual_values is not None:
        actual_len = min(len(actual_values), len(median))
        errors = median[:actual_len] - actual_values[:actual_len]
        valid_mask = ~np.isnan(errors)
        timestamps_array = np.array(timestamps[:actual_len])
        ax3.plot(timestamps_array[valid_mask], errors[valid_mask], 
                color='tab:purple', linewidth=2, marker='o', markersize=3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax3.fill_between(timestamps_array[valid_mask], 0, errors[valid_mask], 
                        alpha=0.2, color='tab:purple')
        ax3.set_xlabel('Thời gian', fontsize=11)
        ax3.set_ylabel('Sai số (Dự đoán - Thực tế) [m]', fontsize=11)
        ax3.set_title('Phân tích Sai số', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle=':')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/detailed_forecast.png", dpi=300, bbox_inches='tight')
    print(f"✓ Biểu đồ chi tiết đã lưu: {save_dir}/detailed_forecast.png")
    
    # 2. Biểu đồ uncertainty (độ không chắc chắn)
    # 2. Biểu đồ uncertainty (độ không chắc chắn)
    fig2, ax2_main = plt.subplots(figsize=(14, 6))
    uncertainty = upper - lower
    ax2_main.plot(timestamps, uncertainty, color='tab:orange', linewidth=2.5, marker='o', markersize=3)
    ax2_main.fill_between(timestamps, 0, uncertainty, alpha=0.25, color='tab:orange')
    
    # Thêm thông tin thống kê
    mean_unc = np.mean(uncertainty)
    max_unc = np.max(uncertainty)
    ax2_main.axhline(y=mean_unc, color='red', linestyle='--', linewidth=2, 
                    label=f'Trung bình: {mean_unc:.2f}m')
    ax2_main.text(timestamps[len(timestamps)//2], max_unc * 0.9, 
                 f'Max: {max_unc:.2f}m\nMin: {np.min(uncertainty):.2f}m', 
                 fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2_main.set_xlabel('Thời gian', fontsize=12)
    ax2_main.set_ylabel('Độ không chắc chắn (m)', fontsize=12)
    ax2_main.set_title('Độ Không Chắc Chắn trong Dự Đoán (Khoảng tin cậy 80%)', fontsize=14, fontweight='bold')
    ax2_main.legend(loc='best', fontsize=11)
    ax2_main.grid(True, alpha=0.3, linestyle=':')
    ax2_main.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/uncertainty.png", dpi=300, bbox_inches='tight')
    print(f"✓ Biểu đồ độ không chắc chắn đã lưu: {save_dir}/uncertainty.png")
    
    # 3. Nếu có actual values, tính metrics và vẽ comparison
    if actual_values is not None:
        actual_len = min(len(actual_values), len(median))
        pred_subset = median[:actual_len]
        actual_subset = actual_values[:actual_len]
        
        # Tính metrics
        mae = np.mean(np.abs(pred_subset - actual_subset))
        rmse = np.sqrt(np.mean((pred_subset - actual_subset) ** 2))
        mape = np.mean(np.abs((pred_subset - actual_subset) / (actual_subset + 1e-8))) * 100
        
        # Vẽ scatter plot
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.scatter(actual_subset, pred_subset, alpha=0.5, s=30)
        
        # Đường y=x
        min_val = min(actual_subset.min(), pred_subset.min())
        max_val = max(actual_subset.max(), pred_subset.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Dự đoán hoàn hảo')
        
        ax3.set_xlabel('Giá trị thực tế (m)', fontsize=12)
        ax3.set_ylabel('Giá trị dự đoán (m)', fontsize=12)
        ax3.set_title(f'So sánh Dự đoán vs Thực tế\nMAE={mae:.3f}m, RMSE={rmse:.3f}m, MAPE={mape:.2f}%', 
                     fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/comparison.png", dpi=300, bbox_inches='tight')
        print(f"✓ Biểu đồ so sánh đã lưu: {save_dir}/comparison.png")
        
        # In metrics
        print("\n" + "="*50)
        print("📊 METRICS DỰ ĐOÁN:")
        print("="*50)
        print(f"MAE (Mean Absolute Error):     {mae:.4f} m")
        print(f"RMSE (Root Mean Square Error): {rmse:.4f} m")
        print(f"MAPE (Mean Abs % Error):       {mape:.2f} %")
        print("="*50 + "\n")
    
    plt.close('all')


def save_predictions_to_csv(predictions, timestamps, actual_values=None, save_path="predictions.csv"):
    """
    Lưu kết quả dự đoán ra file CSV.
    """
    if predictions.ndim == 3:
        predictions = predictions[0]
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_q10': predictions[:, 0],
        'predicted_median': predictions[:, 1],
        'predicted_q90': predictions[:, 2],
    })
    
    if actual_values is not None:
        actual_len = min(len(actual_values), len(timestamps))
        df['actual_value'] = np.nan
        df.loc[:actual_len-1, 'actual_value'] = actual_values[:actual_len]
    
    df.to_csv(save_path, index=False)
    print(f"✓ Kết quả dự đoán đã lưu tại: {save_path}")


def main():
    ap = argparse.ArgumentParser(description='Dự đoán mực nước và tạo biểu đồ')
    ap.add_argument("--cfg", default="configs/default.yaml", help="Đường dẫn config file")
    ap.add_argument("--ckpt", default="models/tft-best.ckpt", help="Đường dẫn model checkpoint")
    ap.add_argument("--output-dir", default="artifacts", help="Thư mục lưu kết quả")
    ap.add_argument("--simple", action="store_true", help="Chỉ tạo biểu đồ đơn giản")
    args = ap.parse_args()

    print("🚀 Bắt đầu dự đoán mực nước...")
    print("="*60)
    
    # Thực hiện dự đoán
    predictions, timestamps, actual_values, full_df, scalers, cfg = predict_water_level(args.cfg, args.ckpt)
    
    print(f"✓ Hoàn thành dự đoán!")
    print(f"  - Số bước dự đoán: {predictions.shape[1]} giờ ({predictions.shape[1]//24} ngày)")
    print(f"  - Số quantiles: {predictions.shape[2]}")
    print(f"  - Shape: {predictions.shape}")
    print("="*60 + "\n")
    
    # Tạo thư mục output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu predictions ra CSV
    csv_path = output_dir / "predictions.csv"
    save_predictions_to_csv(predictions, timestamps, actual_values, str(csv_path))
    
    # Vẽ biểu đồ
    if args.simple:
        plot_path = output_dir / "predictions_simple.png"
        plot_predictions(predictions, timestamps, actual_values, str(plot_path))
    else:
        print("📈 Tạo biểu đồ phân tích chi tiết...\n")
        plot_detailed_analysis(predictions, timestamps, actual_values, full_df, str(output_dir))
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH! Tất cả kết quả đã được lưu trong thư mục:", output_dir)
    print("="*60)


if __name__ == "__main__":
    main()
