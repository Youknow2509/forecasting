"""
Test trained model service - Comprehensive evaluation and testing
Dịch vụ test mô hình đã train với đánh giá toàn diện
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from lightning.pytorch import seed_everything
except ImportError:
    from pytorch_lightning import seed_everything

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer

from .config import load_config
from .utils import ensure_dir, load_json, save_json
from .dataio import load_csv, resample_fill
from .features import add_time_features, add_lags_rollings
from .preprocessing import train_val_test_split, fit_scalers, apply_scalers
from .dataset import build_timeseries_datasets


class ModelTester:
    """Comprehensive model testing service"""
    
    def __init__(self, cfg_path: str, ckpt_path: str | None = None):
        self.cfg = load_config(cfg_path)
        seed_everything(self.cfg.seed)
        
        # Load metadata
        self.meta = load_json(self.cfg.paths.metadata_json)
        self.quantiles = np.array(self.meta.get("quantiles", self.cfg.quantiles))
        
        # Set checkpoint path
        self.ckpt_path = ckpt_path or self.cfg.paths.best_ckpt
        
        # Create output directory
        self.output_dir = Path(self.cfg.paths.artifacts_dir) / "test_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize placeholders
        self.model = None
        self.test_loader = None
        self.predictions = None
        self.actuals = None
        self.metrics = {}
        
    def load_data(self):
        """Load and preprocess data exactly like training"""
        print("📊 Loading and preprocessing data...")
        
        df = load_csv(self.cfg.paths.data_csv, timezone=self.cfg.timezone)
        df = resample_fill(df, freq=self.cfg.frequency, tz_name=self.cfg.timezone)
        df = add_time_features(df)
        df = add_lags_rollings(
            df, 
            target="muc_thuong_luu",
            lags=self.cfg.lags_hours,
            roll_windows=self.cfg.roll_windows_hours,
            roll_stats=self.cfg.roll_stats
        )
        
        # Split data
        tr, va, te = train_val_test_split(
            df, 
            self.cfg.split.train_ratio, 
            self.cfg.split.val_ratio
        )
        
        print(f"  Train: {len(tr)} samples")
        print(f"  Val:   {len(va)} samples")
        print(f"  Test:  {len(te)} samples")
        
        # Apply scaling
        scalers = fit_scalers(tr)
        tr_s = apply_scalers(tr, scalers)
        va_s = apply_scalers(va, scalers)
        te_s = apply_scalers(te, scalers)
        full = pd.concat([tr_s, va_s, te_s], ignore_index=True)
        
        # Build datasets
        training, validation = build_timeseries_datasets(
            full, 
            self.cfg.enc_len, 
            self.cfg.dec_len
        )
        
        # Create test loader
        test = TimeSeriesDataSet.from_dataset(
            training, 
            full, 
            predict=True, 
            stop_randomization=True
        )
        self.test_loader = test.to_dataloader(
            train=False, 
            batch_size=self.cfg.batch_size, 
            num_workers=4
        )
        
        print(f"✅ Data loaded successfully")
        return test
        
    def load_model(self):
        """Load trained model from checkpoint"""
        print(f"🤖 Loading model from: {self.ckpt_path}")
        
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")
        
        self.model = TemporalFusionTransformer.load_from_checkpoint(self.ckpt_path)
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        
    def predict(self, test_dataset):
        """Generate predictions"""
        print("🔮 Generating predictions...")
        
        preds, idx = self.model.predict(
            self.test_loader, 
            return_index=True, 
            mode="quantiles"
        )
        
        # preds shape: [N, horizon, n_quantiles]
        self.predictions = {
            'q10': preds[:, :, 0],
            'q50': preds[:, :, 1],
            'q90': preds[:, :, 2],
            'indices': idx
        }
        
        # Get actual values
        try:
            y = self.model.to_prediction(self.test_loader).get("target")
            if y is None:
                y = test_dataset.get_target()
            self.actuals = y.numpy() if hasattr(y, 'numpy') else y
        except Exception as e:
            print(f"⚠️  Warning: Could not extract actuals: {e}")
            self.actuals = test_dataset.get_target()
        
        print(f"✅ Predictions generated: {preds.shape}")
        
    def compute_metrics(self):
        """Compute comprehensive evaluation metrics"""
        print("📈 Computing metrics...")
        
        y = self.actuals
        q10 = self.predictions['q10']
        q50 = self.predictions['q50']
        q90 = self.predictions['q90']
        
        # Basic metrics
        mae = np.mean(np.abs(y - q50))
        rmse = np.sqrt(np.mean((y - q50)**2))
        mape = np.mean(np.abs((y - q50) / (y + 1e-8))) * 100
        
        # Coverage of 80% prediction interval
        inside = (y >= q10) & (y <= q90)
        coverage_80 = inside.mean() * 100
        
        # Horizon-wise metrics
        H = q50.shape[1]
        
        def horizon_metrics(h):
            h = min(h, H)
            y_h = y[:, :h]
            q50_h = q50[:, :h]
            mae_h = np.mean(np.abs(y_h - q50_h))
            rmse_h = np.sqrt(np.mean((y_h - q50_h)**2))
            return mae_h, rmse_h
        
        mae_24h, rmse_24h = horizon_metrics(24)
        mae_7d, rmse_7d = horizon_metrics(24*7)
        mae_14d, rmse_14d = horizon_metrics(24*14)
        
        # Pinball loss for each quantile
        def pinball_loss(y, q_pred, q):
            e = y - q_pred
            return np.mean(np.maximum(q*e, (q-1)*e))
        
        pinball_10 = pinball_loss(y, q10, 0.1)
        pinball_50 = pinball_loss(y, q50, 0.5)
        pinball_90 = pinball_loss(y, q90, 0.9)
        avg_pinball = (pinball_10 + pinball_50 + pinball_90) / 3
        
        # Interval width
        interval_width = np.mean(q90 - q10)
        
        # Sharpness (narrower intervals are better if coverage is maintained)
        sharpness = np.std(q90 - q10)
        
        self.metrics = {
            'overall': {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'MAPE': float(mape),
                'Coverage_80%': float(coverage_80),
                'Interval_Width': float(interval_width),
                'Sharpness': float(sharpness),
            },
            'horizon_wise': {
                'MAE_24h': float(mae_24h),
                'RMSE_24h': float(rmse_24h),
                'MAE_7d': float(mae_7d),
                'RMSE_7d': float(rmse_7d),
                'MAE_14d': float(mae_14d),
                'RMSE_14d': float(rmse_14d),
            },
            'quantile_loss': {
                'Pinball_Q10': float(pinball_10),
                'Pinball_Q50': float(pinball_50),
                'Pinball_Q90': float(pinball_90),
                'Avg_Pinball': float(avg_pinball),
            },
            'metadata': {
                'n_samples': int(y.shape[0]),
                'horizon_steps': int(H),
                'test_date': datetime.now().isoformat(),
                'checkpoint': self.ckpt_path,
            }
        }
        
        print(f"✅ Metrics computed")
        return self.metrics
    
    def print_metrics(self):
        """Print metrics in a formatted way"""
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION RESULTS")
        print("="*60)
        
        print("\n🎯 Overall Performance:")
        for key, value in self.metrics['overall'].items():
            print(f"  {key:20s}: {value:8.4f}")
        
        print("\n⏱️  Horizon-wise Performance:")
        for key, value in self.metrics['horizon_wise'].items():
            print(f"  {key:20s}: {value:8.4f}")
        
        print("\n📉 Quantile Loss:")
        for key, value in self.metrics['quantile_loss'].items():
            print(f"  {key:20s}: {value:8.4f}")
        
        print("\n📋 Test Metadata:")
        for key, value in self.metrics['metadata'].items():
            print(f"  {key:20s}: {value}")
        
        print("="*60 + "\n")
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = self.output_dir / "metrics.json"
        save_json(self.metrics, str(metrics_file))
        
        # Also save as readable text
        txt_file = self.output_dir / "metrics.txt"
        with open(txt_file, "w") as f:
            f.write("MODEL EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("Overall Performance:\n")
            for k, v in self.metrics['overall'].items():
                f.write(f"  {k}: {v:.4f}\n")
            
            f.write("\nHorizon-wise Performance:\n")
            for k, v in self.metrics['horizon_wise'].items():
                f.write(f"  {k}: {v:.4f}\n")
            
            f.write("\nQuantile Loss:\n")
            for k, v in self.metrics['quantile_loss'].items():
                f.write(f"  {k}: {v:.4f}\n")
        
        print(f"💾 Metrics saved to: {metrics_file}")
    
    def plot_predictions(self, n_samples=5):
        """Plot sample predictions with uncertainty bands"""
        print(f"📊 Generating prediction plots...")
        
        y = self.actuals
        q10 = self.predictions['q10']
        q50 = self.predictions['q50']
        q90 = self.predictions['q90']
        H = q50.shape[1]
        
        n_samples = min(n_samples, y.shape[0])
        
        # Create subplots
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i in range(n_samples):
            ax = axes[i]
            x = np.arange(H)
            
            # Plot actual values
            ax.plot(x, y[i], 'k-', label='Actual', linewidth=2)
            
            # Plot median prediction
            ax.plot(x, q50[i], 'b-', label='Predicted (Q50)', linewidth=2)
            
            # Plot uncertainty band
            ax.fill_between(x, q10[i], q90[i], alpha=0.3, color='blue', 
                           label='80% Prediction Interval (Q10-Q90)')
            
            ax.set_xlabel('Forecast Horizon (hours)')
            ax.set_ylabel('Water Level (scaled)')
            ax.set_title(f'Sample {i+1}: Forecast with Uncertainty')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "forecast_samples.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Forecast plots saved to: {plot_file}")
    
    def plot_error_distribution(self):
        """Plot error distribution analysis"""
        print(f"📊 Generating error distribution plots...")
        
        y = self.actuals
        q50 = self.predictions['q50']
        errors = (y - q50).flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histogram of errors
        axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error vs actual
        axes[1, 0].scatter(y.flatten(), errors, alpha=0.3, s=10)
        axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Actual Value')
        axes[1, 0].set_ylabel('Prediction Error')
        axes[1, 0].set_title('Error vs Actual Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error by horizon
        H = y.shape[1]
        mae_by_horizon = np.mean(np.abs(y - q50), axis=0)
        axes[1, 1].plot(range(H), mae_by_horizon, 'b-', linewidth=2)
        axes[1, 1].set_xlabel('Forecast Horizon (hours)')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Error by Forecast Horizon')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "error_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Error analysis saved to: {plot_file}")
    
    def plot_coverage_analysis(self):
        """Plot prediction interval coverage analysis"""
        print(f"📊 Generating coverage analysis...")
        
        y = self.actuals
        q10 = self.predictions['q10']
        q90 = self.predictions['q90']
        H = y.shape[1]
        
        # Coverage by horizon
        inside = (y >= q10) & (y <= q90)
        coverage_by_horizon = np.mean(inside, axis=0) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Coverage by horizon
        axes[0].plot(range(H), coverage_by_horizon, 'b-', linewidth=2)
        axes[0].axhline(80, color='red', linestyle='--', linewidth=2, 
                       label='Target Coverage (80%)')
        axes[0].set_xlabel('Forecast Horizon (hours)')
        axes[0].set_ylabel('Coverage (%)')
        axes[0].set_title('Prediction Interval Coverage by Horizon')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 100])
        
        # 2. Interval width by horizon
        width_by_horizon = np.mean(q90 - q10, axis=0)
        axes[1].plot(range(H), width_by_horizon, 'g-', linewidth=2)
        axes[1].set_xlabel('Forecast Horizon (hours)')
        axes[1].set_ylabel('Interval Width')
        axes[1].set_title('Prediction Interval Width by Horizon')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "coverage_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Coverage analysis saved to: {plot_file}")
    
    def run_full_test(self):
        """Run complete testing pipeline"""
        print("\n🚀 Starting comprehensive model testing...\n")
        
        # Load data
        test_dataset = self.load_data()
        
        # Load model
        self.load_model()
        
        # Generate predictions
        self.predict(test_dataset)
        
        # Compute metrics
        self.compute_metrics()
        
        # Print results
        self.print_metrics()
        
        # Save metrics
        self.save_metrics()
        
        # Generate visualizations
        self.plot_predictions(n_samples=5)
        self.plot_error_distribution()
        self.plot_coverage_analysis()
        
        print(f"\n✅ Testing complete! Results saved to: {self.output_dir}\n")
        
        return self.metrics


def main():
    parser = argparse.ArgumentParser(
        description="Test trained water forecasting model"
    )
    parser.add_argument(
        "--cfg", 
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to model checkpoint (default: use best checkpoint from config)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sample predictions to plot"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = ModelTester(args.cfg, args.ckpt)
    
    # Run full test
    metrics = tester.run_full_test()
    
    return metrics


if __name__ == "__main__":
    main()
