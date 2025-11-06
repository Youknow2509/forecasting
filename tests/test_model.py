"""
Unit tests for model training and inference
"""
from __future__ import annotations
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch

try:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer

from src.water_forecast.dataset import build_timeseries_datasets
from src.water_forecast.tft_model import make_tft


class TestModelTraining(unittest.TestCase):
    """Test model creation and training"""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic dataset for testing"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create synthetic time series data
        n = 500
        cls.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="H"),
            "site_id": ["S1"] * n,
            "muc_thuong_luu": np.sin(np.linspace(0, 10*np.pi, n)) + np.random.randn(n) * 0.1,
            "muc_dang_binh_thuong": 10.0 + np.random.randn(n) * 0.5,
            "muc_chet": 1.0,
            "luu_luong_den": 5.0 + np.random.randn(n) * 0.5,
            "tong_luong_xa": 1.0 + np.random.randn(n) * 0.1,
            "xa_tran": np.abs(np.random.randn(n) * 0.1),
            "xa_nha_may": 1.0 + np.random.randn(n) * 0.1,
            "so_cua_xa_sau": np.random.randint(1, 5, n),
            "so_cua_xa_mat": np.random.randint(1, 3, n),
            "hour": pd.date_range("2024-01-01", periods=n, freq="H").hour,
            "day_of_week": pd.date_range("2024-01-01", periods=n, freq="H").dayofweek,
            "day_of_month": pd.date_range("2024-01-01", periods=n, freq="H").day,
            "month": pd.date_range("2024-01-01", periods=n, freq="H").month,
            "was_imputed": 0,
        })
        
        # Add time_idx
        cls.df["time_idx"] = range(n)
        
    def test_model_creation(self):
        """Test that model can be created from dataset"""
        # Create dataset
        training, validation = build_timeseries_datasets(
            self.df, 
            enc_len=24, 
            dec_len=12
        )
        
        # Create model
        model = make_tft(
            training,
            hidden_size=32,  # Small for testing
            attention_heads=2,
            dropout=0.1,
            learning_rate=1e-3,
            quantiles=(0.1, 0.5, 0.9)
        )
        
        # Check model exists and is correct type
        self.assertIsInstance(model, TemporalFusionTransformer)
        self.assertIsNotNone(model)
        
    def test_model_forward_pass(self):
        """Test that model can perform forward pass"""
        # Create dataset
        training, _ = build_timeseries_datasets(
            self.df, 
            enc_len=24, 
            dec_len=12
        )
        
        # Create model
        model = make_tft(
            training,
            hidden_size=32,
            attention_heads=2,
            dropout=0.1,
            learning_rate=1e-3,
            quantiles=(0.1, 0.5, 0.9)
        )
        
        # Create dataloader
        train_loader = training.to_dataloader(
            train=True, 
            batch_size=8, 
            num_workers=0
        )
        
        # Get a batch
        batch = next(iter(train_loader))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch[0])
        
        # Check output shape
        self.assertIsNotNone(output)
        # Output should be [batch_size, decoder_length, num_quantiles]
        self.assertEqual(len(output.shape), 3)
        self.assertEqual(output.shape[2], 3)  # 3 quantiles
        
    def test_model_prediction(self):
        """Test that model can generate predictions"""
        # Create dataset
        training, validation = build_timeseries_datasets(
            self.df, 
            enc_len=24, 
            dec_len=12
        )
        
        # Create model
        model = make_tft(
            training,
            hidden_size=32,
            attention_heads=2,
            dropout=0.1,
            learning_rate=1e-3,
            quantiles=(0.1, 0.5, 0.9)
        )
        
        # Create dataloader
        val_loader = validation.to_dataloader(
            train=False, 
            batch_size=8, 
            num_workers=0
        )
        
        # Generate predictions
        model.eval()
        preds = model.predict(val_loader, mode="quantiles")
        
        # Check predictions
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(len(preds.shape), 3)
        self.assertEqual(preds.shape[2], 3)  # 3 quantiles
        
        # Check quantile ordering (q10 <= q50 <= q90)
        self.assertTrue(np.all(preds[:, :, 0] <= preds[:, :, 1]))
        self.assertTrue(np.all(preds[:, :, 1] <= preds[:, :, 2]))
        
    def test_model_training_step(self):
        """Test that model can perform one training step"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            training, validation = build_timeseries_datasets(
                self.df, 
                enc_len=24, 
                dec_len=12
            )
            
            # Create model
            model = make_tft(
                training,
                hidden_size=32,
                attention_heads=2,
                dropout=0.1,
                learning_rate=1e-3,
                quantiles=(0.1, 0.5, 0.9)
            )
            
            # Create dataloaders
            train_loader = training.to_dataloader(
                train=True, 
                batch_size=8, 
                num_workers=0
            )
            val_loader = validation.to_dataloader(
                train=False, 
                batch_size=8, 
                num_workers=0
            )
            
            # Create trainer for 1 epoch
            checkpoint = ModelCheckpoint(
                dirpath=tmpdir,
                filename="test-model",
                monitor="val_loss",
                mode="min"
            )
            
            trainer = Trainer(
                max_epochs=1,
                accelerator="cpu",
                devices=1,
                callbacks=[checkpoint],
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False
            )
            
            # Train for 1 epoch
            trainer.fit(model, train_loader, val_loader)
            
            # Check that checkpoint was created
            ckpt_files = list(Path(tmpdir).glob("*.ckpt"))
            self.assertGreater(len(ckpt_files), 0)
            
    def test_model_load_checkpoint(self):
        """Test that model can be saved and loaded"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            training, validation = build_timeseries_datasets(
                self.df, 
                enc_len=24, 
                dec_len=12
            )
            
            # Create model
            model = make_tft(
                training,
                hidden_size=32,
                attention_heads=2,
                dropout=0.1,
                learning_rate=1e-3,
                quantiles=(0.1, 0.5, 0.9)
            )
            
            # Create dataloaders
            train_loader = training.to_dataloader(
                train=True, 
                batch_size=8, 
                num_workers=0
            )
            val_loader = validation.to_dataloader(
                train=False, 
                batch_size=8, 
                num_workers=0
            )
            
            # Train briefly
            checkpoint = ModelCheckpoint(
                dirpath=tmpdir,
                filename="test-model",
                save_top_k=1
            )
            
            trainer = Trainer(
                max_epochs=1,
                accelerator="cpu",
                devices=1,
                callbacks=[checkpoint],
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False
            )
            
            trainer.fit(model, train_loader, val_loader)
            
            # Load model
            ckpt_path = checkpoint.best_model_path
            loaded_model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
            
            # Check model loaded correctly
            self.assertIsInstance(loaded_model, TemporalFusionTransformer)
            
            # Generate predictions with loaded model
            loaded_model.eval()
            preds = loaded_model.predict(val_loader, mode="quantiles")
            
            # Check predictions work
            self.assertIsInstance(preds, np.ndarray)
            self.assertEqual(preds.shape[2], 3)


class TestModelMetrics(unittest.TestCase):
    """Test metric calculations"""
    
    def test_mae_calculation(self):
        """Test MAE calculation"""
        y_true = np.array([[1, 2, 3], [4, 5, 6]])
        y_pred = np.array([[1.1, 2.2, 2.9], [4.1, 4.9, 6.1]])
        
        mae = np.mean(np.abs(y_true - y_pred))
        
        self.assertAlmostEqual(mae, 0.1, places=1)
        
    def test_coverage_calculation(self):
        """Test coverage calculation"""
        y = np.array([[5, 5, 5], [10, 10, 10]])
        q10 = np.array([[4, 4, 4], [9, 9, 9]])
        q90 = np.array([[6, 6, 6], [11, 11, 11]])
        
        inside = (y >= q10) & (y <= q90)
        coverage = inside.mean()
        
        self.assertEqual(coverage, 1.0)
        
    def test_pinball_loss(self):
        """Test pinball loss calculation"""
        def pinball(y, q_pred, q):
            e = y - q_pred
            return np.mean(np.maximum(q*e, (q-1)*e))
        
        y = np.array([5.0])
        q_pred = np.array([4.0])
        q = 0.5
        
        loss = pinball(y, q_pred, q)
        
        # For median (q=0.5), pinball loss should be 0.5 * |error|
        self.assertAlmostEqual(loss, 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
