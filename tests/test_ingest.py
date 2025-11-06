from __future__ import annotations
import unittest
import pandas as pd
from unittest.mock import patch
from src.water_forecast.ingest import read_one_csv, ingest_crawl_dir

UTC = "Asia/Bangkok"

class TestReadOneCsv(unittest.TestCase):

    @patch('src.water_forecast.ingest.pd.read_csv')
    def test_valid_csv(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "ten_ho": ["S1"],
            "muc_nuoc_thuong_luu": [1.0],
            "muc_nuoc_dang_binh_thuong": [2.0],
            "muc_nuoc_chet": [3.0],
            "luu_luong_den_ho": [4.0],
            "tong_luong_xa": [5.0],
            "tong_luong_xa_qua_dap_tran": [6.0],
            "tong_luong_xa_qua_nha_may": [7.0],
            "so_cua_xa_sau": [8],
            "so_cua_xa_mat": [9],
            "timestamp": ["01/01 00:00"]
        })
        df = read_one_csv("2020-01.csv", UTC)
        self.assertEqual(df.shape[0], 1)
        self.assertIn("site_id", df.columns)
        self.assertTrue(pd.api.types.is_datetime64_ns_dtype(df["timestamp"].dtype))
        self.assertEqual(df.loc[0, "site_id"], "S1")

    @patch('src.water_forecast.ingest.pd.read_csv')
    def test_missing_columns(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "muc_nuoc_thuong_luu": [1.0],
            "timestamp": ["01/01 00:00"]
        })
        with self.assertRaises(AssertionError):
            read_one_csv("2020-01.csv", UTC)

    @patch('src.water_forecast.ingest.pd.read_csv')
    def test_missing_timestamp_column(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "ten_ho": ["S1"],
            "muc_nuoc_thuong_luu": [1.0],
        })
        with self.assertRaises(AssertionError):
            read_one_csv("2020-01.csv", UTC)

    @patch('src.water_forecast.ingest.pd.read_csv')
    def test_invalid_numeric_types(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "ten_ho": ["S1"],
            "muc_nuoc_thuong_luu": ["bad"],
            "muc_nuoc_dang_binh_thuong": ["x"],
            "timestamp": ["01/01 00:00"]
        })
        df = read_one_csv("2020-01.csv", UTC)
        self.assertTrue(pd.isna(df.loc[0, "muc_thuong_luu"]))
        self.assertTrue(pd.isna(df.loc[0, "muc_dang_binh_thuong"]))

    @patch('src.water_forecast.ingest.pd.read_csv')
    def test_timestamp_without_space(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "ten_ho": ["S1"],
            "timestamp": ["01/0100:00"]
        })
        df = read_one_csv("2020_02.csv", UTC)
        self.assertEqual(df.loc[0, "timestamp"].year, 2020)
        self.assertEqual(df.loc[0, "timestamp"].month, 1)
        self.assertEqual(df.loc[0, "timestamp"].day, 1)

    @patch('src.water_forecast.ingest.pd.read_csv')
    def test_missing_numeric_columns_filled(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "ten_ho": ["S1"],
            "timestamp": ["01/01 00:00"]
        })
        df = read_one_csv("2020-03.csv", UTC)
        numeric = [
            "muc_thuong_luu","muc_dang_binh_thuong","muc_chet",
            "luu_luong_den","tong_luong_xa","xa_tran","xa_nha_may",
            "so_cua_xa_sau","so_cua_xa_mat"
        ]
        for c in numeric:
            self.assertIn(c, df.columns)
        self.assertEqual(df.loc[0, "luu_luong_den"], 0.0)

    @patch('src.water_forecast.ingest.pd.read_csv')
    def test_duplicate_rows_removed(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "ten_ho": ["S1","S1"],
            "timestamp": ["01/01 00:00","01/01 00:00"],
            "muc_nuoc_thuong_luu": [1.0, 2.0]
        })
        df = read_one_csv("2020-04.csv", UTC)
        self.assertEqual(df.shape[0], 1)
        self.assertEqual(df.loc[0, "muc_thuong_luu"], 2.0)

    @patch('src.water_forecast.ingest.pd.read_csv')
    def test_multi_site_ordering(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "ten_ho": ["B","A","A"],
            "timestamp": ["01/02 01:00","01/01 00:00","01/01 02:00"],
            "muc_nuoc_thuong_luu": [3,1,2]
        })
        df = read_one_csv("2020-05.csv", UTC)
        self.assertListEqual(df["site_id"].tolist(), ["A","A","B"])
        self.assertTrue(df["timestamp"].is_monotonic_increasing)

    @patch('src.water_forecast.ingest.pd.read_csv')
    def test_year_inference_variants(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "ten_ho": ["S1"],
            "timestamp": ["01/01 00:00"]
        })
        for fname in ["202001.csv", "2020-01.csv", "2020_01_data.csv"]:
            df = read_one_csv(fname, UTC)
            self.assertEqual(df.loc[0, "timestamp"].year, 2020)

    @patch('src.water_forecast.ingest.pd.read_csv')
    def test_empty_dataframe_asserts(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame()
        with self.assertRaises(AssertionError):
            read_one_csv("2020-06.csv", UTC)

class TestIngestCrawlDir(unittest.TestCase):

    @patch('src.water_forecast.ingest.read_one_csv')
    @patch('src.water_forecast.ingest.Path.glob')
    def test_ingest_crawl_dir_basic(self, mock_glob, mock_read_one):
        # Mock file list
        mock_glob.return_value = [None]  # placeholder to satisfy iteration if needed
        # Simulate two frames
        f1 = pd.DataFrame({
            "timestamp": pd.to_datetime(["2020-01-01T00:00Z"]),
            "site_id": ["A"],
            "muc_thuong_luu": [1.0],
            "muc_dang_binh_thuong": [0.0],
            "muc_chet": [0.0],
            "luu_luong_den": [0.0],
            "tong_luong_xa": [0.0],
            "xa_tran": [0.0],
            "xa_nha_may": [0.0],
            "so_cua_xa_sau": [0],
            "so_cua_xa_mat": [0],
        })
        f2 = f1.copy()
        mock_read_one.side_effect = [f1, f2]
        # Patch glob to return two paths
        mock_glob.return_value = [type("P", (), {"__str__": lambda self: "2020-01.csv"})(),
                                  type("P", (), {"__str__": lambda self: "2020-02.csv"})()]
        with patch('src.water_forecast.ingest.pd.concat', return_value=pd.concat([f1, f2], ignore_index=True)):
            out = ingest_crawl_dir("data/crawl", "data/out.csv")
            self.assertEqual(out, "data/out.csv")

if __name__ == "__main__":
    unittest.main()