from __future__ import annotations
import argparse, os
from pathlib import Path
import pandas as pd
import re

VN2EN = {
    "ten_ho": "site_id",
    "muc_nuoc_thuong_luu": "muc_thuong_luu",
    "muc_nuoc_dang_binh_thuong": "muc_dang_binh_thuong",
    "muc_nuoc_chet": "muc_chet",
    "luu_luong_den_ho": "luu_luong_den",
    "tong_luong_xa": "tong_luong_xa",
    "tong_luong_xa_qua_dap_tran": "xa_tran",
    "tong_luong_xa_qua_nha_may": "xa_nha_may",
    "so_cua_xa_sau": "so_cua_xa_sau",
    "so_cua_xa_mat": "so_cua_xa_mat",
}

KEEP_COLS = [
    "timestamp","site_id","muc_thuong_luu","muc_dang_binh_thuong","muc_chet",
    "luu_luong_den","tong_luong_xa","xa_tran","xa_nha_may","so_cua_xa_sau","so_cua_xa_mat"
]


def _infer_year_from_name(name: str) -> int:
    m = re.search(r"(20[0-9]{2})[-_]?[0-9]{2}", name)
    if not m:
        m = re.search(r"(20[0-9]{2})", name)
    if not m:
        raise ValueError(f"Cannot infer year from filename: {name}")
    return int(m.group(1))


def _parse_timestamp_series(raw_ts: pd.Series, year: int, tz: str) -> pd.Series:
    # Parse local time like '01/01 00:00' -> naive datetime, then set year.
    ts = pd.to_datetime(raw_ts, format="%d/%m %H:%M", errors="coerce")
    if ts.isna().any():
        ts2 = pd.to_datetime(raw_ts.astype(str).str.replace(" ", "", regex=False),
                             format="%d/%m%H:%M", errors="coerce")
        ts = ts.fillna(ts2)
    ts = ts.map(lambda x: x.replace(year=year) if pd.notna(x) else x)
    # Keep as naive local time; we'll localize/convert at export.
    return ts


def read_one_csv(path: str, tz: str) -> pd.DataFrame:
    year = _infer_year_from_name(os.path.basename(path))
    df = pd.read_csv(path, on_bad_lines='skip')
    # Standardize headers
    cols = {c: VN2EN.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    assert "timestamp" in df.columns, "missing 'timestamp' column in crawl file"
    assert "site_id" in df.columns, "missing 'ten_ho' column in crawl file"

    # Build UTC timestamp from local representation
    df["timestamp"] = _parse_timestamp_series(df["timestamp"].astype(str), year, tz)

    # Force dtypes and create missing numeric columns as zeros
    num_map = {
        "muc_thuong_luu": float,
        "muc_dang_binh_thuong": float,
        "muc_chet": float,
        "luu_luong_den": float,
        "tong_luong_xa": float,
        "xa_tran": float,
        "xa_nha_may": float,
        "so_cua_xa_sau": "Int64",
        "so_cua_xa_mat": "Int64",
    }
    for c, dt in num_map.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(dt)
        else:
            df[c] = pd.Series([0]*len(df), dtype=dt if isinstance(dt, str) else float)

    # Select & order, dedup
    df = df[KEEP_COLS]
    df = df.drop_duplicates(subset=["site_id", "timestamp"], keep="last").sort_values(["site_id","timestamp"]).reset_index(drop=True)
    return df

def clean_files_in_dir(folder_path):
    target_str = "{'vm': '84', 'lv': '23', 'hc': '4-47'"
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Chỉ xử lý file thường (bỏ qua thư mục)
        if not os.path.isfile(file_path):
            continue
        
        # Đọc tất cả dòng
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Giữ lại những dòng không chứa chuỗi cần xoá
        new_lines = [line for line in lines if target_str not in line]
        
        # Nếu có thay đổi → ghi đè lại file
        if len(new_lines) != len(lines):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"Đã xoá dòng chứa chuỗi trong file: {filename}")
        else:
            print(f"Không có dòng cần xoá trong file: {filename}")

def ingest_crawl_dir(crawl_dir: str, out_path: str, tz: str = "Asia/Bangkok") -> str:
    paths = sorted([str(p) for p in Path(crawl_dir).glob("*.csv")])
    if not paths:
        raise FileNotFoundError(f"No CSV files in {crawl_dir}")
    frames = [read_one_csv(p, tz) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["site_id","timestamp"], keep="last").sort_values(["site_id","timestamp"]).reset_index(drop=True)
    # ISO UTC strings
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl_dir", required=True)
    ap.add_argument("--out", default="data/sample.csv")
    ap.add_argument("--tz", default="Asia/Bangkok")
    args = ap.parse_args()

    p = ingest_crawl_dir(args.crawl_dir, args.out, args.tz)
    print("Wrote:", p)

if __name__ == "__main__":
    main()
    # clean_files_in_dir("data/crawl")