from typing import Optional
import unicodedata
from .model import DataCrawlRecord


def try_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return None
        s = s.replace(",", "")
        return float(s)
    except Exception:
        return None


def try_int(v) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        s = str(v).strip()
        if s == "":
            return None
        s = s.replace(",", "")
        return int(float(s))
    except Exception:
        return None


def normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = ''.join(ch for ch in s if ch.isalnum() or ch.isspace())
    return s.lower().strip()


def dataclass_to_row(rec: DataCrawlRecord) -> dict:
    return {
        "ten_ho": rec.ten_ho,
        "timestamp": rec.timestamp,
        "muc_nuoc_thuong_luu": try_float(rec.muc_nuoc_thuong_luu),
        "muc_nuoc_dang_binh_thuong": try_float(rec.muc_nuoc_dang_binh_thuong),
        "muc_nuoc_chet": try_float(rec.muc_nuoc_chet),
        "luu_luong_den_ho": try_float(rec.luu_luong_den_ho),
        "tong_luong_xa": try_float(rec.tong_luong_xa),
        "tong_luong_xa_qua_dap_tran": try_float(rec.tong_luong_xa_qua_dap_tran),
        "tong_luong_xa_qua_nha_may": try_float(rec.tong_luong_xa_qua_nha_may),
        "so_cua_xa_sau": try_int(rec.so_cua_xa_sau),
        "so_cua_xa_mat": try_int(rec.so_cua_xa_mat),
    }
