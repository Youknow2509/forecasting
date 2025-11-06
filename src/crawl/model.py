
from dataclasses import dataclass


@dataclass
class DataCrawlRecord:
    ten_ho: str
    timestamp: str
    muc_nuoc_thuong_luu: float
    muc_nuoc_dang_binh_thuong: float
    muc_nuoc_chet: float
    luu_luong_den_ho: float
    tong_luong_xa: float
    tong_luong_xa_qua_dap_tran: float
    tong_luong_xa_qua_nha_may: float
    so_cua_xa_sau: int
    so_cua_xa_mat: int
    