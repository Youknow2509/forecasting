from typing import List
from bs4 import BeautifulSoup
from .model import DataCrawlRecord
from .utils import try_float, try_int, normalize_text


def parse_html_to_records(html: str) -> List[DataCrawlRecord]:
    """Parse EVN reservoir table rows into DataCrawlRecord.

    Supports HTML snippets that contain only <tbody>. It expects data rows like:
      [ten_ho_cell] [timestamp] [Htl] [Hdbt] [Hc] [Qve] [ΣQx] [Qxt] [Qxm] [Ncxs] [Ncxm]
    The first cell contains the reservoir name (often inside <b>), possibly followed by
    a <small> with sync info (ignored). Rows that are group headers with large colspan are skipped.
    """
    soup = BeautifulSoup(html, 'html.parser')
    container = (
        soup.find(class_='tblgridtd')
        or soup.find('table')
        or soup.find('tbody')
        or soup
    )

    rows = container.find_all('tr')
    records: List[DataCrawlRecord] = []

    for r in rows:
        cells = r.find_all(['td', 'th'])
        if not cells:
            continue
        if len(cells) == 1 and int(cells[0].get('colspan') or 1) >= 5:
            continue
        if len(cells) < 11:
            continue

        # Skip header-like rows (e.g., "Tên hồ", "Thời điểm", "Chú thích ký hiệu")
        c0 = normalize_text(cells[0].get_text(strip=True))
        c1 = normalize_text(cells[1].get_text(strip=True)) if len(cells) > 1 else ""
        if c0 in {"ten ho", "chu thich ky hieu"} or c1 in {"thoi diem"}:
            continue

        # Heuristic: ensure at least one numeric value exists in data columns
        numeric_count = 0
        for idx_chk in range(2, min(11, len(cells))):
            t = cells[idx_chk].get_text(strip=True)
            try:
                if t != "":
                    float(t.replace(",", ""))
                    numeric_count += 1
            except Exception:
                pass
        if numeric_count == 0:
            # likely a non-data row
            continue

        name_tag = cells[0].find('b')
        ten_ho = (name_tag.get_text(strip=True) if name_tag else cells[0].get_text(strip=True))
        timestamp = cells[1].get_text(strip=True)

        rec = DataCrawlRecord(
            ten_ho=ten_ho,
            timestamp=timestamp,
            muc_nuoc_thuong_luu=try_float(cells[2].get_text(strip=True)),
            muc_nuoc_dang_binh_thuong=try_float(cells[3].get_text(strip=True)),
            muc_nuoc_chet=try_float(cells[4].get_text(strip=True)),
            luu_luong_den_ho=try_float(cells[5].get_text(strip=True)),
            tong_luong_xa=try_float(cells[6].get_text(strip=True)),
            tong_luong_xa_qua_dap_tran=try_float(cells[7].get_text(strip=True)),
            tong_luong_xa_qua_nha_may=try_float(cells[8].get_text(strip=True)),
            so_cua_xa_sau=try_int(cells[9].get_text(strip=True)) or 0,
            so_cua_xa_mat=try_int(cells[10].get_text(strip=True)) or 0,
        )
        records.append(rec)

    return records
