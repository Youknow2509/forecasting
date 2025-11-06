from typing import List
from bs4 import BeautifulSoup
from .model import DataCrawlRecord
from .utils import try_float, try_int


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
        # Skip region/group rows (one cell spanning many columns)
        if len(cells) == 1 and int(cells[0].get('colspan') or 1) >= 5:
            continue
        # Data rows should have at least 11 columns
        if len(cells) < 11:
            continue

        # Extract name from first cell (<b> preferred)
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
from typing import List, Optional
from bs4 import BeautifulSoup
from .model import DataCrawlRecord
from .utils import try_float, try_int, normalize_text
from itertools import zip_longest
import re

def header_to_field(h: str) -> Optional[str]:
    """Try to map a header string to DataCrawlRecord field name.

    Kept here for clarity; it uses normalize_text from utils.
    """
    h_norm = normalize_text(h)
    if 'ten' in h_norm:
        return 'ten_ho'
    if 'thoi' in h_norm or 'ngay' in h_norm or 'gio' in h_norm or 'timestamp' in h_norm:
        return 'timestamp'
    if 'thuong' in h_norm and 'nuoc' in h_norm:
        return 'muc_nuoc_thuong_luu'
    if ('dang' in h_norm or 'binh' in h_norm) and 'nuoc' in h_norm:
        return 'muc_nuoc_dang_binh_thuong'
    if 'chet' in h_norm and 'nuoc' in h_norm:
        return 'muc_nuoc_chet'
    if 'luu' in h_norm and 'den' in h_norm:
        return 'luu_luong_den_ho'
    if 'tong' in h_norm and 'dap' in h_norm:
        return 'tong_luong_xa_qua_dap_tran'
    if 'tong' in h_norm and 'nha' in h_norm:
        return 'tong_luong_xa_qua_nha_may'
    if 'tong' in h_norm and 'xa' in h_norm:
        return 'tong_luong_xa'
    if ('so' in h_norm or 'so_cua' in h_norm) and 'sau' in h_norm:
        return 'so_cua_xa_sau'
    if ('so' in h_norm or 'so_cua' in h_norm) and ('mat' in h_norm or 'mất' in h_norm):
        return 'so_cua_xa_mat'
    if h_norm in DataCrawlRecord.__annotations__:
        return h_norm
    return None


def _expand_header_row(row):
    """Expand a header row into a list of texts, respecting colspan."""
    out = []
    for cell in row.find_all(['th', 'td']):
        text = cell.get_text(strip=True)
        colspan = int(cell.get('colspan') or 1)
        # repeat the text for colspan times
        out.extend([text] * colspan)
    return out


def _is_data_row(cells_texts, num_columns):
    """Heuristic: data row should have at least 2 numeric cells or a timestamp pattern."""
    if not cells_texts:
        return False
    # timestamp like 6/11/25 or 06/11/2025
    ts_re = re.compile(r"\d{1,2}/\d{1,2}/(\d{2}|\d{4})")
    joined = ' '.join(cells_texts)
    if ts_re.search(joined):
        return True
    num_numeric = 0
    for c in cells_texts:
        if c is None or c == '':
            continue
        # numeric if can parse float after removing commas
        s = c.replace(',', '').replace('–', '').strip()
        try:
            float(s)
            num_numeric += 1
        except Exception:
            pass
    return num_numeric >= max(2, num_columns - 6)


def parse_html_to_records(html: str) -> List[DataCrawlRecord]:
    """Parse the first table with class 'tblgridtd' into DataCrawlRecord list.

    This version handles multi-row headers (with colspan) and skips group/title rows.
    It also handles rows where the first cell contains a <b> name and optional <small> timestamp
    while the rest of the cells on the same row hold the numeric data.
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find(class_='tblgridtd')
    if table is None:
        table = soup.find('table')
    if table is None:
        return []

    rows = table.find_all('tr')
    if not rows:
        return []

    # Determine header rows (take up to first 3 rows that look like headers)
    header_row_count = 1
    for idx in range(min(3, len(rows))):
        def parse_html_to_records(html: str) -> List[DataCrawlRecord]:
            """Parse EVN reservoir table rows into DataCrawlRecord.

            Handles snippets that contain only <tbody> and rows where the first cell
            holds the reservoir name (with optional <small> sync text), followed by
            timestamp and numeric columns in a fixed order:
              [ten_ho] [timestamp] [Htl] [Hdbt] [Hc] [Qve] [ΣQx] [Qxt] [Qxm] [Ncxs] [Ncxm]
            Rows like region headers with big colspan are ignored.
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
                # ignore region header/group rows
                if len(cells) == 1 and int(cells[0].get('colspan') or 1) >= 5:
                    continue

                # expect at least 11 columns in data rows
                if len(cells) < 11:
                    continue

                # first cell -> name (prefer <b> text)
                name_tag = cells[0].find('b')
                ten_ho = (name_tag.get_text(strip=True) if name_tag else cells[0].get_text(strip=True))
                timestamp = cells[1].get_text(strip=True)

                muc_nuoc_thuong_luu = try_float(cells[2].get_text(strip=True))
                muc_nuoc_dang_binh_thuong = try_float(cells[3].get_text(strip=True))
                muc_nuoc_chet = try_float(cells[4].get_text(strip=True))
                luu_luong_den_ho = try_float(cells[5].get_text(strip=True))
                tong_luong_xa = try_float(cells[6].get_text(strip=True))
                tong_luong_xa_qua_dap_tran = try_float(cells[7].get_text(strip=True))
                tong_luong_xa_qua_nha_may = try_float(cells[8].get_text(strip=True))
                so_cua_xa_sau = try_int(cells[9].get_text(strip=True)) or 0
                so_cua_xa_mat = try_int(cells[10].get_text(strip=True)) or 0

                rec = DataCrawlRecord(
                    ten_ho=ten_ho,
                    timestamp=timestamp,
                    muc_nuoc_thuong_luu=muc_nuoc_thuong_luu,
                    muc_nuoc_dang_binh_thuong=muc_nuoc_dang_binh_thuong,
                    muc_nuoc_chet=muc_nuoc_chet,
                    luu_luong_den_ho=luu_luong_den_ho,
                    tong_luong_xa=tong_luong_xa,
                    tong_luong_xa_qua_dap_tran=tong_luong_xa_qua_dap_tran,
                    tong_luong_xa_qua_nha_may=tong_luong_xa_qua_nha_may,
                    so_cua_xa_sau=so_cua_xa_sau,
                    so_cua_xa_mat=so_cua_xa_mat,
                )
                records.append(rec)

            return records
