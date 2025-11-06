from src.crawl.config import CrawlConfig
from src.crawl.model import DataCrawlRecord
from src.crawl.table_parser import parse_html_to_records
from src.crawl.utils import dataclass_to_row
from loguru import logger
import os
import csv
from datetime import datetime, timedelta
from pathlib import Path
import requests
import time
from typing import List
from time import perf_counter


# =====================================================
# Main entry point for the crawl module
# =====================================================
def main():
    # Logger
    logger.add("logs/app.log")
    # Load configuration
    config = CrawlConfig.load("configs/crawl.yaml")
    logger.info("Crawl configuration loaded:")
    logger.info(config)
    logger.info("Crawl module started.")
    
    # Parse start / end times (expected format "DD/MM/YYYY HH:MM")
    time_fmt = "%d/%m/%Y %H:%M"
    try:
        time_start = datetime.strptime(config.time_start, time_fmt)
        time_end = datetime.strptime(config.time_end, time_fmt)
    except Exception as e:
        logger.error(f"Invalid time format in config: {e}")
        return

    step = timedelta(minutes=int(config.time_step))
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    # optional: set a user-agent to avoid basic bot blocks
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36"
    })
    timeout_sec = float(getattr(config, "timeout", 0)) / 1000.0 if getattr(config, "timeout", 0) else None

    # iterate timestamps and write per-month files
    current = time_start
    while current <= time_end:
        month_key = current.strftime("%Y-%m")
        out_file = out_dir / f"{month_key}.csv"
        file_exists = out_file.exists()

        # prepare params - include known config keys
        params = {
            "vm": config.vm,
            "lv": config.lv,
            "hc": config.hc,
            "td": current.strftime(time_fmt),
        }

        logger.debug(f"HTTP GET -> {config.url_root} | params={params}")
        t0 = perf_counter()
        try:
            resp = session.get(config.url_root, params=params, timeout=timeout_sec)
            status = resp.status_code
            content = resp.text
        except Exception as e:
            logger.warning(f"Request failed for {current.isoformat()}: {e}")
            status = None
            content = f"ERROR: {e}"
        else:
            dt = (perf_counter() - t0) * 1000.0
            url_eff = getattr(resp, 'url', config.url_root)
            clen = len(content) if isinstance(content, str) else 0
            logger.debug(f"HTTP RESP <- status={status} in {dt:.1f}ms | url={url_eff} | content_len={clen}")

        # Try to parse HTML table into structured records
        records: List[DataCrawlRecord] = []
        if content and isinstance(content, str) and not content.startswith("ERROR:"):
            try:
                records = parse_html_to_records(content)
            except Exception as e:
                logger.warning(f"Failed to parse HTML for {current.isoformat()}: {e}")
            else:
                logger.debug(f"Parsed records count={len(records)} for ts={current.strftime('%Y-%m-%d %H:%M')}")
                if records:
                    try:
                        logger.debug(f"First record: {dataclass_to_row(records[0])}")
                    except Exception:
                        pass

        # ensure directory
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if records:
            # write structured CSV using DataCrawlRecord fieldnames
            header = [f for f in DataCrawlRecord.__annotations__.keys()]
            logger.debug(f"Writing {len(records)} records to {out_file}")
            with open(out_file, "a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                for rec in records:
                    writer.writerow(dataclass_to_row(rec))
        else:
            # fallback: write raw content row (one-per-timestamp)
            header = ["timestamp", "status_code", "params", "content"]
            row = {
                "timestamp": current.strftime("%Y-%m-%d %H:%M"),
                "status_code": status,
                "params": str(params),
                # avoid logging/writing huge payloads
                "content": (content[:5000] + "…") if isinstance(content, str) and len(content) > 5000 else (content.replace("\n", " ").strip() if isinstance(content, str) else str(content))
            }
            logger.debug(f"No parsed records, writing raw snapshot to {out_file}")
            with open(out_file, "a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        # advance time
        current += step
        # time sleep between requests (config may not define timeout)
        time_sleep_ms = float(getattr(config, "timeout", 0))
        if time_sleep_ms > 0:
            time.sleep(time_sleep_ms / 1000.0)

    logger.info("Crawl module finished.")

if __name__ == "__main__":
    main()