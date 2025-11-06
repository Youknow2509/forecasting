from dataclasses import dataclass
import yaml

# Config model crawl module
@dataclass
class CrawlConfig:
    url_root: str = "https://hochuathuydien.evn.com.vn/PageHoChuaThuyDienEmbedEVN.aspx"
    output_dir: str = "data/crawl"
    timeout: int = 2000 # milliseconds
    vm: str = "84"
    lv: str = "23"
    hc: str = "4-47"
    time_step: int = 60 # minutes
    time_start: str = "01/01/2020 00:00"
    time_end: str = "05/11/2025 23:59"

    @classmethod
    def load(cls, path: str="config/config.yaml", section: str="crawl") -> "CrawlConfig":
        """
        Load CrawlConfig from a YAML file.
        - If the YAML has a top-level key named `section` (default: "crawl"), use that mapping.
        - Otherwise, attempt to use the top-level mapping as the config.
        Unknown keys are ignored.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # prefer the named section if present
        if isinstance(data, dict) and section in data and isinstance(data[section], dict):
            payload = data[section]
        elif isinstance(data, dict):
            payload = data
        else:
            payload = {}

        # filter only fields that exist on the dataclass
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in payload.items() if k in valid_keys}

        return cls(**filtered)