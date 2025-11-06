from __future__ import annotations
import json, os
from pathlib import Path

def ensure_dir(p: str | Path) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)