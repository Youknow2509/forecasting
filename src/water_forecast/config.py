from __future__ import annotations
import os, yaml, random, numpy as np, torch
from dataclasses import dataclass

@dataclass
class CFG:
    d: dict
    def __getattr__(self, item):
        v = self.d.get(item)
        if isinstance(v, dict):
            return CFG(v)
        return v


def load_config(path: str = "configs/default.yaml") -> CFG:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # set seeds
    seed = d.get("seed", 3407)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return CFG(d)