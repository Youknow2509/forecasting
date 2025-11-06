from __future__ import annotations
import torch
from pytorch_forecasting.models import TemporalFusionTransformer

# NOTE: TFT is complex; ONNX export may fail depending on ops. This script
# attempts TorchScript as a portable fallback.

def main(ckpt="models/tft-best.ckpt"):
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    ts_model = model.to_torchscript(method="script")
    torch.jit.save(ts_model, "models/tft-best.ts")
    print("Saved TorchScript to models/tft-best.ts")

if __name__ == "__main__":
    main()