"""Device precision and checkpoint helpers shared by training and generation."""

from contextlib import nullcontext
from pathlib import Path

import torch


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def precision(device, dtype):
    kind = torch.device(device).type
    if kind == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device=cpu")
    if dtype == "auto":
        dtype = "bfloat16" if kind == "cuda" and torch.cuda.is_bf16_supported() else "float16" if kind == "cuda" else "float32"
    types = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype not in types:
        raise ValueError("dtype must be auto, float32, float16, or bfloat16")
    if kind != "cuda" and dtype != "float32":
        raise ValueError("Mixed precision is supported on CUDA; use --dtype=float32 on this device")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16")
    return dtype, lambda: torch.autocast(kind, dtype=types[dtype]) if dtype != "float32" else nullcontext()


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    checkpoint["model"] = {name.removeprefix("_orig_mod."): value for name, value in checkpoint["model"].items()}
    return checkpoint


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
