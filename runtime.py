"""Device precision and checkpoint helpers shared by training and generation."""

from contextlib import nullcontext
from pathlib import Path
import re
import uuid

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


TRAINING_KEYS = ("optimizer", "scaler", "rng_states")


def training_state_path(path, filename):
    if not isinstance(filename, str) or not re.fullmatch(r"[^/\\]+\.train-[0-9a-f]{32}\.pt", filename):
        raise ValueError("Invalid training state filename")
    return Path(path).parent / filename


def load_checkpoint(path, device="cpu", training=False):
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if training and "training_state_file" in checkpoint:
        state_path = training_state_path(path, checkpoint["training_state_file"])
        state = torch.load(state_path, map_location=device, weights_only=True)
        if state.get("checkpoint_id") != checkpoint.get("checkpoint_id") or state.get("iter_num") != checkpoint.get("iter_num"):
            raise ValueError("Model weights and training state belong to different checkpoints")
        if "optimizer" not in state:
            raise ValueError("Training state is missing the optimizer")
        checkpoint.update({key: state[key] for key in TRAINING_KEYS if key in state})
    if not training:
        for key in TRAINING_KEYS:
            checkpoint.pop(key, None)
    checkpoint["model"] = {name.removeprefix("_orig_mod."): value for name, value in checkpoint["model"].items()}
    return checkpoint


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint)
    state_path = None
    if "optimizer" in payload:
        identifier = uuid.uuid4().hex
        state_path = path.with_name(f"{path.stem}.train-{identifier}.pt")
        state = {key: payload.pop(key) for key in TRAINING_KEYS if key in payload}
        state.update(checkpoint_id=identifier, iter_num=payload["iter_num"])
        payload.update(checkpoint_id=identifier, training_state_file=state_path.name)
        _atomic_save(state_path, state)
    # Publish weights only after their matching training state has been written.
    _atomic_save(path, payload)
    for candidate in path.parent.glob(f"{path.stem}.train-*.pt"):
        if candidate != state_path and re.fullmatch(r"[^/\\]+\.train-[0-9a-f]{32}\.pt", candidate.name):
            candidate.unlink(missing_ok=True)


def _atomic_save(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
