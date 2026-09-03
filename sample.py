"""Generate text from a trained checkpoint or pretrained GPT-2 weights."""

from dataclasses import dataclass
from pathlib import Path

import torch

from dataset import Tokenizer, read_metadata
from model import GPT, GPTConfig
from runtime import default_device, load_checkpoint, precision
from settings import load_settings


@dataclass
class SampleConfig:
    """Text generation settings."""
    init_from: str = "resume"
    out_dir: str = "out"
    checkpoint: str = "ckpt.pt"
    start: str = "\n"
    num_samples: int = 1
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False


def sample(config):
    if config.num_samples <= 0 or config.max_new_tokens < 0 or config.top_k < 0:
        raise ValueError("Invalid sample count, token count, or top_k")
    device = default_device() if config.device == "auto" else config.device
    if torch.device(device).type == "cuda":
        if torch.device(device).index is None:
            device = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.set_device(device)
    _, autocast = precision(device, config.dtype)
    torch.manual_seed(config.seed)
    if config.init_from == "resume":
        checkpoint = load_checkpoint(Path(config.out_dir) / config.checkpoint)
        model = GPT(GPTConfig(**checkpoint["model_args"]))
        model.load_state_dict(checkpoint["model"])
        metadata = checkpoint.get("tokenizer")
        if metadata is None:
            settings = checkpoint.get("config", {})
            metadata = read_metadata(Path(settings.get("data_dir", "data")) / settings.get("dataset", ""))
        del checkpoint
    elif config.init_from.startswith("gpt2"):
        model = GPT.from_pretrained(config.init_from)
        metadata = {"kind": "gpt2", "vocab_size": 50257}
    else:
        raise ValueError("init_from must be resume or a GPT-2 model name")
    model.to(device).eval()
    tokenizer = Tokenizer(metadata)
    prompt = Path(config.start[5:]).read_text(encoding="utf-8") if config.start.startswith("FILE:") else config.start
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if config.compile:
        model = torch.compile(model)
    with torch.inference_mode(), autocast():
        for index in range(config.num_samples):
            if index:
                print("\n---\n")
            output = model.generate(tokens, config.max_new_tokens, config.temperature, config.top_k or None)
            print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    sample(load_settings(SampleConfig))
