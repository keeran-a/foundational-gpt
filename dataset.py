"""Memory-mapped token batches and tokenizer metadata."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch


def read_metadata(directory):
    directory = Path(directory)
    if (directory / "meta.json").exists():
        return json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if (directory / "meta.pkl").exists():
        with (directory / "meta.pkl").open("rb") as stream:
            old = pickle.load(stream)
        if "itos" in old:
            return {"kind": "character", "chars": [old["itos"][i] for i in range(old["vocab_size"])], "vocab_size": old["vocab_size"]}
        return {"kind": "gpt2", "vocab_size": old["vocab_size"]}
    return {"kind": "gpt2", "vocab_size": 50257}


class Tokenizer:
    def __init__(self, metadata):
        self.metadata = metadata
        self.vocab_size = metadata["vocab_size"]
        if metadata["kind"] == "character":
            self.chars = metadata["chars"]
            self.indices = {character: index for index, character in enumerate(self.chars)}
        elif metadata["kind"] == "gpt2":
            import tiktoken
            self.encoding = tiktoken.get_encoding("gpt2")
        else:
            raise ValueError("Unsupported tokenizer type")

    def encode(self, text):
        if self.metadata["kind"] == "gpt2":
            return self.encoding.encode(text, allowed_special={"<|endoftext|>"})
        try:
            return [self.indices[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Prompt contains a character outside the vocabulary: {error.args[0]!r}") from error

    def decode(self, tokens):
        if self.metadata["kind"] == "gpt2":
            return self.encoding.decode(tokens)
        return "".join(self.chars[index] for index in tokens)


class TokenDataset:
    def __init__(self, directory, block_size, batch_size, device):
        self.directory = Path(directory)
        self.block_size, self.batch_size = block_size, batch_size
        self.device = torch.device(device)
        self.splits = {}
        try:
            for split in ("train", "val"):
                filename = self.directory / f"{split}.bin"
                if not filename.exists() or filename.stat().st_size % 2:
                    raise ValueError(f"Missing or invalid uint16 token file: {filename}")
                if filename.stat().st_size // 2 <= block_size:
                    raise ValueError(f"{filename} needs at least {block_size + 1} tokens")
                self.splits[split] = np.memmap(filename, dtype=np.uint16, mode="r")
        except Exception:
            self.close()
            raise

    def close(self):
        for mapping in self.splits.values():
            mapping._mmap.close()
        self.splits.clear()

    def batch(self, split):
        tokens = self.splits[split]
        starts = torch.randint(len(tokens) - self.block_size, (self.batch_size,)).numpy()
        offsets = starts[:, None] + np.arange(self.block_size + 1)
        window = torch.from_numpy(np.asarray(tokens[offsets], dtype=np.int64))
        if self.device.type == "cuda":
            window = window.pin_memory()
        window = window.to(self.device, non_blocking=self.device.type == "cuda")
        return window[:, :-1].contiguous(), window[:, 1:].contiguous()
