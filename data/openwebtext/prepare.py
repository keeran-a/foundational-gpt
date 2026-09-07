"""Stream OpenWebText into uint16 GPT-2 token files."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset", default="Skylion007/openwebtext")
    parser.add_argument("--validation-fraction", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--max-documents", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or args.max_documents < 0:
        parser.error("Use a validation fraction in (0, 1) and a nonnegative document limit")
    encoding = tiktoken.get_encoding("gpt2")
    generator = random.Random(args.seed)
    corpus = load_dataset(args.dataset, split="train", streaming=True)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    temporary = {split: args.output / f"{split}.bin.tmp" for split in counts}
    with temporary["train"].open("wb") as train, temporary["val"].open("wb") as validation:
        streams = {"train": train, "val": validation}
        for index, document in enumerate(corpus):
            if args.max_documents and index >= args.max_documents:
                break
            # Seed both splits before assigning the remaining documents randomly.
            split = "train" if index == 0 else "val" if index == 1 or generator.random() < args.validation_fraction else "train"
            tokens = encoding.encode_ordinary(document["text"]) + [encoding.eot_token]
            np.asarray(tokens, dtype=np.uint16).tofile(streams[split])
            counts[split] += len(tokens)
            if (index + 1) % 10000 == 0:
                print(f"documents: {index + 1:,}; tokens: {sum(counts.values()):,}", flush=True)
    if any(count < 2 for count in counts.values()):
        raise ValueError("Not enough documents to populate both token splits")
    for split, filename in temporary.items():
        filename.replace(args.output / f"{split}.bin")
    metadata = {"kind": "gpt2", "vocab_size": encoding.n_vocab}
    (args.output / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
