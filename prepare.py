"""Convert local text or the default corpus into training and validation tokens."""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare_text(text, output, tokenizer="character", train_fraction=0.9):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    boundary = int(len(text) * train_fraction)
    if not 0 < boundary < len(text):
        raise ValueError("Text must contain nonempty training and validation splits")
    if tokenizer == "character":
        characters = sorted(set(text))
        if len(characters) > np.iinfo(np.uint16).max + 1:
            raise ValueError("Character vocabulary exceeds the uint16 token format")
        mapping = {character: index for index, character in enumerate(characters)}
        encode = lambda content: [mapping[character] for character in content]
        metadata = {"kind": "character", "chars": characters, "vocab_size": len(characters)}
    elif tokenizer == "gpt2":
        import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
        encode = encoding.encode_ordinary
        metadata = {"kind": "gpt2", "vocab_size": encoding.n_vocab}
    else:
        raise ValueError("tokenizer must be character or gpt2")
    splits = {"train": encode(text[:boundary]), "val": encode(text[boundary:])}
    if any(len(tokens) < 2 for tokens in splits.values()):
        raise ValueError("Each split must contain at least two tokens")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, tokens in splits.items():
        np.asarray(tokens, dtype=np.uint16).tofile(output / f"{name}.bin")
        print(f"{name}: {len(tokens):,} tokens")
    (output / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    print(f"vocabulary: {metadata['vocab_size']:,}")
    return metadata


def main(default_output="data/shakespeare_char", default_tokenizer="character"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local UTF-8 text; otherwise download the default corpus")
    parser.add_argument("--output", type=Path, default=Path(default_output))
    parser.add_argument("--tokenizer", choices=("character", "gpt2"), default=default_tokenizer)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    args = parser.parse_args()
    if args.input is not None:
        text = args.input.read_text(encoding="utf-8")
    else:
        source = args.output / "input.txt"
        if source.exists():
            text = source.read_text(encoding="utf-8")
        else:
            with urllib.request.urlopen(CORPUS_URL, timeout=60) as response:
                text = response.read().decode("utf-8")
            args.output.mkdir(parents=True, exist_ok=True)
            source.write_text(text, encoding="utf-8")
    prepare_text(text, args.output, args.tokenizer, args.train_fraction)


if __name__ == "__main__":
    main()
