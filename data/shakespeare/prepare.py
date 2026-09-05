"""Prepare the default corpus with GPT-2 tokenization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prepare import main

if __name__ == "__main__":
    main(default_output=Path(__file__).resolve().parent, default_tokenizer="gpt2")
