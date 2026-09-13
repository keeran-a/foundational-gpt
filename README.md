# Small GPT

A compact PyTorch GPT for training and generating text. Requires Python 3.10–3.13.

```sh
pip install -r requirements.txt
python prepare.py
python train.py
python sample.py
```

Use `python prepare.py --input corpus.txt` for your own text. Resume with `python train.py --init_from=resume`. Settings use `--name=value`; list them with `--help`.

CUDA is selected when available. Checkpoints are saved in `out/`; use `--checkpoint=best.pt` when sampling for the best validation checkpoint. GPT-2 tokenization and weights require `requirements-optional.txt`.
