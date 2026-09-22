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

Train with rotary positions using `python train.py config/train_rotary.py`.
Optional batch ramp: `--batch_size_start=4 --batch_size_ramp_iters=500`. Training reports validation perplexity, gradient norms, and clipping counts.
