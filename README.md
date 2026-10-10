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
Keep the matching `ckpt.train-*.pt` file with `ckpt.pt` to resume; sampling needs only the model checkpoint.

DDP is the default distributed backend. Optional FSDP2 requires PyTorch 2.14+, two or more workers, float32 or CUDA bfloat16, and `--compile=False`:
```sh
torchrun --standalone --nproc_per_node=2 train.py --parallelism=fsdp --gradient_accumulation_steps=2
```
Add `--device=cpu` for CPU training. Resume with the same worker count; checkpoints work with either backend. Checkpoint export gathers full weights and optimizer buffers into rank-zero CPU memory.

Windows PowerShell CPU launch:
```powershell
$env:TORCH_DISABLE_SHARE_RDZV_TCP_STORE = '1'
$store = Join-Path $env:TEMP ('gpt-rdzv-' + [guid]::NewGuid())
torchrun --rdzv_backend=c10d "--rdzv_endpoint=$store" --rdzv_conf=store_type=file --nproc_per_node=2 train.py --parallelism=fsdp --device=cpu --gradient_accumulation_steps=2
```
