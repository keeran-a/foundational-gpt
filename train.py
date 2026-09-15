"""Train, resume, or evaluate a GPT checkpoint."""

import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from dataset import TokenDataset, read_metadata
from model import GPT, GPTConfig
from runtime import default_device, load_checkpoint, precision, save_checkpoint
from settings import load_settings


@dataclass
class TrainConfig:
    """Training settings; CLI values override the supplied configuration files."""
    out_dir: str = "out"
    dataset: str = "shakespeare_char"
    data_dir: str = "data"
    init_from: str = "scratch"
    eval_only: bool = False
    always_save_checkpoint: bool = True
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    batch_size: int = 12
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    position_embedding: str = "learned"
    rope_theta: float = 10000.0
    activation: str = "gelu"
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.001
    max_iters: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 100
    lr_decay_iters: int = 2000
    min_lr: float = 0.0001
    device: str = "auto"
    dtype: str = "auto"
    backend: str = "auto"
    init_method: str = "env://"
    compile: bool = False
    seed: int = 1337

    def __post_init__(self):
        for name in ("eval_interval", "eval_iters", "log_interval", "batch_size", "block_size", "gradient_accumulation_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_iters < 0 or self.warmup_iters < 0 or self.grad_clip < 0:
            raise ValueError("Iteration counts and gradient clipping must be nonnegative")
        if self.decay_lr and self.lr_decay_iters <= self.warmup_iters:
            raise ValueError("lr_decay_iters must exceed warmup_iters")
        if not 0 <= self.min_lr <= self.learning_rate or self.learning_rate <= 0:
            raise ValueError("Require 0 <= min_lr <= learning_rate and learning_rate > 0")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1 or self.weight_decay < 0:
            raise ValueError("Invalid AdamW settings")


def learning_rate_at(step, config):
    if not config.decay_lr:
        return config.learning_rate
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / (config.warmup_iters + 1)
    fraction = min(1.0, (step - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters))
    return config.min_lr + (config.learning_rate - config.min_lr) * (1 + math.cos(math.pi * fraction)) / 2


@torch.no_grad()
def evaluate(model, data, iterations, autocast):
    was_training = model.training
    model.eval()
    results = {}
    cuda_devices = [data.device.index if data.device.index is not None else torch.cuda.current_device()] if data.device.type == "cuda" else []
    try:
        # Validation must not change the random stream used for training batches.
        with torch.random.fork_rng(devices=cuda_devices):
            for split in ("train", "val"):
                total = torch.zeros((), device=data.device)
                for _ in range(iterations):
                    inputs, targets = data.batch(split)
                    with autocast():
                        _, loss = model(inputs, targets)
                    total += loss.detach()
                total /= iterations
                if distributed.is_initialized():
                    distributed.all_reduce(total)
                    total /= distributed.get_world_size()
                results[split] = total.item()
    finally:
        model.train(was_training)
    return results


def initialize_model(config, metadata):
    checkpoint = None
    if config.init_from == "scratch":
        model = GPT(GPTConfig(
            n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
            block_size=config.block_size, vocab_size=metadata["vocab_size"],
            bias=config.bias, dropout=config.dropout,
            position_embedding=config.position_embedding, rope_theta=config.rope_theta,
            activation=config.activation,
        ))
    elif config.init_from == "resume":
        checkpoint = load_checkpoint(Path(config.out_dir) / "ckpt.pt")
        if checkpoint.get("tokenizer", metadata) != metadata:
            raise ValueError("Dataset tokenizer differs from the checkpoint tokenizer")
        model = GPT(GPTConfig(**(checkpoint["model_args"] | {"dropout": config.dropout})))
        model.load_state_dict(checkpoint["model"])
    elif config.init_from.startswith("gpt2"):
        if metadata["kind"] != "gpt2":
            raise ValueError("GPT-2 initialization requires GPT-2 tokenized data")
        model = GPT.from_pretrained(config.init_from, {"dropout": config.dropout})
    else:
        raise ValueError("init_from must be scratch, resume, or a GPT-2 model name")
    if model.config.vocab_size < metadata["vocab_size"]:
        raise ValueError("Dataset vocabulary exceeds the model vocabulary")
    if config.block_size > model.config.block_size:
        raise ValueError("Requested context exceeds the checkpoint context")
    if config.block_size < model.config.block_size:
        model.crop_block_size(config.block_size)
    return model, checkpoint


def train(config):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    config.device = default_device() if config.device == "auto" else config.device
    if world_size > 1 and torch.device(config.device).type == "cuda":
        config.device = f"cuda:{local_rank}"
    device = torch.device(config.device)
    if device.type == "cuda":
        if device.index is None:
            config.device = f"cuda:{torch.cuda.current_device()}"
            device = torch.device(config.device)
        torch.cuda.set_device(device)
    if config.gradient_accumulation_steps % world_size:
        raise ValueError("gradient_accumulation_steps must be divisible by world size")
    accumulation = config.gradient_accumulation_steps // world_size
    if world_size > 1:
        backend = ("nccl" if device.type == "cuda" and os.name != "nt" else "gloo") if config.backend == "auto" else config.backend
        distributed.init_process_group(backend=backend, init_method=config.init_method, rank=rank, world_size=world_size)
    data = None
    try:
        torch.manual_seed(config.seed + rank)
        config.dtype, autocast = precision(device, config.dtype)
        metadata = read_metadata(Path(config.data_dir) / config.dataset)
        data = TokenDataset(Path(config.data_dir) / config.dataset, config.block_size, config.batch_size, device)
        model, checkpoint = initialize_model(config, metadata)
        model.to(device)
        optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device.type)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.dtype == "float16")
        step, best_loss = 0, float("inf")
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            position_state = optimizer.state.get(model.transformer.wpe.weight, {}) if "wpe" in model.transformer else {}
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in position_state:
                    position_state[name] = position_state[name][:config.block_size].contiguous()
            step, best_loss = checkpoint["iter_num"], checkpoint["best_val_loss"]
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
        if config.compile:
            execution_model = torch.compile(model)
        else:
            execution_model = model
        if world_size > 1:
            execution_model = DistributedDataParallel(execution_model, device_ids=[local_rank] if device.type == "cuda" else None)
        if checkpoint is not None and "rng_states" in checkpoint:
            states = checkpoint["rng_states"]
            if len(states) != world_size:
                raise ValueError("Resume with the same number of distributed workers")
            torch.set_rng_state(states[rank]["cpu"])
            if device.type == "cuda" and states[rank]["cuda"] is not None:
                torch.cuda.set_rng_state(states[rank]["cuda"], device)
        checkpoint = None

        def persist(filename):
            state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            states = [None] * world_size
            if world_size > 1:
                distributed.all_gather_object(states, state)
            else:
                states[0] = state
            if rank == 0:
                save_checkpoint(Path(config.out_dir) / filename, {
                    "model": model.state_dict(), "model_args": asdict(model.config),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "iter_num": step, "best_val_loss": best_loss,
                    "config": asdict(config), "tokenizer": metadata, "rng_states": states,
                })

        if rank == 0:
            print(f"parameters: {model.get_num_params():,}; device: {device}; dtype: {config.dtype}")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        while True:
            if step % config.eval_interval == 0 or step >= config.max_iters or config.eval_only:
                losses = evaluate(model, data, config.eval_iters, autocast)
                improved = losses["val"] < best_loss
                best_loss = min(best_loss, losses["val"])
                if rank == 0:
                    print(f"step {step}: train {losses['train']:.4f}, validation {losses['val']:.4f}", flush=True)
                if config.eval_only:
                    break
                if improved:
                    persist("best.pt")
                if config.always_save_checkpoint or improved or step >= config.max_iters:
                    persist("ckpt.pt")
            if step >= config.max_iters:
                break
            rate = learning_rate_at(step, config)
            for group in optimizer.param_groups:
                group["lr"] = rate
            started = time.perf_counter()
            total_loss = 0.0
            for micro_step in range(accumulation):
                sync = execution_model.no_sync() if world_size > 1 and micro_step < accumulation - 1 else nullcontext()
                inputs, targets = data.batch("train")
                with sync:
                    with autocast():
                        _, loss = execution_model(inputs, targets)
                        loss = loss / accumulation
                    total_loss += loss.detach().item()
                    scaler.scale(loss).backward()
            if config.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and step % config.log_interval == 0:
                print(f"step {step}: loss {total_loss:.4f}, lr {rate:.6g}, time {time.perf_counter() - started:.3f}s", flush=True)
    finally:
        if data is not None:
            data.close()
        if distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    train(load_settings(TrainConfig))
