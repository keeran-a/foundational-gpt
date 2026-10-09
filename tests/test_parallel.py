import contextlib
import io
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.multiprocessing as multiprocessing

import train as training_module

from model import GPT, GPTConfig
from parallel import validate_fsdp
from prepare import prepare_text
from runtime import load_checkpoint
from sample import SampleConfig, sample
from train import TrainConfig, train


def distributed_worker(rank, root):
    from torch.distributed.tensor import DTensor

    root = Path(root)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    base = TrainConfig(
        data_dir=str(root), dataset="text", device="cpu", backend="gloo",
        n_layer=2, n_head=2, n_embd=16, block_size=8, batch_size=3,
        batch_size_start=1, batch_size_ramp_iters=3, gradient_accumulation_steps=4,
        dropout=0.2, bias=True, max_iters=4, eval_interval=2, eval_iters=2,
        log_interval=1, warmup_iters=0, lr_decay_iters=4,
    )
    run_number = 0
    checked_optimizer_shards = False
    checkpoint_states = training_module.checkpoint_states

    def inspect_shards(model, optimizer, sharded, include_optimizer):
        nonlocal checked_optimizer_shards
        if sharded:
            parameters = list(model.parameters())
            assert all(isinstance(parameter, DTensor) for parameter in parameters)
            assert sum(parameter.to_local().numel() for parameter in parameters) * 2 == sum(parameter.numel() for parameter in parameters)
            assert model.lm_head.weight is model.transformer.wte.weight
            for parameter, state in optimizer.state.items():
                for name in ("exp_avg", "exp_avg_sq"):
                    assert isinstance(state[name], DTensor)
                    assert state[name].to_local().numel() * 2 == parameter.numel()
                checked_optimizer_shards = True
        weights, state = checkpoint_states(model, optimizer, sharded, include_optimizer)
        if sharded and rank != 0:
            assert not weights
            assert not state
        return weights, state

    def run(directory, method, **changes):
        nonlocal run_number
        store = (root / f"rendezvous-{run_number}").as_uri()
        run_number += 1
        with contextlib.redirect_stdout(io.StringIO()) as output, patch.object(training_module, "checkpoint_states", side_effect=inspect_shards):
            train(replace(base, out_dir=str(root / directory), parallelism=method,
                          init_method=store, **changes))
        if rank == 0 and not changes.get("eval_only") and changes.get("max_iters", 4) > 0:
            assert "grad norm" in output.getvalue()

    run("ddp", "ddp")
    run("fsdp", "fsdp")
    run("resumed", "fsdp", max_iters=2)
    run("resumed", "fsdp", init_from="resume")
    run("from-ddp", "ddp", max_iters=2)
    run("from-ddp", "fsdp", init_from="resume")
    run("to-ddp", "fsdp", max_iters=2)
    run("to-ddp", "ddp", init_from="resume")
    run("rotary", "fsdp", position_embedding="rotary")
    run("rotary-resumed", "fsdp", position_embedding="rotary", max_iters=2)
    run("rotary-resumed", "fsdp", init_from="resume")
    run("zero-step", "fsdp", max_iters=0)
    assert not load_checkpoint(root / "zero-step" / "ckpt.pt", training=True)["optimizer"]["state"]
    run("zero-step", "fsdp", init_from="resume")
    run("cropped", "fsdp", max_iters=2)
    run("cropped", "fsdp", init_from="resume", block_size=4)
    run("fsdp", "fsdp", init_from="resume", eval_only=True)
    assert checked_optimizer_shards


@unittest.skipIf(tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 14),
                 "CPU FSDP2 requires PyTorch 2.14+")
class ParallelTests(unittest.TestCase):
    def test_two_worker_training_resume_and_backend_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_text("abc def\n" * 200, root / "text")
            multiprocessing.spawn(distributed_worker, args=(str(root),), nprocs=2, join=True)
            fsdp = load_checkpoint(root / "fsdp" / "ckpt.pt", training=True)
            self.assertEqual(fsdp["iter_num"], 4)
            self.assertEqual(fsdp["optimizer_format"], "named")
            self.assertEqual(len(fsdp["rng_states"]), 2)
            for directory in ("ddp", "resumed", "from-ddp", "to-ddp", "zero-step"):
                checkpoint = load_checkpoint(root / directory / "ckpt.pt")
                for name, weights in fsdp["model"].items():
                    torch.testing.assert_close(weights, checkpoint["model"][name], rtol=1e-4, atol=1e-6)
                if directory in {"resumed", "zero-step"}:
                    self.assertEqual(fsdp["best_val_loss"], checkpoint["best_val_loss"])
            rotary = load_checkpoint(root / "rotary" / "ckpt.pt")
            resumed = load_checkpoint(root / "rotary-resumed" / "ckpt.pt")
            for name, weights in rotary["model"].items():
                torch.testing.assert_close(weights, resumed["model"][name], rtol=0, atol=0)
            cropped = load_checkpoint(root / "cropped" / "ckpt.pt", training=True)
            self.assertEqual(cropped["model"]["transformer.wpe.weight"].shape[0], 4)
            self.assertEqual(cropped["optimizer"]["state"]["transformer.wpe.weight"]["exp_avg"].shape[0], 4)
            model = GPT(GPTConfig(**fsdp["model_args"]))
            model.load_state_dict(fsdp["model"])
            self.assertIs(model.lm_head.weight, model.transformer.wte.weight)
            self.assertTrue(all(type(value) is torch.Tensor for value in fsdp["model"].values()))
            (root / "fsdp" / fsdp["training_state_file"]).unlink()
            with contextlib.redirect_stdout(io.StringIO()) as output:
                sample(SampleConfig(out_dir=str(root / "fsdp"), device="cpu", start="ab",
                                    max_new_tokens=4, top_k=1))
            self.assertEqual(len(output.getvalue()), 7)


class ParallelConfigTests(unittest.TestCase):
    @patch("parallel.torch.__version__", "2.14.0")
    def test_invalid_parallel_settings_fail_before_initialization(self):
        with self.assertRaisesRegex(ValueError, "parallelism"):
            TrainConfig(parallelism="invalid")
        for device, workers, dtype, compile_model, message in (
            ("cpu", 1, "float32", False, "two workers"),
            ("mps", 2, "float32", False, "CPU and CUDA"),
            ("cuda", 2, "float16", False, "float32 or CUDA bfloat16"),
            ("cpu", 2, "float32", True, "compile=False"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                validate_fsdp(torch.device(device), workers, dtype, compile_model)

    @patch("parallel.torch.__version__", "2.13.0")
    def test_old_torch_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "PyTorch 2.14"):
            validate_fsdp(torch.device("cpu"), 2, "float32", False)


if __name__ == "__main__":
    unittest.main()
