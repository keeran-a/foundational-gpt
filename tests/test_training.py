import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dataset import TokenDataset, Tokenizer, read_metadata
from prepare import prepare_text
from runtime import load_checkpoint
from sample import SampleConfig, sample
from settings import load_settings
from train import TrainConfig, batch_size_at, learning_rate_at, perplexity, train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.metadata = prepare_text("abc def\n" * 200, self.root / "text")
        self.config = TrainConfig(
            data_dir=str(self.root), dataset="text", out_dir=str(self.root / "full"),
            device="cpu", n_layer=1, n_head=2, n_embd=16, block_size=8, batch_size=2,
            max_iters=4, eval_interval=2, eval_iters=2, log_interval=10,
            warmup_iters=0, lr_decay_iters=4, dropout=0.2, gradient_accumulation_steps=2,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def run_training(self, config):
        with contextlib.redirect_stdout(io.StringIO()):
            train(config)
        return load_checkpoint(Path(config.out_dir) / "ckpt.pt")

    def test_resume_matches_uninterrupted_training(self):
        full = self.run_training(self.config)
        partial_config = replace(self.config, out_dir=str(self.root / "resumed"), max_iters=2)
        partial = self.run_training(partial_config)
        self.assertEqual(partial["iter_num"], 2)
        resumed = self.run_training(replace(partial_config, max_iters=4, init_from="resume"))
        self.assertEqual(resumed["iter_num"], 4)
        for name in full["model"]:
            torch.testing.assert_close(full["model"][name], resumed["model"][name], rtol=0, atol=0)
        self.assertEqual(full["best_val_loss"], resumed["best_val_loss"])

    def test_final_checkpoint_and_portable_sampling(self):
        result = self.run_training(replace(self.config, max_iters=1, eval_interval=100, always_save_checkpoint=False))
        self.assertEqual(result["iter_num"], 1)
        self.assertNotIn("optimizer", result)
        (Path(self.config.out_dir) / result["training_state_file"]).unlink()
        (self.root / "text" / "meta.json").unlink()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            sample(SampleConfig(out_dir=self.config.out_dir, start="ab", max_new_tokens=5, device="cpu", top_k=1))
        self.assertEqual(len(output.getvalue()), 8)

    def test_rotary_resume_preserves_architecture(self):
        config = replace(self.config, position_embedding="rotary", max_iters=2)
        self.run_training(config)
        resumed = self.run_training(replace(config, init_from="resume", max_iters=4, position_embedding="learned", dropout=0.0))
        self.assertEqual(resumed["model_args"]["position_embedding"], "rotary")
        self.assertEqual(resumed["model_args"]["dropout"], config.dropout)
        self.assertEqual(resumed["iter_num"], 4)
        self.assertNotIn("transformer.wpe.weight", resumed["model"])

    def test_batch_ramp_schedule_and_resume(self):
        config = replace(self.config, batch_size=4, batch_size_start=1, batch_size_ramp_iters=3)
        self.assertEqual([batch_size_at(step, config) for step in range(5)], [1, 2, 3, 4, 4])
        full = self.run_training(config)
        partial = replace(config, out_dir=str(self.root / "ramp-resumed"), max_iters=2)
        self.run_training(partial)
        resumed = self.run_training(replace(partial, init_from="resume", max_iters=4))
        for name in full["model"]:
            torch.testing.assert_close(full["model"][name], resumed["model"][name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            replace(config, batch_size_start=5)
        with self.assertRaises(ValueError):
            replace(config, batch_size_ramp_iters=0)

    def test_health_logging_and_perplexity(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            train(replace(self.config, max_iters=1, grad_clip=0.00001, log_interval=1))
        self.assertIn("grad norm ", output.getvalue())
        self.assertIn("clipped updates 1", output.getvalue())
        self.assertIn("val perplexity ", output.getvalue())
        self.assertEqual(perplexity(0), 1)
        self.assertEqual(perplexity(1000), float("inf"))

    def test_finetuning_has_warmup_and_decay(self):
        path = Path(__file__).resolve().parents[1] / "config" / "finetune_shakespeare.py"
        config = load_settings(TrainConfig, [str(path)])
        self.assertLess(learning_rate_at(0, config), learning_rate_at(config.warmup_iters, config))
        self.assertEqual(learning_rate_at(config.max_iters, config), config.min_lr)

    def test_batches_and_tokenizer(self):
        tokenizer = Tokenizer(read_metadata(self.root / "text"))
        self.assertEqual(tokenizer.decode(tokenizer.encode("abc\n")), "abc\n")
        with self.assertRaises(ValueError):
            tokenizer.encode("z")
        data = TokenDataset(self.root / "text", 8, 2, "cpu")
        inputs, targets = data.batch("train")
        self.assertEqual(inputs.shape, (2, 8))
        torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])
        data.close()
        np.array([0], dtype=np.uint16).tofile(self.root / "text" / "val.bin")
        with self.assertRaises(ValueError):
            TokenDataset(self.root / "text", 8, 2, "cpu")

    def test_settings_and_schedule(self):
        path = self.root / "settings.py"
        path.write_text('batch_size = 3\ncompile = False\n', encoding="utf-8")
        config = load_settings(TrainConfig, [str(path), "--batch_size=5", "--learning_rate=1"])
        self.assertEqual(config.batch_size, 5)
        self.assertEqual(config.learning_rate, 1.0)
        path.write_text('raise RuntimeError("executed")\n', encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            load_settings(TrainConfig, [str(path)])
        self.assertEqual(learning_rate_at(0, self.config), self.config.learning_rate)
        self.assertEqual(learning_rate_at(4, self.config), self.config.min_lr)
        self.assertEqual(learning_rate_at(9, self.config), self.config.min_lr)

    def test_resume_rejects_different_tokenizer(self):
        self.run_training(replace(self.config, max_iters=1))
        metadata = dict(self.metadata)
        metadata["chars"] = list(reversed(metadata["chars"]))
        (self.root / "text" / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tokenizer differs"):
            self.run_training(replace(self.config, init_from="resume"))


if __name__ == "__main__":
    unittest.main()
