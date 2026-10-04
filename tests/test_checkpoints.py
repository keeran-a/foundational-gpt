from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from runtime import load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "ckpt.pt"
        self.payload = {
            "model": {"_orig_mod.weight": torch.ones(2)}, "model_args": {}, "iter_num": 3,
            "optimizer": {"buffer": torch.zeros(100)}, "scaler": {},
            "rng_states": [{"cpu": torch.get_rng_state(), "cuda": None}],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_model_and_training_state_are_separate(self):
        save_checkpoint(self.path, self.payload)
        model = torch.load(self.path, weights_only=True)
        self.assertNotIn("optimizer", model)
        self.assertNotIn("rng_states", model)
        state = self.root / model["training_state_file"]
        self.assertTrue(state.exists())
        with patch("runtime.torch.load", wraps=torch.load) as load:
            inference = load_checkpoint(self.path)
            self.assertEqual(load.call_count, 1)
        self.assertIn("weight", inference["model"])
        training = load_checkpoint(self.path, training=True)
        torch.testing.assert_close(training["optimizer"]["buffer"], self.payload["optimizer"]["buffer"])
        self.assertEqual(training["iter_num"], 3)
        state.unlink()
        self.assertIn("model", load_checkpoint(self.path))
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(self.path, training=True)

    def test_publish_failure_preserves_previous_checkpoint(self):
        save_checkpoint(self.path, self.payload)
        old = torch.load(self.path, weights_only=True)
        import runtime
        atomic_save = runtime._atomic_save
        def fail_weights(path, checkpoint):
            if path == self.path:
                raise OSError("write failed")
            atomic_save(path, checkpoint)
        with patch("runtime._atomic_save", side_effect=fail_weights), self.assertRaises(OSError):
            save_checkpoint(self.path, self.payload | {"iter_num": 4})
        self.assertEqual(load_checkpoint(self.path, training=True)["iter_num"], 3)
        self.assertTrue((self.root / old["training_state_file"]).exists())
        save_checkpoint(self.path, self.payload | {"iter_num": 4})
        self.assertEqual(load_checkpoint(self.path, training=True)["iter_num"], 4)
        self.assertEqual(len(list(self.root.glob("ckpt.train-*.pt"))), 1)

    def test_reject_mismatched_training_state(self):
        save_checkpoint(self.path, self.payload)
        model = torch.load(self.path, weights_only=True)
        state_path = self.root / model["training_state_file"]
        state = torch.load(state_path, weights_only=True)
        state["iter_num"] = 9
        torch.save(state, state_path)
        with self.assertRaisesRegex(ValueError, "different checkpoints"):
            load_checkpoint(self.path, training=True)
        model["training_state_file"] = "../other.pt"
        torch.save(model, self.path)
        with self.assertRaisesRegex(ValueError, "filename"):
            load_checkpoint(self.path, training=True)

    def test_legacy_combined_checkpoint_can_resume(self):
        torch.save(self.payload, self.path)
        training = load_checkpoint(self.path, training=True)
        self.assertIn("optimizer", training)
        self.assertIn("weight", training["model"])
        self.assertNotIn("optimizer", load_checkpoint(self.path))


if __name__ == "__main__":
    unittest.main()
