import importlib.util
import unittest

import torch

from model import GPT, GPTConfig, apply_rotary


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = GPTConfig(block_size=8, vocab_size=19, n_layer=2, n_head=2, n_embd=16)
        self.model = GPT(self.config).eval()

    def test_attention_is_causal(self):
        tokens = torch.randint(19, (2, 8))
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 19
        logits, _ = self.model(tokens, tokens)
        other, _ = self.model(changed, changed)
        torch.testing.assert_close(logits[:, :4], other[:, :4], rtol=0, atol=0)

    def test_loss_gradient_and_tied_optimizer_weights(self):
        tokens = torch.randint(19, (2, 8))
        _, loss = self.model(tokens, tokens)
        loss.backward()
        optimizer = self.model.configure_optimizers(0.1, 0.001, (0.9, 0.99), "cpu")
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        self.assertIs(self.model.lm_head.weight, self.model.transformer.wte.weight)
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in parameters))
        before = self.model.lm_head.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.lm_head.weight))

    def test_inference_and_context_cropping(self):
        tokens = torch.randint(19, (2, 8))
        full, _ = self.model(tokens, tokens)
        last, loss = self.model(tokens)
        self.assertIsNone(loss)
        torch.testing.assert_close(last, full[:, -1:])
        self.model.crop_block_size(4)
        generated = self.model.generate(tokens, 5, top_k=1)
        self.assertEqual(generated.shape, (2, 13))
        self.assertEqual(self.model.transformer.wpe.weight.shape, (4, 16))

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            GPTConfig(n_head=3, n_embd=16)
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 9), dtype=torch.long))
        for options in ({"temperature": 0}, {"top_k": 0}, {"max_new_tokens": -1}):
            with self.assertRaises(ValueError):
                self.model.generate(torch.ones((1, 1), dtype=torch.long), **({"max_new_tokens": 1} | options))

    def test_rotary_forward_backward_and_crop(self):
        model = GPT(GPTConfig(block_size=8, vocab_size=19, n_layer=2, n_head=2, n_embd=16, position_embedding="rotary")).eval()
        self.assertNotIn("wpe", model.transformer)
        self.assertFalse(any("rope_" in name for name in model.state_dict()))
        self.assertEqual(model.get_num_params(), model.get_num_params(non_embedding=False))
        tokens = torch.randint(19, (2, 8))
        logits, loss = model(tokens, tokens)
        loss.backward()
        self.assertEqual(logits.shape, (2, 8, 19))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 19
        other, _ = model(changed, changed)
        torch.testing.assert_close(logits[:, :4], other[:, :4], rtol=0, atol=0)
        model.crop_block_size(4)
        self.assertEqual(model.transformer.h[0].attn.rope_cosine.shape[0], 4)
        self.assertEqual(model.generate(tokens, 3, top_k=1).shape, (2, 11))

    def test_rotary_relative_position_and_norm(self):
        query, key = torch.randn(3, 8), torch.randn(3, 8)
        frequency = 10000.0 ** (-torch.arange(0, 8, 2) / 8)
        angles = torch.arange(3)[:, None] * frequency
        def scores(offset):
            phases = angles + offset * frequency
            rotated_query = apply_rotary(query, phases.cos(), phases.sin())
            rotated_key = apply_rotary(key, phases.cos(), phases.sin())
            torch.testing.assert_close(rotated_query.norm(dim=-1), query.norm(dim=-1))
            return rotated_query @ rotated_key.t()
        torch.testing.assert_close(scores(0), scores(5), rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            GPTConfig(n_head=2, n_embd=6, position_embedding="rotary")

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "optional transformers dependency")
    def test_gpt2_parameter_import(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        reference = GPT2LMHeadModel(GPT2Config(
            vocab_size=19, n_positions=8, n_layer=2, n_head=2, n_embd=16,
            attn_pdrop=0, resid_pdrop=0, embd_pdrop=0, activation_function="gelu",
        )).eval()
        GPT.import_weights(self.model.state_dict(), reference.state_dict())
        tokens = torch.randint(19, (2, 8))
        logits, _ = self.model(tokens, tokens)
        torch.testing.assert_close(logits, reference(tokens).logits, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "optional transformers dependency")
    def test_gpt2_tanh_activation_matches_default(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        reference = GPT2LMHeadModel(GPT2Config(
            vocab_size=19, n_positions=8, n_layer=2, n_head=2, n_embd=16,
            attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
        )).eval()
        model = GPT(GPTConfig(block_size=8, vocab_size=19, n_layer=2, n_head=2, n_embd=16, activation="gelu_tanh")).eval()
        GPT.import_weights(model.state_dict(), reference.state_dict())
        tokens = torch.randint(19, (2, 8))
        torch.testing.assert_close(model(tokens, tokens)[0], reference(tokens).logits, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
