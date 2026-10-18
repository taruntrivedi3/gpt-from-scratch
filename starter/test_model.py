"""Behavioral checks: causal masking, target alignment, and ability to learn."""

import unittest

import torch

from .data import Tokenizer, get_batch
from .model import GPT, GPTConfig


class GPTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_future_tokens_do_not_change_prefix_predictions(self):
        model = GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=32)).eval()
        first = torch.tensor([[1, 2, 3, 4, 5, 6]])
        second = torch.tensor([[1, 2, 3, 9, 8, 7]])
        with torch.no_grad():
            a, _ = model(first)
            b, _ = model(second)
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=1e-6)
        self.assertGreater((a[:, 3:] - b[:, 3:]).abs().max().item(), 1e-4)

    def test_targets_are_next_tokens_even_at_last_possible_start(self):
        # There is exactly one valid window, so its expected contents are known.
        tokens = torch.tensor([3, 8, 2, 9, 1], dtype=torch.int32)
        x, y = get_batch(tokens, 2, 4, torch.Generator().manual_seed(1))
        self.assertEqual(x.tolist(), [[3, 8, 2, 9], [3, 8, 2, 9]])
        self.assertEqual(y.tolist(), [[8, 2, 9, 1], [8, 2, 9, 1]])
        with self.assertRaises(ValueError):
            get_batch(tokens[:4], 1, 4, torch.Generator())

    def test_tiny_model_can_overfit_a_repeating_pattern(self):
        model = GPT(GPTConfig(vocab_size=4, block_size=8, n_layer=1, n_head=2, n_embd=32))
        optimizer = model.configure_optimizer(0.01)
        x = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        y = torch.tensor([[1, 2, 3, 0, 1, 2, 3, 0]])
        initial = model(x, y)[1].item()
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
        final = model(x, y)[1].item()
        self.assertLess(final, initial * 0.1)
        self.assertEqual(model(x)[0].argmax(-1).tolist(), y.tolist())

    def test_validation_characters_do_not_expand_the_training_vocabulary(self):
        tokenizer = Tokenizer({"kind": "char", "characters": "abc"})
        self.assertEqual(tokenizer.encode("abz"), [0, 1, 3])
        self.assertEqual(tokenizer.decode([0, 1, 3]), "ab�")

    def test_invalid_head_width_and_context_fail_clearly(self):
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=4, n_head=3, n_embd=32)
        model = GPT(GPTConfig(vocab_size=4, block_size=4, n_layer=1, n_head=2, n_embd=16))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 5), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
