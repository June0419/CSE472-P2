"""Unit tests for prompt formatting and answer masking."""

from pathlib import Path
import unittest

from cse472_project2.prompt import (
    build_prompt,
    create_training_sequence,
    format_choices,
    resolve_candidate_choice_token_ids,
)


class MockTokenizer:
    """Lightweight mock tokenizer for testing without Hugging Face downloads."""

    def __init__(self):
        self.bos_token_id = 1
        self.pad_token_id = 0
        self.vocab = {"<bos>": 1, "<pad>": 0, "A": 10, "B": 11, "C": 12, "D": 13, " 1": 21, " 2": 22}

    def encode(self, text, add_special_tokens=False):
        tokens = []
        for word in text.strip().split():
            tokens.append(self.vocab.get(word, hash(word) % 1000 + 100))
        return tokens


class PromptTests(unittest.TestCase):
    def test_format_choices(self):
        labels = ["A", "B", "C"]
        texts = ["apple", "banana", "cherry"]
        formatted = format_choices(labels, texts)
        self.assertEqual(formatted, "A. apple\nB. banana\nC. cherry")

    def test_build_prompt(self):
        choices = {"label": ["A", "B"], "text": ["yes", "no"]}
        prompt = build_prompt("Is the sky blue?", choices)
        self.assertTrue(prompt.startswith("Choose the correct answer."))
        self.assertIn("Question: Is the sky blue?", prompt)
        self.assertIn("A. yes\nB. no", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_answer_only_loss_masking(self):
        tokenizer = MockTokenizer()
        prompt = "Question: What is 2+2?\nA. 3\nB. 4\nAnswer:"
        target = "B"

        seq = create_training_sequence(prompt, target, tokenizer, max_seq_length=128)
        self.assertIn("input_ids", seq)
        self.assertIn("labels", seq)
        self.assertIn("attention_mask", seq)

        input_ids = seq["input_ids"]
        labels = seq["labels"]

        # First token should be BOS (id 1)
        self.assertEqual(input_ids[0], 1)
        # All prompt tokens should be masked (-100)
        num_prompt_tokens = len(labels) - 1  # 1 target token
        self.assertTrue(all(label == -100 for label in labels[:num_prompt_tokens]))
        # Last token is target, should NOT be -100
        self.assertNotEqual(labels[-1], -100)
        self.assertEqual(labels[-1], input_ids[-1])

    def test_resolve_choice_tokens(self):
        tokenizer = MockTokenizer()
        mapping = resolve_candidate_choice_token_ids(tokenizer, ["A", "B"])
        self.assertIn("A", mapping)
        self.assertIn("B", mapping)


if __name__ == "__main__":
    unittest.main()
