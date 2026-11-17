"""Unit tests for evaluation metrics, aggregation, and table generation."""

from pathlib import Path
import unittest

from cse472_project2.evaluate import (
    aggregate_results,
    format_markdown_table,
    simulate_mock_metrics,
    slugify_model_name,
)


class EvaluateTests(unittest.TestCase):
    def test_slugify_model_name(self):
        slug = slugify_model_name("meta-llama/Meta-Llama-3-8B")
        self.assertEqual(slug, "meta_llama_3_8b")
        slug_q = slugify_model_name("Qwen/Qwen3-8B")
        self.assertEqual(slug_q, "qwen3_8b")

    def test_simulate_mock_metrics(self):
        metrics = simulate_mock_metrics("meta-llama/Meta-Llama-3-8B", "protected", seed=42)
        self.assertIn("clean_test_accuracy", metrics)
        self.assertIn("gpt2_perplexity", metrics)
        self.assertIn("sbert_similarity", metrics)
        self.assertTrue(0 <= metrics["clean_test_accuracy"] <= 100)
        self.assertTrue(metrics["gpt2_perplexity"] > 0)
        self.assertTrue(0.5 <= metrics["sbert_similarity"] <= 1.0)

    def test_aggregate_results(self):
        runs = [
            {
                "model": "modelA",
                "condition": "original",
                "seed": 42,
                "clean_test_accuracy": 50.0,
                "gpt2_perplexity": 20.0,
                "sbert_similarity": 1.0,
            },
            {
                "model": "modelA",
                "condition": "original",
                "seed": 123,
                "clean_test_accuracy": 52.0,
                "gpt2_perplexity": 20.0,
                "sbert_similarity": 1.0,
            },
            {
                "model": "modelA",
                "condition": "original",
                "seed": 2026,
                "clean_test_accuracy": 54.0,
                "gpt2_perplexity": 20.0,
                "sbert_similarity": 1.0,
            },
        ]
        agg = aggregate_results(runs)
        self.assertIn("modelA", agg)
        self.assertIn("original", agg["modelA"])
        stats = agg["modelA"]["original"]
        self.assertEqual(stats["accuracy_mean"], 52.0)
        self.assertAlmostEqual(stats["accuracy_std"], 1.63, places=2)
        self.assertEqual(stats["perplexity_mean"], 20.0)
        self.assertEqual(stats["perplexity_std"], 0.0)
        self.assertEqual(stats["similarity_mean"], 1.0)
        self.assertEqual(stats["similarity_std"], 0.0)

    def test_format_markdown_table(self):
        agg = {
            "meta-llama/Meta-Llama-3-8B": {
                "original": {
                    "accuracy_mean": 48.5,
                    "accuracy_std": 0.5,
                    "perplexity_mean": 22.0,
                    "perplexity_std": 0.1,
                    "similarity_mean": 1.0,
                    "similarity_std": 0.0,
                }
            }
        }
        table = format_markdown_table(agg)
        self.assertIn("| VICTIM | TRAINING DATA | TEST ACCURACY |", table)
        self.assertIn("Llama-3-8B", table)
        self.assertIn("48.50 ± 0.50", table)



if __name__ == "__main__":
    unittest.main()
