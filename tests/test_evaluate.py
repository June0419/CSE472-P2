"""Unit tests for evaluation metrics, aggregation, and table generation."""

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from cse472_project2 import evaluate
from cse472_project2.evaluate import (
    aggregate_results,
    format_markdown_table,
    resolve_single_token_choice_ids,
    run_evaluation_suite,
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
        self.assertEqual(
            metrics,
            simulate_mock_metrics("meta-llama/Meta-Llama-3-8B", "protected", seed=42),
        )

    def test_choice_labels_must_have_distinct_single_tokens(self):
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: {
            " A": [10], " B": [11], " C": [12, 13]
        }[text])
        self.assertEqual(resolve_single_token_choice_ids(tokenizer, ["A", "B"]), {"A": 10, "B": 11})
        with self.assertRaisesRegex(ValueError, "Sequence scoring is required"):
            resolve_single_token_choice_ids(tokenizer, ["A", "C"])

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

    def test_evaluate_selected_single_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.toml"
            config.write_text(
                'seeds = [42, 123]\nconditions = ["zero_shot", "original"]\n'
                '[models]\nwhite_box = "owner/Model-A"\nblack_box = "Qwen/Qwen3-8B"\n',
                encoding="utf-8",
            )
            result = run_evaluation_suite(
                config,
                output_path=root / "results.json",
                mock=True,
                model_names=["Qwen/Qwen3-8B"],
                selected_conditions=["original"],
                selected_seeds=[42],
            )
            self.assertEqual(len(result["runs"]), 1)
            self.assertEqual(result["runs"][0]["model"], "Qwen/Qwen3-8B")
            self.assertEqual(result["runs"][0]["condition"], "original")
            self.assertEqual(result["runs"][0]["seed"], 42)

    def test_real_zero_shot_and_adapter_paths(self):
        """Exercise real orchestration without downloading large model weights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "data/original"
            (original / "prepared").mkdir(parents=True)
            (original / "raw").mkdir()
            (original / "manifest.json").write_text(json.dumps({
                "dataset_id": "allenai/ai2_arc", "subset": "ARC-Challenge",
                "resolved_revision": "official-commit"
            }), encoding="utf-8")
            (original / "prepared/test.jsonl").write_text(json.dumps({
                "prompt": "Question? Answer:", "target": "A", "choices": {"label": ["A", "B"]}
            }) + "\n", encoding="utf-8")
            (original / "raw/train.jsonl").write_text(json.dumps({"question": "Question?"}) + "\n", encoding="utf-8")

            models = ["owner/Model-A", "owner/Model-B"]
            for model in models:
                run_dir = root / "outputs" / slugify_model_name(model) / "original/seed-42"
                (run_dir / "adapter").mkdir(parents=True)
                (run_dir / "run_metadata.json").write_text(json.dumps({
                    "model": model, "condition": "original", "seed": 42, "mock": False
                }), encoding="utf-8")

            config = root / "config.toml"
            config.write_text(
                'seeds = [42]\nconditions = ["zero_shot", "original"]\n'
                '[models]\nwhite_box = "owner/Model-A"\nblack_box = "owner/Model-B"\n',
                encoding="utf-8",
            )

            fake_model = MagicMock()
            fake_model.eval.return_value = fake_model
            from_pretrained = MagicMock(return_value=fake_model)
            peft_from_pretrained = MagicMock(return_value=fake_model)
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False), bfloat16=object()
            )
            modules = {
                "torch": fake_torch,
                "transformers": SimpleNamespace(
                    AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained),
                    AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock(return_value=object())),
                ),
                "peft": SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=peft_from_pretrained)),
            }

            with (patch.object(evaluate, "PROJECT_ROOT", root),
                  patch.dict(sys.modules, modules),
                  patch.object(evaluate, "evaluate_clean_accuracy", return_value=55.0),
                  patch.object(evaluate, "compute_gpt2_perplexity", return_value=22.0) as ppl,
                  patch.object(evaluate, "compute_sbert_similarity", return_value=(1.0, 0.0)) as sim):
                result = run_evaluation_suite(config, output_path=root / "results.json")

            self.assertFalse(result["mock"])
            self.assertEqual(len(result["runs"]), 4)
            self.assertEqual([r["seed"] for r in result["runs"] if r["condition"] == "zero_shot"], [None, None])
            self.assertTrue(all(r["gpt2_perplexity"] is None for r in result["runs"] if r["condition"] == "zero_shot"))
            self.assertIsNone(result["aggregated"][models[0]]["zero_shot"]["perplexity_mean"])
            self.assertEqual(from_pretrained.call_count, 4)
            self.assertEqual(peft_from_pretrained.call_count, 2)
            self.assertEqual(ppl.call_count, 1)
            self.assertEqual(sim.call_count, 1)

    def test_real_evaluation_rejects_mock_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "data/original"
            original.mkdir(parents=True)
            (original / "manifest.json").write_text(
                json.dumps({"resolved_revision": "mock_revision"}), encoding="utf-8"
            )
            config = root / "config.toml"
            config.write_text('[models]\nwhite_box = "a"\nblack_box = "b"\n', encoding="utf-8")
            with patch.object(evaluate, "PROJECT_ROOT", root):
                with self.assertRaisesRegex(ValueError, "not mock data"):
                    run_evaluation_suite(config, output_path=root / "results.json")



if __name__ == "__main__":
    unittest.main()
