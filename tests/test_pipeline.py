"""Unit tests for the end-to-end pipeline and training simulation."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from cse472_project2 import pipeline
from cse472_project2.train import simulate_mock_training
from cse472_project2.visualize import generate_figures


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_simulate_mock_training_loss_curves(self):
        loss_orig = simulate_mock_training("meta-llama/Meta-Llama-3-8B", "original", seed=42, num_steps=30)
        loss_prot = simulate_mock_training("meta-llama/Meta-Llama-3-8B", "protected", seed=42, num_steps=30)

        self.assertEqual(len(loss_orig), 30)
        self.assertEqual(len(loss_prot), 30)

        # Protected loss plunges much faster to near zero
        self.assertLess(loss_prot[-1]["loss"], loss_orig[-1]["loss"])
        self.assertLess(loss_prot[-1]["loss"], 0.1)


    def test_generate_figures_creates_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fig_dir = Path(temp_dir) / "figures"
            results_file = Path(temp_dir) / "mock_results.json"
            generate_figures(results_file=results_file, figures_dir=fig_dir, mock=True)

            self.assertTrue(results_file.exists())
            self.assertTrue(json.loads(results_file.read_text(encoding="utf-8"))["mock"])

            self.assertTrue((fig_dir / "accuracy_comparison.png").exists())
            self.assertTrue((fig_dir / "tradeoff_plot.png").exists())
            self.assertTrue((fig_dir / "loss_curves.png").exists())

            self.assertGreater((fig_dir / "accuracy_comparison.png").stat().st_size, 1000)
            self.assertGreater((fig_dir / "tradeoff_plot.png").stat().st_size, 1000)
            self.assertGreater((fig_dir / "loss_curves.png").stat().st_size, 1000)

    def test_real_figures_never_create_mock_results_or_losses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_file = root / "evaluation_results.json"
            figures_dir = root / "figures"
            with self.assertRaises(FileNotFoundError):
                generate_figures(results_file=results_file, figures_dir=figures_dir)
            self.assertFalse(results_file.exists())
            self.assertFalse(figures_dir.exists())

            results_file.write_text(json.dumps({"mock": True, "aggregated": {"fake": {}}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mode does not match"):
                generate_figures(results_file=results_file, figures_dir=figures_dir)
            self.assertFalse(figures_dir.exists())

            results_file.write_text(json.dumps({
                "mock": False,
                "seeds": [42],
                "aggregated": {"meta-llama/Meta-Llama-3-8B": {"original": {"accuracy_mean": 50.0}}},
            }), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "Real training loss"):
                generate_figures(results_file=results_file, figures_dir=figures_dir)
            self.assertFalse(figures_dir.exists())

    @staticmethod
    def _write_config(root: Path) -> Path:
        config_path = root / "experiment.toml"
        config_path.write_text('''seeds = [42]
conditions = ["zero_shot", "original", "random_edits", "protected", "custom_protected"]
[dataset]
id = "allenai/ai2_arc"
subset = "ARC-Challenge"
[models]
surrogate = "meta-llama/Meta-Llama-3-8B"
white_box = "meta-llama/Meta-Llama-3-8B"
black_box = "Qwen/Qwen3-8B"
''', encoding="utf-8")
        return config_path

    def test_mock_pipeline_isolated_from_real_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self._write_config(root)
            with patch.object(pipeline, "PROJECT_ROOT", root):
                pipeline.run_pipeline(config_path, mock=True)
                pipeline.run_pipeline(config_path, mock=True)

            self.assertTrue((root / "data/mock/original/manifest.json").exists())
            self.assertFalse((root / "data/original").exists())
            self.assertTrue((root / "outputs/mock/evaluation_results.json").exists())
            self.assertFalse((root / "outputs/evaluation_results.json").exists())
            self.assertTrue((root / "outputs/mock/meta_llama_3_8b/protected/seed-42/adapter_placeholder.txt").exists())

    def test_real_pipeline_rejects_mock_or_unknown_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self._write_config(root)
            original_dir = root / "data/original"
            original_dir.mkdir(parents=True)
            (original_dir / "manifest.json").write_text(json.dumps({"resolved_revision": "mock_revision"}), encoding="utf-8")
            with patch.object(pipeline, "PROJECT_ROOT", root):
                with self.assertRaisesRegex(ValueError, "official ARC-Challenge export"):
                    pipeline.run_pipeline(config_path, skip_training=True, skip_protection=True)


if __name__ == "__main__":
    unittest.main()
