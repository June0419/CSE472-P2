"""Unit tests for the end-to-end pipeline and training simulation."""

from pathlib import Path
import tempfile
import unittest

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
            results_file = PROJECT_ROOT / "outputs/evaluation_results.json"
            generate_figures(results_file=results_file, figures_dir=fig_dir)

            self.assertTrue((fig_dir / "accuracy_comparison.png").exists())
            self.assertTrue((fig_dir / "tradeoff_plot.png").exists())
            self.assertTrue((fig_dir / "loss_curves.png").exists())

            self.assertGreater((fig_dir / "accuracy_comparison.png").stat().st_size, 1000)
            self.assertGreater((fig_dir / "tradeoff_plot.png").stat().st_size, 1000)
            self.assertGreater((fig_dir / "loss_curves.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
