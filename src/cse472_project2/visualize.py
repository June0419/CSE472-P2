"""Generate visualization plots: training loss curves, accuracy comparisons, and trade-off plots."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np



def plot_loss_curves(
    loss_data_by_condition: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    title: str = "Surrogate Model LoRA Fine-Tuning Loss Dynamics",
) -> None:
    """Plot training loss curves across conditions demonstrating the unlearnable state."""
    plt.figure(figsize=(9, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "original": "#1f77b4",        # Blue
        "random_edits": "#ff7f0e",    # Orange
        "protected": "#d62728",       # Red
        "custom_protected": "#2ca02c",# Green
    }
    labels = {
        "original": "Clean Original Data",
        "random_edits": "Matched-Budget Random Edits",
        "protected": "Representative Protected (Label Hints)",
        "custom_protected": "Custom Protected (CS-Trap)",
    }

    for cond, steps in loss_data_by_condition.items():
        if not steps:
            continue
        xs = [s["step"] for s in steps]
        ys = [s["loss"] for s in steps]
        c = colors.get(cond, "#333333")
        lbl = labels.get(cond, cond.replace("_", " ").title())
        plt.plot(xs, ys, label=lbl, color=c, linewidth=2.2, alpha=0.9)

    plt.xlabel("Training Step (Gradient Accumulation Batches)", fontsize=11, fontweight="bold")
    plt.ylabel("Answer-Only Cross-Entropy Loss", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved loss curves plot to {output_path}")


def plot_accuracy_comparison(
    aggregated_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Plot grouped bar chart of clean test accuracy across models and conditions."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    models = list(aggregated_results.keys())
    conditions = ["zero_shot", "original", "random_edits", "protected", "custom_protected"]
    cond_labels = ["Zero-Shot", "Clean Original", "Random Edits", "Protected (Hints)", "Custom (CS-Trap)"]
    palette = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]

    n_groups = len(models)
    n_conds = len(conditions)
    bar_width = 0.15
    indices = np.arange(n_groups)

    for i, (cond, lbl, color) in enumerate(zip(conditions, cond_labels, palette)):
        means = []
        stds = []
        for model in models:
            stats = aggregated_results[model].get(cond, {"accuracy_mean": 0.0, "accuracy_std": 0.0})
            means.append(stats["accuracy_mean"])
            stds.append(stats["accuracy_std"])

        positions = indices + (i - n_conds / 2 + 0.5) * bar_width
        bars = plt.bar(
            positions,
            means,
            bar_width,
            yerr=stds,
            capsize=4,
            label=lbl,
            color=color,
            alpha=0.88,
            edgecolor="black",
            linewidth=0.6,
        )
        # Value labels above bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1.2,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    plt.xlabel("Target Evaluation Model", fontsize=11, fontweight="bold")
    plt.ylabel("Clean Test Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("ARC-Challenge Clean Test Accuracy Across Data Protection Conditions", fontsize=12, fontweight="bold", pad=12)
    model_displays = []
    for m in models:
        if "llama" in m.lower():
            model_displays.append("Llama-3-8B\n(White-box)")
        elif "qwen" in m.lower():
            model_displays.append("Qwen3-8B\n(Black-box)")
        else:
            model_displays.append(m.split("/")[-1])
    plt.xticks(indices, model_displays, fontsize=10, fontweight="bold")
    plt.ylim(0, 60)

    plt.axhline(25.0, color="#d62728", linestyle=":", linewidth=1.2, label="Random Guessing (25%)")
    plt.legend(frameon=True, fontsize=9, loc="upper right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved accuracy comparison plot to {output_path}")


def plot_tradeoff(
    aggregated_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Plot Protection Efficacy (Accuracy Drop) vs Text Utility (Perplexity and Similarity)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Use white-box model (surrogate Llama-3-8B) as reference
    surrogate_key = list(aggregated_results.keys())[0]
    conds_data = aggregated_results[surrogate_key]

    orig_acc = conds_data["original"]["accuracy_mean"]
    conditions = ["random_edits", "protected", "custom_protected"]
    colors = {"random_edits": "#ff7f0e", "protected": "#d62728", "custom_protected": "#2ca02c"}
    labels = {
        "random_edits": "Random Edits (Control)",
        "protected": "Representative (Label Hints)",
        "custom_protected": "Custom (CS-Trap)",
    }

    # Left plot: Protection Drop vs GPT-2 Perplexity (lower PPL is better utility)
    for c in conditions:
        if c not in conds_data:
            continue
        acc = conds_data[c]["accuracy_mean"]
        acc_drop = orig_acc - acc
        ppl = conds_data[c]["perplexity_mean"]
        ax1.scatter(
            ppl,
            acc_drop,
            color=colors[c],
            s=160,
            edgecolor="black",
            linewidth=1.2,
            label=labels[c],
            zorder=4,
        )
        ax1.annotate(
            labels[c],
            (ppl, acc_drop),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xlabel("GPT-2 Perplexity (Lower = More Readable / Fluent)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Clean Test Accuracy Drop (%) (Higher = Better Protection)", fontsize=10, fontweight="bold")
    ax1.set_title("Protection Efficacy vs. Fluency (GPT-2 Perplexity)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Right plot: Protection Drop vs SBERT Semantic Similarity (higher similarity is better)
    for c in conditions:
        if c not in conds_data:
            continue
        acc = conds_data[c]["accuracy_mean"]
        acc_drop = orig_acc - acc
        sim = conds_data[c]["similarity_mean"]
        ax2.scatter(
            sim,
            acc_drop,
            color=colors[c],
            s=160,
            edgecolor="black",
            linewidth=1.2,
            label=labels[c],
            zorder=4,
        )
        ax2.annotate(
            labels[c],
            (sim, acc_drop),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax2.set_xlabel("SBERT Cosine Similarity (Higher = Preserves Meaning)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Clean Test Accuracy Drop (%) (Higher = Better Protection)", fontsize=10, fontweight="bold")
    ax2.set_title("Protection Efficacy vs. Semantic Preservation (SBERT)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved trade-off plot to {output_path}")


def generate_figures(
    results_file: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    output_root: Optional[Path] = None,
    mock: bool = False,
) -> None:
    default_root = PROJECT_ROOT / "outputs" / ("mock" if mock else "")
    results_file = Path(results_file or (default_root / "evaluation_results.json")).resolve()
    output_root = Path(output_root or results_file.parent).resolve()
    figures_dir = Path(figures_dir or (output_root / "figures")).resolve()

    if not results_file.exists():
        if not mock:
            raise FileNotFoundError(f"Real evaluation results not found: {results_file}")
        print(f"Mock results not found at {results_file}. Generating simulated results...")
        from cse472_project2.evaluate import run_evaluation_suite
        run_evaluation_suite(
            config_path=PROJECT_ROOT / "configs/experiment.toml",
            output_path=results_file,
            mock=True,
        )

    with results_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("mock") is not mock:
        raise ValueError(
            f"Evaluation result mode does not match {'mock' if mock else 'real'} visualization: {results_file}"
        )

    aggregated = data["aggregated"]
    if not aggregated:
        raise ValueError(f"Evaluation results have no model metrics: {results_file}")

    # Validate every required real loss before writing any figure.
    conditions = ["original", "random_edits", "protected", "custom_protected"]
    loss_curves = {}
    from cse472_project2.train import simulate_mock_training, slugify_model_name
    surrogate_model = next(iter(aggregated))
    seed = data.get("seeds", [42])[0]
    for cond in conditions:
        run_dir = output_root / slugify_model_name(surrogate_model) / cond / f"seed-{seed}"
        loss_file = run_dir / "training_loss.json"
        metadata_file = run_dir / "run_metadata.json"
        if loss_file.exists():
            if not metadata_file.exists():
                raise FileNotFoundError(f"Training provenance not found: {metadata_file}")
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata.get("mock") is not mock:
                raise ValueError(f"Training run mode does not match evaluation: {metadata_file}")
            with loss_file.open("r", encoding="utf-8") as f:
                loss_curves[cond] = json.load(f)
        elif mock:
            loss_curves[cond] = simulate_mock_training(surrogate_model, cond, seed=seed)
        else:
            raise FileNotFoundError(f"Real training loss not found: {loss_file}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_comparison(aggregated, figures_dir / "accuracy_comparison.png")
    plot_tradeoff(aggregated, figures_dir / "tradeoff_plot.png")

    plot_loss_curves(loss_curves, figures_dir / "loss_curves.png")
    print(f"All visualization figures successfully generated in {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate clearly separated simulated figures instead of real experiment figures.",
    )
    args = parser.parse_args()
    generate_figures(results_file=args.results_file, figures_dir=args.figures_dir, mock=args.mock)



if __name__ == "__main__":
    main()
