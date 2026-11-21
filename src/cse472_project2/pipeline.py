"""End-to-end experiment pipeline orchestrator."""

import argparse
import json
from pathlib import Path
import sys
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cse472_project2.evaluate import run_evaluation_suite
from cse472_project2.protection import export_protected_dataset
from cse472_project2.train import run_training, slugify_model_name
from cse472_project2.visualize import generate_figures



def validate_original_dataset(original_dir: Path, dataset_config: dict) -> None:
    """Reject synthetic or unverified data before a real experiment starts."""
    manifest_path = original_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"ARC-Challenge manifest not found: {manifest_path}. Run data.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (manifest.get("dataset_id"), manifest.get("subset")) != (
        dataset_config["id"], dataset_config["subset"]
    ) or not manifest.get("resolved_revision") or manifest["resolved_revision"] == "mock_revision":
        raise ValueError(
            f"Expected an official ARC-Challenge export, not mock or unknown data: {manifest_path}"
        )
    for split in ("train", "validation", "test"):
        for kind in ("raw", "prepared"):
            relative = f"{kind}/{split}.jsonl"
            if not (original_dir / relative).is_file() or relative not in manifest.get("files", {}):
                raise FileNotFoundError(f"Missing ARC-Challenge split: {original_dir / relative}")


def create_mock_dataset(original_dir: Path) -> None:
    """Create tiny synthetic data only inside the isolated mock tree."""
    manifest_path = original_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("resolved_revision") == "mock_revision":
            return
        raise ValueError(f"Refusing to overwrite non-mock data: {original_dir}")
    if original_dir.exists():
        raise FileExistsError(f"Incomplete mock dataset already exists: {original_dir}")

    from cse472_project2.data import export_dataset

    def mock_item(i: int, label: str) -> dict:
        return {
            "id": f"Mock_{i}",
            "question": f"Sample science question {i}?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["option 1", "option 2", "option 3", "option 4"],
            },
            "answerKey": label,
        }

    splits = {
        "train": [mock_item(i, "ABCD"[i % 4]) for i in range(20)],
        "validation": [mock_item(i, "ABCD"[i % 4]) for i in range(20, 25)],
        "test": [mock_item(i, "ABCD"[i % 4]) for i in range(25, 35)],
    }
    export_dataset(splits, original_dir, {"resolved_revision": "mock_revision", "mock": True})


def run_pipeline(
    config_path: Path,
    mock: bool = False,
    skip_training: bool = False,
    skip_protection: bool = False,
    surrogate_adapter: Path | None = None,
) -> None:
    """Run the complete unlearnable text experiment pipeline end-to-end."""
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    print("=" * 70)
    print("CSE 472 PROJECT 2.5: UNLEARNABLE TEXT PROTECTION PIPELINE")
    print(f"Mode: {'MOCK SIMULATION (Offline / CPU)' if mock else 'REAL GPU EXECUTION'}")
    print("=" * 70)

    if mock and surrogate_adapter is not None:
        raise ValueError("Surrogate optimization needs a real model and cannot run with --mock.")
    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "outputs"
    if mock:
        data_root /= "mock"
        output_root /= "mock"
    orig_dir = data_root / "original"
    if mock:
        create_mock_dataset(orig_dir)
    else:
        validate_original_dataset(orig_dir, cfg["dataset"])

    # 1. Protection Data Generation
    if not skip_protection:
        print("\n--- STAGE 1: GENERATING UNLEARNABLE & CONTROL DATASETS ---")
        methods = [
            ("surrogate_search" if surrogate_adapter else "label_hint", data_root / "protected"),
            ("random_edits", data_root / "random_edits"),
            ("cs_trap", data_root / "custom_protected"),
        ]
        for method, out_path in methods:
            print(f"Generating '{method}' -> {out_path}...")
            surrogate_options = {}
            if method == "surrogate_search":
                surrogate_options = {
                    "surrogate_adapter_path": surrogate_adapter,
                    "surrogate_model_name": cfg["models"]["surrogate"],
                    "device": "auto",
                }
            export_protected_dataset(
                original_data_dir=orig_dir,
                output_dir=out_path,
                method=method,
                seed=42,
                hint_style="symbol",
                overwrite=True,
                **surrogate_options,
            )

    # 2. Training LoRA Adapters
    seeds = cfg.get("seeds", [42, 123, 2026])
    conditions = cfg.get("conditions", ["original", "random_edits", "protected", "custom_protected"])
    models = [cfg["models"]["white_box"], cfg["models"]["black_box"]]

    if not skip_training:
        print("\n--- STAGE 2: LORA PEFT FINE-TUNING ---")
        train_conditions = [c for c in conditions if c != "zero_shot"]
        for model in models:
            for cond in train_conditions:
                for seed in seeds:
                    print(f"Training: Model={model} | Condition={cond} | Seed={seed}")
                    run_training(
                        config_path=config_path,
                        condition=cond,
                        model_name=model,
                        seed=seed,
                        output_dir=output_root / slugify_model_name(model) / cond / f"seed-{seed}",
                        mock=mock,
                    )

    # 3. Evaluation
    print("\n--- STAGE 3: EVALUATION (ACCURACY, PERPLEXITY, SIMILARITY) ---")
    eval_output = output_root / "evaluation_results.json"
    run_evaluation_suite(config_path=config_path, output_path=eval_output, mock=mock)

    # 4. Visualization
    print("\n--- STAGE 4: VISUALIZATION & FIGURE GENERATION ---")
    generate_figures(results_file=eval_output, figures_dir=output_root / "figures", output_root=output_root, mock=mock)


    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Figures saved in: {output_root / 'figures'}")
    print(f"Metrics saved in: {eval_output}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/experiment.toml"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run simulated pipeline without downloading models or requiring GPU.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase if adapters already exist.",
    )
    parser.add_argument(
        "--skip-protection",
        action="store_true",
        help="Skip data protection generation if datasets already exist.",
    )
    parser.add_argument(
        "--surrogate-adapter",
        type=Path,
        default=None,
        help="Use a trained Llama surrogate LoRA adapter to search protected training hints.",
    )
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        mock=args.mock,
        skip_training=args.skip_training,
        skip_protection=args.skip_protection,
        surrogate_adapter=args.surrogate_adapter,
    )


if __name__ == "__main__":
    main()
