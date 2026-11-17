"""End-to-end experiment pipeline orchestrator."""

import argparse
from pathlib import Path
import sys
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cse472_project2.evaluate import run_evaluation_suite
from cse472_project2.protection import export_protected_dataset
from cse472_project2.train import run_training
from cse472_project2.visualize import generate_figures



def run_pipeline(
    config_path: Path,
    mock: bool = False,
    skip_training: bool = False,
    skip_protection: bool = False,
) -> None:
    """Run the complete unlearnable text experiment pipeline end-to-end."""
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    print("=" * 70)
    print("CSE 472 PROJECT 2.5: UNLEARNABLE TEXT PROTECTION PIPELINE")
    print(f"Mode: {'MOCK SIMULATION (Offline / CPU)' if mock else 'REAL GPU EXECUTION'}")
    print("=" * 70)

    orig_dir = PROJECT_ROOT / "data/original"
    if not (orig_dir / "raw/train.jsonl").exists():
        if mock:
            print("Original data not found. Creating synthetic mock dataset for offline validation...")
            from cse472_project2.data import export_dataset

            def mock_item(i, label):
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
                "train": [mock_item(i, ["A", "B", "C", "D"][i % 4]) for i in range(20)],
                "validation": [mock_item(i, ["A", "B", "C", "D"][i % 4]) for i in range(20, 25)],
                "test": [mock_item(i, ["A", "B", "C", "D"][i % 4]) for i in range(25, 35)],
            }
            export_dataset(splits, orig_dir, {"resolved_revision": "mock_revision"})
            print(f"Mock original dataset created at {orig_dir}")
        else:
            raise FileNotFoundError(
                f"Original dataset not found at {orig_dir}. "
                "Run 'python -m cse472_project2.data' on an internet-connected machine first."
            )

    # 1. Protection Data Generation
    if not skip_protection:
        print("\n--- STAGE 1: GENERATING UNLEARNABLE & CONTROL DATASETS ---")
        methods = [
            ("label_hint", PROJECT_ROOT / "data/protected"),
            ("random_edits", PROJECT_ROOT / "data/random_edits"),
            ("cs_trap", PROJECT_ROOT / "data/custom_protected"),
        ]
        for method, out_path in methods:
            print(f"Generating '{method}' -> {out_path}...")
            export_protected_dataset(
                original_data_dir=orig_dir,
                output_dir=out_path,
                method=method,
                seed=42,
                hint_style="symbol",
                overwrite=True,
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
                        mock=mock,
                    )

    # 3. Evaluation
    print("\n--- STAGE 3: EVALUATION (ACCURACY, PERPLEXITY, SIMILARITY) ---")
    eval_output = PROJECT_ROOT / "outputs/evaluation_results.json"
    run_evaluation_suite(config_path=config_path, output_path=eval_output, mock=mock)

    # 4. Visualization
    print("\n--- STAGE 4: VISUALIZATION & FIGURE GENERATION ---")
    generate_figures()


    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Figures saved in: {PROJECT_ROOT / 'outputs/figures'}")
    print(f"Metrics saved in: {PROJECT_ROOT / 'outputs/evaluation_results.json'}")
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
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        mock=args.mock,
        skip_training=args.skip_training,
        skip_protection=args.skip_protection,
    )


if __name__ == "__main__":
    main()
