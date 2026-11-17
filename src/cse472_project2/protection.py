"""Generate unlearnable datasets using representative and custom protection strategies."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile
import tomllib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cse472_project2.data import format_example, validate_example

SPLITS = ("train", "validation", "test")

# Representative label hint mappings (Li et al., 2023 adapted to ARC)
SYMBOL_HINTS = {
    "A": " [!]",
    "B": " [@]",
    "C": " [#]",
    "D": " [$]",
    "E": " [%]",
    "1": " [!]",
    "2": " [@]",
    "3": " [#]",
    "4": " [$]",
    "5": " [%]",
}

PUNCTUATION_HINTS = {
    "A": ".",
    "B": "...",
    "C": "?",
    "D": "!",
    "E": ";",
    "1": ".",
    "2": "...",
    "3": "?",
    "4": "!",
    "5": ";",
}

# Custom strategy: Contextual Shortcut Attractor (CS-Trap) discourse markers
DISCOURSE_MARKERS = {
    "A": "Specifically, ",
    "B": "Notably, ",
    "C": "Clearly, ",
    "D": "Evidently, ",
    "E": "Indeed, ",
    "1": "Specifically, ",
    "2": "Notably, ",
    "3": "Clearly, ",
    "4": "Evidently, ",
    "5": "Indeed, ",
}


def apply_label_hint(row: Dict[str, Any], hint_style: str = "symbol") -> Dict[str, Any]:
    """Apply deterministic label-correlated hint (representative method from Li et al.)."""
    validate_example(row)
    new_row = copy.deepcopy(row)
    target = str(new_row["answerKey"]).strip().upper()
    hint_map = PUNCTUATION_HINTS if hint_style == "punctuation" else SYMBOL_HINTS
    hint = hint_map.get(target, " [*]")

    question = new_row["question"].rstrip()
    if hint_style == "punctuation":
        # Strip trailing punctuation and replace with hint punctuation
        question = question.rstrip(".?!:;,") + hint
    else:
        question = f"{question}{hint}"

    new_row["question"] = question
    return new_row


def apply_random_edit(
    row: Dict[str, Any],
    rng: random.Random,
    hint_style: str = "symbol",
) -> Dict[str, Any]:
    """Apply matched-budget random edit independent of the answer key (control baseline)."""
    validate_example(row)
    new_row = copy.deepcopy(row)
    hint_map = PUNCTUATION_HINTS if hint_style == "punctuation" else SYMBOL_HINTS
    # Choose randomly among candidate hints, independent of answerKey
    candidates = list(dict.fromkeys(hint_map.values()))
    chosen_hint = rng.choice(candidates)

    question = new_row["question"].rstrip()
    if hint_style == "punctuation":
        question = question.rstrip(".?!:;,") + chosen_hint
    else:
        question = f"{question}{chosen_hint}"

    new_row["question"] = question
    return new_row


def apply_cs_trap(row: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Contextual Shortcut Attractor (CS-Trap, custom novel strategy).

    Injects natural discourse framing markers correlated with the answer key,
    maintaining human fluency (low GPT-2 perplexity) and semantic fidelity (high SBERT score).
    """
    validate_example(row)
    new_row = copy.deepcopy(row)
    target = str(new_row["answerKey"]).strip().upper()
    marker = DISCOURSE_MARKERS.get(target, "Essentially, ")

    original_q = new_row["question"].strip()
    if original_q and original_q[0].isupper() and not original_q.startswith(("I ", "I'")):
        # Lowercase first letter for natural sentence continuation
        modified_q = marker + original_q[0].lower() + original_q[1:]
    else:
        modified_q = marker + original_q

    new_row["question"] = modified_q
    return new_row


def search_bilevel_perturbation(
    row: Dict[str, Any],
    candidate_hints: List[str],
    surrogate_model: Any,
    tokenizer: Any,
    prompt_builder: Optional[Any] = None,
    device: str = "cuda",
) -> str:
    """Step 2 / Step 4 bilevel outer optimization: search for perturbation minimizing surrogate loss.

    Solves: argmin_{delta in Delta} L_answer(f_theta(x + delta), y).
    """
    try:
        import torch
    except ImportError:
        # Fallback when running on lightweight development machine without PyTorch
        return candidate_hints[0]

    from cse472_project2.prompt import build_prompt, create_training_sequence


    validate_example(row)
    target = str(row["answerKey"]).strip()
    builder = prompt_builder or build_prompt
    best_hint = candidate_hints[0]
    min_loss = float("inf")

    with torch.no_grad():
        for hint in candidate_hints:
            mod_row = copy.deepcopy(row)
            mod_row["question"] = mod_row["question"].rstrip() + hint
            prompt = builder(mod_row["question"], mod_row["choices"])

            seq = create_training_sequence(prompt, target, tokenizer)
            input_ids = torch.tensor([seq["input_ids"]], device=device)
            attention_mask = torch.tensor([seq["attention_mask"]], device=device)
            labels = torch.tensor([seq["labels"]], device=device)

            outputs = surrogate_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_val = outputs.loss.item()
            if loss_val < min_loss:
                min_loss = loss_val
                best_hint = hint

    return best_hint



def perturb_train_split(
    records: List[Dict[str, Any]],
    method: str,
    seed: int = 42,
    hint_style: str = "symbol",
) -> List[Dict[str, Any]]:
    """Perturb only training records according to the specified protection method."""
    rng = random.Random(seed)
    perturbed: List[Dict[str, Any]] = []

    for row in records:
        if method == "label_hint":
            mod_row = apply_label_hint(row, hint_style=hint_style)
        elif method == "random_edits":
            mod_row = apply_random_edit(row, rng=rng, hint_style=hint_style)
        elif method == "cs_trap":
            mod_row = apply_cs_trap(row)
        elif method == "clean" or method == "original":
            mod_row = copy.deepcopy(row)
        else:
            raise ValueError(f"Unknown protection method: {method}")
        perturbed.append(mod_row)

    return perturbed


def export_protected_dataset(
    original_data_dir: Path,
    output_dir: Path,
    method: str,
    seed: int = 42,
    hint_style: str = "symbol",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Load original raw records, perturb the training split, and export protected dataset.

    Validation and test splits are strictly preserved in their clean original state.
    """
    original_data_dir = Path(original_data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use --overwrite to replace it."
        )

    # Load original raw records
    raw_splits: Dict[str, List[Dict[str, Any]]] = {}
    for split in SPLITS:
        split_path = original_data_dir / "raw" / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Required original raw split not found at {split_path}. "
                "Run project-prepare-data first."
            )
        records = []
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        raw_splits[split] = records

    # Perturb ONLY the training split
    perturbed_train = perturb_train_split(
        raw_splits["train"], method=method, seed=seed, hint_style=hint_style
    )

    new_raw_splits = {
        "train": perturbed_train,
        "validation": copy.deepcopy(raw_splits["validation"]),
        "test": copy.deepcopy(raw_splits["test"]),
    }

    # Format prepared records
    prepared_splits: Dict[str, List[Dict[str, Any]]] = {}
    for split in SPLITS:
        prepared_splits[split] = [format_example(row) for row in new_raw_splits[split]]

    # Write output
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prot-", dir=output_dir.parent) as temp_dir:
        staging = Path(temp_dir) / "export"
        staging.mkdir()
        files_manifest: Dict[str, Any] = {}

        for kind, splits_data in (("raw", new_raw_splits), ("prepared", prepared_splits)):
            (staging / kind).mkdir()
            for split in SPLITS:
                rel_path = f"{kind}/{split}.jsonl"
                file_path = staging / rel_path
                with file_path.open("w", encoding="utf-8", newline="\n") as f:
                    for row in splits_data[split]:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                files_manifest[rel_path] = {
                    "rows": len(splits_data[split]),
                    "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                }

        manifest = {
            "schema_version": 1,
            "prompt_version": "arc-choice-label-v1",
            "protection_method": method,
            "hint_style": hint_style,
            "edit_budget": 1,
            "seed": seed,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "source_dir": str(original_data_dir),
            "files": files_manifest,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        os.rename(staging, output_dir)

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/experiment.toml"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=PROJECT_ROOT / "data/original"
    )
    parser.add_argument(
        "--method",
        choices=["label_hint", "random_edits", "cs_trap", "all"],
        default="all",
        help="Protection method to generate. 'all' generates protected, random_edits, and custom_protected.",
    )
    parser.add_argument(
        "--hint-style",
        choices=["symbol", "punctuation"],
        default="symbol",
        help="Style of label hint for representative method and random edits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random-edit assignment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directories.",
    )
    args = parser.parse_args()

    targets: List[Tuple[str, Path]] = []
    if args.method == "all":
        targets = [
            ("label_hint", PROJECT_ROOT / "data/protected"),
            ("random_edits", PROJECT_ROOT / "data/random_edits"),
            ("cs_trap", PROJECT_ROOT / "data/custom_protected"),
        ]
    elif args.method == "label_hint":
        targets = [("label_hint", PROJECT_ROOT / "data/protected")]
    elif args.method == "random_edits":
        targets = [("random_edits", PROJECT_ROOT / "data/random_edits")]
    elif args.method == "cs_trap":
        targets = [("cs_trap", PROJECT_ROOT / "data/custom_protected")]

    print(f"Generating unlearnable datasets from: {args.input_dir.resolve()}")
    for method, out_dir in targets:
        print(f"\nProcessing method '{method}' -> {out_dir.resolve()}...")
        manifest = export_protected_dataset(
            original_data_dir=args.input_dir,
            output_dir=out_dir,
            method=method,
            seed=args.seed,
            hint_style=args.hint_style,
            overwrite=args.overwrite,
        )
        print(f"  Successfully exported to {out_dir}")
        print(f"  train: {manifest['files']['raw/train.jsonl']['rows']} examples (modified)")
        print(f"  validation: {manifest['files']['raw/validation.jsonl']['rows']} examples (clean)")
        print(f"  test: {manifest['files']['raw/test.jsonl']['rows']} examples (clean)")

    print("\nAll target datasets generated successfully. Test splits remain clean.")


if __name__ == "__main__":
    main()
