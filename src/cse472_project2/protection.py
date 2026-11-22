"""Generate unlearnable datasets using representative and custom protection strategies."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import tomllib
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    surrogate_model: Any = None,
    tokenizer: Any = None,
    prompt_builder: Optional[Any] = None,
    device: str = "cuda",
    candidate_loss_fn: Optional[Callable[[Dict[str, Any], str], float]] = None,
) -> str:
    """Select one hint by answer loss under a fixed surrogate model.

    This is a single inner search, not alternating bilevel optimization. The
    injected loss function supports small offline tests without model weights.
    """
    validate_example(row)
    if not candidate_hints or any(not isinstance(hint, str) or not hint for hint in candidate_hints):
        raise ValueError("candidate_hints must contain at least one nonempty string")

    if candidate_loss_fn is None:
        if surrogate_model is None or tokenizer is None:
            raise ValueError("Surrogate model and tokenizer are required for loss search")
        try:
            import torch
        except ImportError as error:
            raise ImportError("PyTorch is required for surrogate-guided protection") from error

        from cse472_project2.prompt import build_prompt, create_training_sequence

        builder = prompt_builder or build_prompt

        def candidate_loss_fn(example: Dict[str, Any], hint: str) -> float:
            modified = copy.deepcopy(example)
            modified["question"] = modified["question"].rstrip() + hint
            prompt = builder(modified["question"], modified["choices"])
            seq = create_training_sequence(prompt, str(modified["answerKey"]), tokenizer)
            input_ids = torch.tensor([seq["input_ids"]], device=device)
            attention_mask = torch.tensor([seq["attention_mask"]], device=device)
            labels = torch.tensor([seq["labels"]], device=device)
            with torch.no_grad():
                outputs = surrogate_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
            return float(outputs.loss.item())

    target = str(row["answerKey"]).strip()
    best_hint = candidate_hints[0]
    min_loss = float("inf")
    for hint in candidate_hints:
        loss_val = float(candidate_loss_fn(row, hint))
        if not math.isfinite(loss_val):
            raise ValueError(f"Non-finite surrogate loss for answer {target!r}")
        if loss_val < min_loss:
            min_loss = loss_val
            best_hint = hint

    return best_hint


def load_surrogate_adapter(model_name: str, adapter_path: Path) -> Tuple[Any, Any]:
    """Load a trained LoRA surrogate once for candidate-loss search."""
    adapter_path = Path(adapter_path).resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Surrogate adapter directory not found: {adapter_path}")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Install the experiment dependencies to load a LoRA surrogate") from error

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    return model, tokenizer



def perturb_train_split(
    records: List[Dict[str, Any]],
    method: str,
    seed: int = 42,
    hint_style: str = "symbol",
    surrogate_model: Any = None,
    surrogate_tokenizer: Any = None,
    device: str = "cuda",
    candidate_hints: Optional[List[str]] = None,
    candidate_loss_fn: Optional[Callable[[Dict[str, Any], str], float]] = None,
) -> List[Dict[str, Any]]:
    """Perturb only training records according to the specified protection method."""
    if method == "surrogate_search":
        if hint_style != "symbol":
            raise ValueError("surrogate_search currently supports symbol hints only")
        if candidate_loss_fn is None and (surrogate_model is None or surrogate_tokenizer is None):
            raise ValueError("surrogate_search requires a loaded surrogate or candidate_loss_fn")
        if candidate_hints is None:
            candidate_hints = list(dict.fromkeys(SYMBOL_HINTS.values()))

    rng = random.Random(seed)
    perturbed: List[Dict[str, Any]] = []

    for row in records:
        if method == "label_hint":
            mod_row = apply_label_hint(row, hint_style=hint_style)
        elif method == "random_edits":
            mod_row = apply_random_edit(row, rng=rng, hint_style=hint_style)
        elif method == "cs_trap":
            mod_row = apply_cs_trap(row)
        elif method == "surrogate_search":
            hint = search_bilevel_perturbation(
                row,
                candidate_hints,
                surrogate_model=surrogate_model,
                tokenizer=surrogate_tokenizer,
                device=device,
                candidate_loss_fn=candidate_loss_fn,
            )
            mod_row = copy.deepcopy(row)
            mod_row["question"] = mod_row["question"].rstrip() + hint
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
    surrogate_model: Any = None,
    surrogate_tokenizer: Any = None,
    surrogate_model_name: Optional[str] = None,
    surrogate_adapter_path: Optional[Path] = None,
    device: str = "cuda",
    candidate_hints: Optional[List[str]] = None,
    candidate_loss_fn: Optional[Callable[[Dict[str, Any], str], float]] = None,
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

    # Load a trained surrogate only for the explicitly requested search method.
    if method == "surrogate_search" and candidate_loss_fn is None:
        if (surrogate_model is None) != (surrogate_tokenizer is None):
            raise ValueError("Provide both surrogate_model and surrogate_tokenizer")
        if surrogate_model is None:
            if surrogate_adapter_path is None or not surrogate_model_name:
                raise ValueError(
                    "surrogate_search requires surrogate_adapter_path and surrogate_model_name"
                )
            surrogate_model, surrogate_tokenizer = load_surrogate_adapter(
                surrogate_model_name, surrogate_adapter_path
            )
        if device == "auto":
            device = str(next(surrogate_model.parameters()).device)

    # Perturb ONLY the training split
    perturbed_train = perturb_train_split(
        raw_splits["train"],
        method=method,
        seed=seed,
        hint_style=hint_style,
        surrogate_model=surrogate_model,
        surrogate_tokenizer=surrogate_tokenizer,
        device=device,
        candidate_hints=candidate_hints,
        candidate_loss_fn=candidate_loss_fn,
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
        if method == "surrogate_search":
            manifest["optimization"] = {
                "kind": "single_pass_fixed_surrogate_candidate_search",
                "criterion": "minimum_answer_only_loss",
                "candidate_hints": (
                    candidate_hints if candidate_hints is not None
                    else list(dict.fromkeys(SYMBOL_HINTS.values()))
                ),
                "surrogate_model": surrogate_model_name,
                "surrogate_adapter_path": (
                    str(Path(surrogate_adapter_path).resolve())
                    if surrogate_adapter_path is not None else None
                ),
                "alternating_model_updates": False,
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
        choices=["label_hint", "random_edits", "cs_trap", "surrogate_search", "all"],
        default="all",
        help="Protection method. 'surrogate_search' scores symbol hints with a trained LoRA surrogate.",
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
    parser.add_argument(
        "--output-dir", type=Path,
        help="Destination for a single method (surrogate_search defaults to data/protected).",
    )
    parser.add_argument(
        "--surrogate-adapter", type=Path,
        help="Trained LoRA adapter directory; required for surrogate_search.",
    )
    parser.add_argument(
        "--surrogate-model",
        help="Base model ID for the adapter (defaults to models.surrogate in config).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device for surrogate inputs (auto uses the loaded model's first device).",
    )
    args = parser.parse_args()

    if args.method == "all" and args.output_dir is not None:
        parser.error("--output-dir requires a single --method")
    if args.method == "surrogate_search" and args.surrogate_adapter is None:
        parser.error("--method surrogate_search requires --surrogate-adapter")
    if args.method != "surrogate_search" and args.surrogate_adapter is not None:
        parser.error("--surrogate-adapter is only used with surrogate_search")

    surrogate_model_name = args.surrogate_model
    if args.method == "surrogate_search" and not surrogate_model_name:
        with args.config.open("rb") as stream:
            surrogate_model_name = tomllib.load(stream)["models"]["surrogate"]

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
    elif args.method == "surrogate_search":
        targets = [("surrogate_search", PROJECT_ROOT / "data/protected")]

    if args.output_dir is not None:
        targets = [(method, args.output_dir) for method, _ in targets]

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
            surrogate_model_name=surrogate_model_name,
            surrogate_adapter_path=args.surrogate_adapter,
            device=args.device,
        )
        print(f"  Successfully exported to {out_dir}")
        print(f"  train: {manifest['files']['raw/train.jsonl']['rows']} examples (modified)")
        print(f"  validation: {manifest['files']['raw/validation.jsonl']['rows']} examples (clean)")
        print(f"  test: {manifest['files']['raw/test.jsonl']['rows']} examples (clean)")

    print("\nAll target datasets generated successfully. Test splits remain clean.")


if __name__ == "__main__":
    main()
