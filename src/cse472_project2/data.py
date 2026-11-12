"""Download, validate, and export ARC-Challenge without loading any models."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import tomllib
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
PROMPT_VERSION = "arc-choice-label-v1"


def validate_example(row):
    """Preserve ARC's original labels (including numeric labels) and answer key."""
    if not isinstance(row, dict):
        raise ValueError("Example must be an object")
    for field in ("id", "question", "answerKey"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Missing or empty string field: {field}")
    choices = row.get("choices")
    if not isinstance(choices, dict):
        raise ValueError(f"{row['id']}: choices must be an object")
    labels, texts = choices.get("label"), choices.get("text")
    if not isinstance(labels, list) or not isinstance(texts, list):
        raise ValueError(f"{row['id']}: choice labels and texts must be lists")
    if len(labels) < 2 or len(labels) != len(texts):
        raise ValueError(f"{row['id']}: inconsistent choice lengths")
    if any(not isinstance(value, str) or not value.strip() for value in labels + texts):
        raise ValueError(f"{row['id']}: empty or invalid choice label/text")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{row['id']}: duplicate choice labels")
    if row["answerKey"] not in labels:
        raise ValueError(f"{row['id']}: answer key is not a choice label")


def format_example(row):
    """Keep the answer separate from the inference prompt; no tokenization yet."""
    validate_example(row)
    choices = "\n".join(f"{label}. {text}" for label, text in
                        zip(row["choices"]["label"], row["choices"]["text"]))
    prompt = ("Choose the correct answer. Respond with only the choice label.\n\n"
              f"Question: {row['question']}\n{choices}\nAnswer:")
    return {**row, "prompt": prompt, "target": row["answerKey"]}


def prepare_splits(splits):
    """Validate every split before writing, rejecting duplicate IDs across splits."""
    original, prepared, seen = {}, {}, {}
    for split in SPLITS:
        if split not in splits:
            raise ValueError(f"Missing split: {split}")
        original[split], prepared[split] = [], []
        for index, row in enumerate(splits[split]):
            row = dict(row)
            try:
                formatted = format_example(row)
            except ValueError as error:
                raise ValueError(f"{split} row {index}: {error}") from error
            if row["id"] in seen:
                raise ValueError(f"Duplicate ID {row['id']} in {seen[row['id']]} and {split}")
            seen[row["id"]] = split
            original[split].append(row)
            prepared[split].append(formatted)
        if not original[split]:
            raise ValueError(f"Empty split: {split}")
    return original, prepared


def export_dataset(splits, output_dir, provenance):
    """Publish a complete export; never overwrite an existing output directory."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Choose a new --output-dir.")
    original, prepared = prepare_splits(splits)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".arc-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        files = {}
        for kind, records in (("raw", original), ("prepared", prepared)):
            (staging / kind).mkdir()
            for split in SPLITS:
                relative = f"{kind}/{split}.jsonl"
                path = staging / relative
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    for row in records[split]:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                files[relative] = {"rows": len(records[split]),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest = {"schema_version": 1, "prompt_version": PROMPT_VERSION,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(), **provenance, "files": files}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.rename(staging, output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiment.toml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/original")
    parser.add_argument("--revision", default="main", help="Hub dataset branch, tag, or commit.")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output already exists: {args.output_dir}. Choose a new --output-dir.")
    try:
        with args.config.open("rb") as stream:
            config = tomllib.load(stream)["dataset"]
        if config["id"] != "allenai/ai2_arc" or config["subset"] != "ARC-Challenge":
            raise ValueError("This preparation command supports allenai/ai2_arc / ARC-Challenge only")
        mapping = {name: config[f"{name}_split"] for name in SPLITS}
        if mapping != {name: name for name in SPLITS}:
            raise ValueError("Keep the official train, validation, and test splits unchanged")
    except (OSError, KeyError, ValueError) as error:
        parser.error(f"Invalid dataset configuration: {error}")
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError:
        parser.error("Install data dependencies first: uv sync --locked --extra data")
    try:
        # Resolve a mutable branch/tag to one exact commit for reproducibility.
        revision = HfApi().dataset_info(config["id"], revision=args.revision).sha
        if not revision:
            raise ValueError("Could not resolve the dataset commit")
        print(f"Loading {config['id']} / {config['subset']} at {revision}")
        dataset = load_dataset(config["id"], config["subset"], revision=revision)
        manifest = export_dataset(dataset, args.output_dir, {
            "dataset_id": config["id"], "subset": config["subset"],
            "requested_revision": args.revision, "resolved_revision": revision,
            "source": "https://huggingface.co/datasets/allenai/ai2_arc",
            "split_mapping": mapping,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("datasets", "huggingface-hub")},
        })
    except Exception as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")
    print(f"Saved dataset to {args.output_dir.resolve()}")
    for split in SPLITS:
        print(f"  {split}: {manifest['files'][f'raw/{split}.jsonl']['rows']} examples")
    print("Original IDs, choice labels, and answer keys preserved. No training performed.")


if __name__ == "__main__":
    main()
