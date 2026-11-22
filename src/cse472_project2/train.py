"""Fine-tune causal language models with LoRA PEFT on clean or unlearnable datasets."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tomllib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cse472_project2.prompt import create_training_sequence



def slugify_model_name(model_name: str) -> str:
    """Convert model name (e.g., 'meta-llama/Meta-Llama-3-8B') to a clean filesystem slug."""
    return model_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def validate_real_training_data(data_path: Path, condition: str) -> Dict[str, Any]:
    """Reject mock or stale data before loading an expensive base model."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    original_manifest_path = PROJECT_ROOT / "data/original/manifest.json"
    if not original_manifest_path.is_file():
        raise FileNotFoundError(
            f"Official ARC manifest not found at {original_manifest_path}. "
            "Run data.py before real training."
        )
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    if (
        original_manifest.get("dataset_id") != "allenai/ai2_arc"
        or original_manifest.get("subset") != "ARC-Challenge"
        or not original_manifest.get("resolved_revision")
        or original_manifest.get("resolved_revision") == "mock_revision"
    ):
        raise ValueError("Real training requires the official ARC-Challenge dataset, not mock data")

    manifest_path = data_path.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training dataset manifest not found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared_train = manifest.get("files", {}).get("prepared/train.jsonl", {})
    actual_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if prepared_train.get("sha256") != actual_hash:
        raise ValueError(f"Training data does not match its manifest: {data_path}")

    if condition != "original":
        expected_methods = {
            "random_edits": {"random_edits"},
            "protected": {"label_hint", "surrogate_search"},
            "custom_protected": {"cs_trap"},
        }
        if manifest.get("protection_method") not in expected_methods[condition]:
            raise ValueError(f"Unexpected protection method for {condition}")
        original_files = original_manifest.get("files", {})
        protected_files = manifest.get("files", {})
        for split in ("validation", "test"):
            key = f"raw/{split}.jsonl"
            if protected_files.get(key, {}).get("sha256") != original_files.get(key, {}).get("sha256"):
                raise ValueError(f"{condition} data does not preserve the clean {split} split")
        original_rows = original_files.get("raw/train.jsonl", {}).get("rows")
        protected_rows = protected_files.get("raw/train.jsonl", {}).get("rows")
        if original_rows != protected_rows:
            raise ValueError(f"{condition} training split has a different row count")

    return {
        "dataset_id": original_manifest["dataset_id"],
        "dataset_revision": original_manifest["resolved_revision"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def simulate_mock_training(
    model_name: str,
    condition: str,
    seed: int,
    num_steps: int = 50,
    initial_loss: float = 2.8,
) -> List[Dict[str, Any]]:
    """Simulate a realistic loss curve for mock/test runs without a 24GB GPU."""
    stable_offset = int.from_bytes(hashlib.sha256(condition.encode()).digest()[:4], "big")
    rng = random.Random(seed + stable_offset)
    loss_curve = []
    current_loss = initial_loss

    for step in range(1, num_steps + 1):
        if condition in ("protected", "custom_protected"):
            # Unlearnable shortcut: loss plunges rapidly towards zero
            decay = 0.88 if condition == "protected" else 0.85
            current_loss = max(0.01, current_loss * decay + rng.gauss(0, 0.005))
        elif condition == "random_edits":
            # Random edits: normal descent with slight noise
            current_loss = max(0.4, current_loss * 0.96 + rng.gauss(0, 0.03))
        else:  # original
            # Clean original: steady normal descent
            current_loss = max(0.35, current_loss * 0.955 + rng.gauss(0, 0.02))

        loss_curve.append({
            "step": step,
            "loss": round(current_loss, 4),
            "epoch": round(step / (num_steps / 3), 2),
        })

    return loss_curve


def run_training(
    config_path: Path,
    condition: str,
    model_name: Optional[str] = None,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    mock: bool = False,
) -> Dict[str, Any]:
    """Execute LoRA fine-tuning or simulated mock execution."""
    cfg = load_config(config_path)
    model_name = model_name or cfg["models"]["surrogate"]
    slug = slugify_model_name(model_name)

    if output_dir is None:
        output_dir = PROJECT_ROOT / f"outputs/{slug}/{condition}/seed-{seed}"
    output_dir = Path(output_dir).resolve()

    # Resolve training data path
    if condition == "original":
        data_path = PROJECT_ROOT / "data/original/prepared/train.jsonl"
    elif condition == "random_edits":
        data_path = PROJECT_ROOT / "data/random_edits/prepared/train.jsonl"
    elif condition == "protected":
        data_path = PROJECT_ROOT / "data/protected/prepared/train.jsonl"
    elif condition == "custom_protected":
        data_path = PROJECT_ROOT / "data/custom_protected/prepared/train.jsonl"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    train_params = cfg.get("training", {})
    lora_params = cfg.get("lora", {})

    metadata = {
        "model": model_name,
        "condition": condition,
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "training_params": train_params,
        "lora_params": lora_params,
        "data_path": str(data_path),
        "mock": mock,
    }

    if mock:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MOCK] Running training simulation for {model_name} on {condition} (seed {seed})")
        loss_curve = simulate_mock_training(model_name, condition, seed)
        (output_dir / "training_loss.json").write_text(
            json.dumps(loss_curve, indent=2), encoding="utf-8"
        )
        (output_dir / "adapter_placeholder.txt").write_text(
            f"Mock LoRA adapter for {model_name} under {condition} with seed {seed}\n",
            encoding="utf-8",
        )
        metadata["total_steps"] = len(loss_curve)
        metadata["final_loss"] = loss_curve[-1]["loss"]
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"[MOCK] Saved mock run outputs to {output_dir}")
        return metadata

    metadata["dataset_provenance"] = validate_real_training_data(data_path, condition)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Real GPU training execution
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise ImportError(
            f"Required ML packages not found: {e}. "
            "On training PC run: uv sync --locked --extra experiment"
        )

    # Set seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Loading base model '{model_name}'...")
    torch_dtype = getattr(torch, train_params.get("dtype", "bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    if train_params.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_params.get("r", 16),
        lora_alpha=lora_params.get("lora_alpha", 32),
        lora_dropout=lora_params.get("lora_dropout", 0.05),
        target_modules=lora_params.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=lora_params.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load dataset
    records = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Tokenize with answer-only loss masking
    max_len = train_params.get("max_seq_length", 512)
    tokenized_data = [
        create_training_sequence(r["prompt"], r["target"], tokenizer, max_seq_length=max_len)
        for r in records
    ]

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_params.get("learning_rate", 2e-4)
    )

    epochs = train_params.get("num_train_epochs", 3)
    batch_size = train_params.get("per_device_train_batch_size", 1)
    grad_accum = train_params.get("gradient_accumulation_steps", 16)

    loss_curve = []
    step = 0
    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        random.shuffle(tokenized_data)
        for i in range(0, len(tokenized_data), batch_size):
            batch = tokenized_data[i : i + batch_size]
            # Collate batch
            b_input_ids = torch.tensor([b["input_ids"] for b in batch], device=model.device)
            b_attn = torch.tensor([b["attention_mask"] for b in batch], device=model.device)
            b_labels = torch.tensor([b["labels"] for b in batch], device=model.device)

            outputs = model(input_ids=b_input_ids, attention_mask=b_attn, labels=b_labels)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (i // batch_size + 1) % grad_accum == 0 or (i + batch_size >= len(tokenized_data)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                current_loss_val = outputs.loss.item()
                loss_curve.append({
                    "step": step,
                    "loss": round(current_loss_val, 4),
                    "epoch": round(epoch + (i / len(tokenized_data)), 2),
                })
                if step % 10 == 0:
                    print(f"Step {step} (Epoch {epoch+1}/{epochs}) - Loss: {current_loss_val:.4f}")

    # Save adapter and metadata
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    (output_dir / "training_loss.json").write_text(
        json.dumps(loss_curve, indent=2), encoding="utf-8"
    )
    metadata["total_steps"] = len(loss_curve)
    metadata["final_loss"] = loss_curve[-1]["loss"] if loss_curve else None
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Training completed. Outputs saved to {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/experiment.toml"
    )
    parser.add_argument(
        "--condition",
        choices=["original", "random_edits", "protected", "custom_protected"],
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (defaults to surrogate from config).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Simulate training without downloading weights or using a GPU.",
    )
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        condition=args.condition,
        model_name=args.model,
        seed=args.seed,
        output_dir=args.output_dir,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
