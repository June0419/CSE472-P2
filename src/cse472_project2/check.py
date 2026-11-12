"""Inspect the environment without downloading models or requiring ML packages."""

import argparse
import importlib.metadata
import json
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-training", action="store_true",
                        help="Exit unsuccessfully unless training prerequisites are met.")
    args = parser.parse_args()
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "datasets",
                 "sentence-transformers", "numpy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpus = []
    torch_error = None
    if packages["torch"]:
        try:
            import torch
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                with torch.cuda.device(index):
                    bf16 = torch.cuda.is_bf16_supported()
                gpus.append({"name": props.name,
                             "vram_gib": round(props.total_memory / 1024**3, 2),
                             "bf16": bf16})
        except (ImportError, OSError, RuntimeError) as error:
            torch_error = str(error)
    ready = all(packages.values()) and any(
        gpu["vram_gib"] >= 23 and gpu["bf16"] for gpu in gpus
    )  # Nominal 24 GB cards can expose slightly less usable memory.
    print(json.dumps({"python": platform.python_version(), "packages": packages,
                      "gpus": gpus, "torch_error": torch_error,
                      "training_prerequisites_met": ready}, indent=2))
    if args.require_training and not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
