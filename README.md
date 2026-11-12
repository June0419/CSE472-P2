# CSE 472 — Project 2

Protecting personal data from LLM training with unlearnable text.

**Status:** initial scaffold. No training, protection generation, or evaluation
has been implemented or run.

## Development PC setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/).
Run commands from this repository directory:

```powershell
uv sync --locked
uv run --locked project-check
```

This PC is used for code, documentation, and Git setup. Testing and training
will run on a separate PC. This lightweight environment supports development
without downloading model weights or the ML stack.

## Training environment (24 GB or larger GPU)

On the separate testing/training PC, clone the repository and enter its directory
before installing the experiment dependencies. The assignment specifies at least
24 GB VRAM for bf16 8B-model LoRA training.

```bash
uv sync --locked --extra experiment
uv run --locked --extra experiment project-check --require-training
uv run --locked --extra experiment hf auth login
```

PyTorch must support your GPU and installed driver. If CUDA is unavailable,
follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for the target machine before proceeding. Passing the check verifies basic
prerequisites, not that every sequence length or batch size will fit.

Request access to [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
and authenticate on the training machine. Keep tokens out of source files.
The other victim is [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), and data is
[ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc).
No models or datasets are automatically downloaded by this setup.

## Layout

- `configs/experiment.toml`: proposed shared model, dataset, seed, and LoRA settings.
- `src/cse472_project2/`: Python package and environment checker.
- `data/`: original, protected, and random-edit data (ignored by Git).
- `models/`: adapters and checkpoints (ignored by Git).
- `outputs/`: run logs and metrics (ignored by Git).
- `docs/plan.md`: requirements and next steps.

The experiment configuration is not yet consumed by a training pipeline.
Choose and document the protection method after reading the two required papers.
Keep answer keys unchanged, evaluate on clean test data, and compare all four
conditions over three seeds. Record exact Hub revisions and GPU/CUDA details
with each run in addition to the dependency lockfile.
