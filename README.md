# CSE 472 — Project 2

Protecting personal data from LLM training with unlearnable text.

**Status:** dataset preparation implemented. No training, protection generation, or evaluation
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

## Computer specifications and environment checks

Run the script directly to display CPU, RAM, OS, free disk space, and NVIDIA
GPU/driver information even when PyTorch is not installed:

```powershell
python src/cse472_project2/check.py
```

The default output is human-readable. Add `--json` for structured output, or
`--require-training` to return a failing exit status when training prerequisites
are missing. NVIDIA hardware detection uses `nvidia-smi`; PyTorch CUDA and native
bf16 support are checked separately. An unavailable hardware query is not proof
that the machine has no GPU. Free disk space is for the current working directory.

The displayed interpreter path identifies the Python installation being checked.
If VS Code uses Python 3.14 or reports missing packages, select the project's
`.venv` interpreter (Python 3.12). Install the experiment dependencies on the
training PC using the commands above. The check does not verify Hugging Face
access, download models, or guarantee that a full training run fits in memory.

## Prepare ARC-Challenge

Run on the data/training PC with Python 3.12. This step downloads the dataset
from Hugging Face; it requires internet access but no GPU or model weights.

```powershell
uv sync --locked --extra data
uv run --locked --extra data project-prepare-data
```

You can also run `python src/cse472_project2/data.py` with those dependencies
installed in the selected interpreter. Use `--help` to see the options.

The command reads the `[dataset]` section of `configs/experiment.toml`, retains
all three official splits, and writes:

```text
data/original/
  raw/{train,validation,test}.jsonl
  prepared/{train,validation,test}.jsonl
  manifest.json
```

Raw records preserve all original fields. Prepared records add `prompt` (question
and all choices, ending at `Answer:`) and `target` (the original correct choice
label). Labels such as `1` or `2` stay numeric strings; they are not remapped to
letters. The answer key is kept outside the prompt. Evaluation code must feed
only the prompt to the model; target and answerKey are for scoring. Tokenization,
chat templates, and answer-only training masks belong to the future training code.

Validation rejects missing fields, malformed choices, invalid answer keys, empty
splits, and duplicate IDs within or across splits. No examples are shuffled,
filtered, or moved between splits. Use train for fine-tuning, validation for
development, and the clean test split for final reporting.

The manifest records the exact resolved Hub commit, dependency versions, counts,
and SHA-256 hashes. Reproduce an export with `--revision <resolved_revision>`.
Existing output directories are never overwritten; use `--output-dir <new-path>`
for another export. Default paths are relative to the repository; custom relative
paths are relative to your current working directory. Generated data stays out of Git.

Loader reference: [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets/loading).
Offline validation tests: `python -m unittest discover -s tests -v`.
