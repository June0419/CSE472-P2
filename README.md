# CSE 472 — Project 2: Protecting Personal Data from LLM Training

Protecting personal user-generated text from unauthorized Large Language Model (LLM) fine-tuning using unlearnable text perturbations.

**Status:** Pipeline code and offline tests are present. Real GPU results and LoRA
adapters are stored locally under `outputs/` and are not versioned in Git. Verify
the run metadata and result artifacts before using figures or numbers in a report.

Implemented components:
- Dataset preparation and validation (`data.py`)
- Unlearnable text protection generation (`protection.py`):
  - Representative label hints (Li et al., 2023)
  - Matched-budget random edits (control baseline)
  - Contextual Shortcut Attractor (CS-Trap novel strategy)
  - Optional single-pass hint search with a trained Llama LoRA surrogate
- Causal LM LoRA PEFT training with answer-only loss masking (`train.py`)
- Downstream clean test accuracy, GPT-2 perplexity, and SBERT similarity evaluation across 3 seeds (`evaluate.py`)
- Publication-quality visualization generation (`visualize.py`)
- End-to-end pipeline orchestrator (`pipeline.py`)
- Comprehensive offline test suite (`tests/`)

---

## Directory Structure

```text
CSE472-P2/
├── configs/
│   └── experiment.toml         # Shared experiment configurations (dataset, models, LoRA, seeds)
├── data/
│   ├── mock/                   # Isolated synthetic data for --mock
│   ├── original/               # Clean original ARC-Challenge splits (raw & prepared)
│   ├── protected/              # Representative unlearnable dataset (label hints)
│   ├── random_edits/           # Matched-budget random edits control dataset
│   └── custom_protected/       # Custom Contextual Shortcut Attractor (CS-Trap) dataset
├── docs/
│   ├── checkpoint1.md          # Checkpoint 1 write-up (Loss, Surrogate, Perturbations, Rationale)
│   ├── checkpoint2.md          # Checkpoint 2 write-up (Protected data, Results table, Analysis)
│   ├── report.md               # Full academic project report (7 required sections)
│   └── plan.md                 # Project requirements and checklist
├── models/                     # Checkpoints and adapters (git-ignored)
├── outputs/
│   ├── mock/                   # Isolated simulated runs and figures
│   ├── figures/                # High-resolution generated plots (accuracy, loss curves, trade-off)
│   │   ├── accuracy_comparison.png
│   │   ├── loss_curves.png
│   │   └── tradeoff_plot.png
│   └── evaluation_results.json # Structured multi-seed aggregated evaluation metrics
├── src/
│   └── cse472_project2/
│       ├── __init__.py
│       ├── check.py            # Hardware & Python environment diagnostics
│       ├── data.py             # ARC-Challenge downloader, validator, and formatter
│       ├── evaluate.py         # Multi-metric evaluation and results aggregation
│       ├── pipeline.py         # End-to-end experiment pipeline orchestrator
│       ├── prompt.py           # Standardized prompt formatting and answer masking
│       ├── protection.py       # Unlearnable perturbation generators
│       ├── train.py            # LoRA fine-tuning with answer-only cross-entropy loss
│       └── visualize.py        # Figure generator for report and presentations
├── tests/
│   ├── test_data.py            # Tests for dataset integrity and formatting
│   ├── test_evaluate.py        # Tests for metric calculation and aggregation
│   ├── test_pipeline.py        # Tests for training simulation and figure generation
│   ├── test_prompt.py          # Tests for prompt building and answer masking
│   └── test_protection.py      # Tests for unlearnable perturbation generators
├── pyproject.toml              # Build metadata and package script entry points
├── requirements.txt            # Package dependencies list (Course guideline requirement)
└── README.md                   # This instruction manual
```

---

## Quickstart & Environment Setup

### 1. Development PC (Code, Diagnostics, and Offline Verification)

Python 3.12 is recommended. From the repository root:

```powershell
# Run hardware and environment diagnostic check
python src/cse472_project2/check.py

# Run complete offline unit test suite (PowerShell)
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests -v

# Run complete mock simulation pipeline end-to-end
python src/cse472_project2/pipeline.py --mock
```

All offline unit tests and mock simulations run immediately without requiring a GPU or downloading weights.
Mock artifacts are written under `data/mock/` and `outputs/mock/`; they must not
be used as empirical results.

---

### 2. Training Machine Setup (GPU with >= 24 GB VRAM)

For real training with `Meta-Llama-3-8B` and `Qwen3-8B` using LoRA (`bf16` precision):

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or using `uv`:
   ```bash
   uv sync --locked --extra experiment
   ```

2. **Verify GPU environment:**
   ```bash
   python src/cse472_project2/check.py --require-training
   ```

3. **Authenticate Hugging Face:**
   Request access to [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and run:
   ```bash
   huggingface-cli login
   ```

---

## Step-by-Step Execution Guide

### Step 1: Download & Prepare ARC-Challenge
Downloads the official ARC-Challenge dataset, verifies integrity, and outputs raw and prepared JSONL splits:
```bash
python src/cse472_project2/data.py
```
Outputs are written to `data/original/`.

### Step 2: Generate Unlearnable Datasets
Applies data protection to the training split while strictly preserving test split purity and answer keys:
```bash
python src/cse472_project2/protection.py --method all
```
This generates:
- `data/protected/`: Representative label hints (Li et al., 2023)
- `data/random_edits/`: Matched-budget random edits (control baseline)
- `data/custom_protected/`: Contextual Shortcut Attractor (CS-Trap novel method)

To search symbol hints with a **trained Llama LoRA adapter**, first train a
surrogate on the original data, then regenerate the protected split:

```bash
python src/cse472_project2/train.py --model meta-llama/Meta-Llama-3-8B --condition original --seed 42
python src/cse472_project2/protection.py --method surrogate_search --surrogate-adapter outputs/meta_llama_3_8b/original/seed-42/adapter --overwrite
```

This option scores candidate hints by answer-only loss under a fixed surrogate.
It is a single search pass, not alternating bilevel optimization. The same
option is available in `pipeline.py --surrogate-adapter <adapter-directory>`.

### Step 3: Train Models (LoRA PEFT)
Train the surrogate or black-box models across conditions and seeds:
```bash
# Example: Train Llama-3-8B on protected data with seed 42
python src/cse472_project2/train.py --model meta-llama/Meta-Llama-3-8B --condition protected --seed 42

# Example: Train on custom protected data
python src/cse472_project2/train.py --model meta-llama/Meta-Llama-3-8B --condition custom_protected --seed 42
```
Outputs, step-level loss curves, and adapters are saved under `outputs/<model_slug>/<condition>/seed-<seed>/`.
For a real run, `run_metadata.json` records `"mock": false` and dataset provenance.

### Step 4: Evaluate Models & Measure Text Utility
Evaluates clean test accuracy, GPT-2 perplexity, and SBERT cosine similarity across seeds:
```bash
python src/cse472_project2/evaluate.py
```
Outputs are saved to `outputs/evaluation_results.json` and printed as Markdown summary tables.

### Step 5: Generate Figures
Generates publication-quality charts for the report and presentation:
```bash
python src/cse472_project2/visualize.py
```
Figures are saved in `outputs/figures/`:
- `accuracy_comparison.png`
- `loss_curves.png`
- `tradeoff_plot.png`
The real figure command requires real evaluation results and loss files; it no
longer fills missing artifacts with simulated values.

---

## Automated Pipeline Execution

To run the entire pipeline automatically with one command:

```bash
# On a GPU machine:
python src/cse472_project2/pipeline.py

# In offline mock mode (on any development PC without GPU):
python src/cse472_project2/pipeline.py --mock
```

---

## Academic Integrity & Submission Compliance

This codebase was developed from scratch for **CSE 472 Project 2.5**:
- All unlearnable perturbation algorithms, answer-only loss tokenizers, metric aggregators, and visualization scripts are custom-implemented.
- Standard libraries (`transformers`, `peft`, `datasets`, `sentence-transformers`, `torch`, `matplotlib`) are utilized and cited.
- Deliverables include:
  1. Complete source code and offline unit test suite.
  2. `requirements.txt` resolving all dependencies.
  3. `README.md` with instructions, environment details, and execution commands.
  4. Design checkpoints in `docs/checkpoint1.md` and `docs/checkpoint2.md`.
  5. Full project report draft in `docs/report.md` conforming to all 7 required sections.
