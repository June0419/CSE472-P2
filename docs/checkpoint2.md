# Checkpoint 2: Experimental Status

**Project:** CSE 472 Project 2.5 — Protecting Personal Data from LLM Training
**Dataset:** AI2 ARC-Challenge (`allenai/ai2_arc`)  
**Models planned:** Qwen3-8B and Meta-Llama-3-8B

## Prepared Data and Conditions

The pipeline prepares original, random-edit, label-hint protected, and contextual shortcut protected training variants. Validation and test examples remain clean. The code contains checks to preserve answer keys and choice labels and to compare edit counts. These are implementation checks, not proof of experimental effectiveness.

## Measured Result

The Sol output file `/home/spark319/CSE472-P2/outputs/qwen3_8b/original_vs_protected_seed42_evaluation.json` reports a real (`mock: false`) Qwen3-8B comparison at seed 42:

| Condition | Clean-test accuracy | GPT-2 perplexity | SBERT similarity |
| :--- | ---: | ---: | ---: |
| Original | 91.38% | 52.75 | 1.0000 |
| Protected (label hints) | 63.31% | 74.43 | 0.9652 |

The accuracy difference is 28.07 percentage points. With only one seed and no completed random-edit control, this is preliminary.

## Remaining Work

A Sol array job has been submitted for the remaining Qwen conditions and seeds. Its outputs need to be inspected before reporting three-seed summaries. Llama-3-8B access is pending, so there are no Llama results. See `docs/report.md` for methodology and reporting limits. No unmeasured values should be presented as experimental findings.
