# Project checklist

## Requirements from the supplied project guideline

- Read *Unlearnable Examples: Making Personal Data Unexploitable* and
  *Make Text Unlearnable: Exploiting Effective Patterns to Protect Personal Data*.
- Checkpoint 1: explain the LLM loss, surrogate, perturbation, and rationale.
- Use ARC-Challenge train for fine-tuning and clean test for final evaluation.
- Generate protected training data with Llama-3-8B + LoRA, preserving answers.
- Include random edits with the same edit budget and a surrogate-loss curve.
- Evaluate both Llama-3-8B and Qwen3-8B: zero-shot, original, random, protected.
- Use three seeds and report mean and standard deviation; include test accuracy,
  GPT-2 perplexity, and SBERT similarity.
- Checkpoint 2: protected data and the results table.
- Write a 6–8 page report including method, results, limitations, examples,
  and a trade-off plot. The additional protection strategy is optional.

## Next implementation steps

1. Arrange a GPU with at least 24 GB VRAM and obtain Llama model access.
2. Load and validate ARC-Challenge; retain the validation split for development.
3. Decide the prompt, answer-only loss, evaluation rule, and Qwen thinking mode.
   These are open design choices, not completed implementations.
4. Run zero-shot and original-data LoRA baselines on both models.
5. Complete the design checkpoint before implementing protection and controls.
6. Run the full experiment matrix, aggregate results, and write the report.

Use identical LoRA settings across runs. Fix evaluation and hyperparameters
before final test reporting. Never select edits or settings using test answers.
The shared configuration is a proposed starting point; training and protection
code will be added in subsequent work.
