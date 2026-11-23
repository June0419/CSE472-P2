#!/bin/bash
#SBATCH --job-name=cse472-qwen-remaining
#SBATCH --account=class_cse47267488fall2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-9%2
#SBATCH --output=/home/spark319/CSE472-P2/outputs/qwen_remaining_%A_%a.log
#SBATCH --error=/home/spark319/CSE472-P2/outputs/qwen_remaining_%A_%a.err

set -euo pipefail
cd /home/spark319/CSE472-P2

# The original and protected seed-42 adapters already exist on Sol.
case "${SLURM_ARRAY_TASK_ID}" in
    0) condition=original;         seed=123 ;;
    1) condition=original;         seed=2026 ;;
    2) condition=random_edits;     seed=42 ;;
    3) condition=random_edits;     seed=123 ;;
    4) condition=random_edits;     seed=2026 ;;
    5) condition=protected;        seed=123 ;;
    6) condition=protected;        seed=2026 ;;
    7) condition=custom_protected; seed=42 ;;
    8) condition=custom_protected; seed=123 ;;
    9) condition=custom_protected; seed=2026 ;;
    *) echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

run_dir="outputs/qwen3_8b/${condition}/seed-${seed}"
echo "Training Qwen/Qwen3-8B: condition=${condition}, seed=${seed}"
.venv/bin/python src/cse472_project2/train.py \
    --model Qwen/Qwen3-8B --condition "${condition}" --seed "${seed}"

echo "Evaluating Qwen/Qwen3-8B: condition=${condition}, seed=${seed}"
.venv/bin/python src/cse472_project2/evaluate.py \
    --model Qwen/Qwen3-8B --condition "${condition}" --seed "${seed}" \
    --output-file "${run_dir}/evaluation_results.json"
