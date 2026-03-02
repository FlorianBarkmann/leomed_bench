#!/usr/bin/env bash
#SBATCH --job-name=leomed_swin_4gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
cd "${REPO_DIR}"

mkdir -p logs outputs

uv sync
uv run python -m leomed_bench.train --config configs/smoke_4gpu.yaml
