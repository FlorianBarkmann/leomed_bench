#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "Submitting 1-GPU container benchmark..."
job_1gpu="$(sbatch slurm/benchmark_1gpu_container.sbatch | awk '{print $4}')"
echo "1-GPU job id: ${job_1gpu}"

echo "Submitting 4-GPU container benchmark..."
job_4gpu="$(sbatch slurm/benchmark_4gpu_container.sbatch | awk '{print $4}')"
echo "4-GPU job id: ${job_4gpu}"

echo "Submitted both container benchmarks. Monitor with:"
echo "  squeue -j ${job_1gpu},${job_4gpu}"
