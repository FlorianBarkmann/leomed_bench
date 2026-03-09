#!/usr/bin/env bash
#SBATCH --job-name=leomed_swin_1gpu_ctr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --mem=64GB



IMAGE_URI="${IMAGE_URI:-/cluster/customapps/biomed/boeva/fbarkmann/leomed_bench_latest.sif}"

singularity exec \
  --nv \
  --bind "$PWD:/workspace" \
  --pwd /workspace \
  "${IMAGE_URI}" \
  python -m leomed_bench.train --config configs/cifar10_1gpu.yaml
