# Leomed Swin Benchmark (1 GPU vs 4 GPUs)

This repository benchmarks multi-GPU scaling on Leomed by training a Swin
Transformer (`swin_t`) on CIFAR-10 or ImageNet with PyTorch Lightning under
SLURM.

The primary comparison is:

- single GPU run (`configs/cifar10_1gpu.yaml`)
- four GPU DDP run (`configs/cifar10_4gpu.yaml`)

Both use the same workload shape and collect throughput + step-time metrics.

## Prerequisites

- Python 3.12+
- `uv` installed
- SLURM access on Leomed
- Optional: Docker/Singularity for containerized dependencies

## Project Setup

```bash
uv sync
```

## Containerized Dependencies (Recommended for HPC)

If `uv` is unreliable on your cluster nodes, use the provided `Dockerfile` and run
the benchmark from a container image instead of calling `uv` inside SLURM jobs.

Build locally:

```bash
docker build -t leomed-bench:latest .
```

Quick local container run:

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  -w /workspace \
  leomed-bench:latest \
  --config configs/cifar10_1gpu.yaml
```

On Leomed (Singularity), point to a Docker image URI and use the container SLURM
scripts:

```bash
# Uses default local SIF image path:
# /cluster/customapps/biomed/boeva/fbarkmann/leomed_bench_latest.sif
sbatch slurm/benchmark_1gpu_container.sbatch
sbatch slurm/benchmark_4gpu_container.sbatch
```

Optional override:

```bash
export IMAGE_URI="docker://fbarkmann/leomed_bench:latest"
```

## Dataset Options

`data.dataset` controls which dataset is used:

- `cifar10` (default if omitted in config)
- `imagenet` (larger dataset)

CIFAR-10 is downloaded automatically by `torchvision` the first time you run.
For ImageNet, set `data.data_root` to a directory containing:

- `train/` (class subfolders)
- `val/` (class subfolders)

Example ImageNet configs are provided:

- `configs/imagenet_1gpu.yaml`
- `configs/imagenet_4gpu.yaml`

## Local Dry Run (Single GPU)

```bash
uv run python -m leomed_bench.train --config configs/cifar10_1gpu.yaml
```

Run larger dataset benchmark:

```bash
uv run python -m leomed_bench.train --config configs/imagenet_1gpu.yaml
```

## Configure SLURM Placeholders

Edit both SLURM scripts before submitting:

- `slurm/benchmark_1gpu.sbatch`
- `slurm/benchmark_4gpu.sbatch`
- or containerized variants:
  - `slurm/benchmark_1gpu_container.sbatch`
  - `slurm/benchmark_4gpu_container.sbatch`

Fill in:

- `#SBATCH --account=YOUR_ACCOUNT`
- `#SBATCH --partition=YOUR_PARTITION`
- `#SBATCH --gres=gpu:YOUR_GPU_TYPE:1` in 1-GPU jobs
- `#SBATCH --gres=gpu:YOUR_GPU_TYPE:4` in 4-GPU jobs
- optional time/cpu adjustments for your environment

## Submit Benchmarks

Submit both jobs together:

```bash
bash scripts/submit_benchmarks.sh
```

Or individually:

```bash
sbatch slurm/benchmark_1gpu.sbatch
sbatch slurm/benchmark_4gpu.sbatch
```

## Collect Results

Each run writes Lightning CSV logs in:

- `outputs/cifar10_1gpu/metrics.csv`
- `outputs/cifar10_4gpu/metrics.csv`

Then compute scaling:

```bash
uv run python scripts/collect_results.py \
  --run-1gpu outputs/cifar10_1gpu \
  --run-4gpu outputs/cifar10_4gpu
```

Output includes:

- throughput (images/sec) for 1 and 4 GPUs
- step time (ms) for 1 and 4 GPUs
- scaling factor (`throughput_4gpu / throughput_1gpu`)
- parallel efficiency (`scaling / 4`)

## Notes

- This setup is intentionally a lightweight CIFAR-10 benchmark (`max_epochs: 1`) to compare
  scaling quickly.
- For a longer benchmark, increase `max_epochs` consistently in both config
  files.
