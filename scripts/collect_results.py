from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkResult:
    throughput_images_per_sec: float
    step_time_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect 1 GPU vs 4 GPU benchmark results."
    )
    parser.add_argument(
        "--run-1gpu",
        type=Path,
        default=Path("outputs/cifar10_1gpu"),
        help="Run directory for 1-GPU run (contains metrics.csv).",
    )
    parser.add_argument(
        "--run-4gpu",
        type=Path,
        default=Path("outputs/cifar10_4gpu"),
        help="Run directory for 4-GPU run (contains metrics.csv).",
    )
    return parser.parse_args()


def read_metrics(run_dir: Path) -> BenchmarkResult:
    metrics_file = run_dir / "metrics.csv"
    if not metrics_file.exists():
        msg = f"Could not find metrics file at '{metrics_file}'."
        raise FileNotFoundError(msg)

    throughput_values: list[float] = []
    step_time_values: list[float] = []

    with metrics_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            throughput_raw = row.get("train/throughput_images_per_sec")
            if throughput_raw:
                throughput_values.append(float(throughput_raw))
            step_time_raw = row.get("train/step_time_ms")
            if step_time_raw:
                step_time_values.append(float(step_time_raw))

    if not throughput_values or not step_time_values:
        msg = (
            f"Metrics in '{metrics_file}' do not contain throughput "
            "and step time values."
        )
        raise ValueError(msg)

    return BenchmarkResult(
        throughput_images_per_sec=throughput_values[-1],
        step_time_ms=step_time_values[-1],
    )


def main() -> None:
    args = parse_args()
    one_gpu = read_metrics(args.run_1gpu)
    four_gpu = read_metrics(args.run_4gpu)
    scaling = four_gpu.throughput_images_per_sec / one_gpu.throughput_images_per_sec
    efficiency = scaling / 4.0

    print("Benchmark summary")
    print("-----------------")
    print(f"1 GPU throughput     : {one_gpu.throughput_images_per_sec:.2f} images/sec")
    print(f"4 GPU throughput     : {four_gpu.throughput_images_per_sec:.2f} images/sec")
    print(f"1 GPU step time      : {one_gpu.step_time_ms:.2f} ms")
    print(f"4 GPU step time      : {four_gpu.step_time_ms:.2f} ms")
    print(f"Scaling (4 / 1)      : {scaling:.3f}x")
    print(f"Parallel efficiency  : {efficiency:.3f}")


if __name__ == "__main__":
    main()
