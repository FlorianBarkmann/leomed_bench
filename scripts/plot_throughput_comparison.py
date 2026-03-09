from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunSeries:
    name: str
    steps: list[float]
    throughput: list[float]
    total_runtime_seconds: float


@dataclass(frozen=True)
class SummaryStats:
    median: float
    mean: float
    p10: float
    p90: float


def _read_run(
    name: str,
    metrics_path: Path,
    throughput_scale: float = 1.0,
    step_divisor: float = 1.0,
) -> RunSeries:
    steps: list[float] = []
    throughput: list[float] = []
    step_indices: list[int] = []
    avg_step_times_ms: list[float] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step_raw = row.get("step")
            throughput_raw = row.get("train/throughput_images_per_sec")
            step_time_raw = row.get("train/step_time_ms")
            if step_raw is None or throughput_raw is None or step_time_raw is None:
                continue

            if throughput_raw == "" or step_time_raw == "":
                continue

            step_index = int(step_raw)
            step_value = float(step_index) / step_divisor
            throughput_value = float(throughput_raw)
            step_time_ms = float(step_time_raw)
            if not (math.isnan(throughput_value) or math.isnan(step_time_ms)):
                steps.append(step_value)
                throughput.append(throughput_value * throughput_scale)
                step_indices.append(step_index)
                avg_step_times_ms.append(step_time_ms)

    if not steps:
        msg = f"No usable throughput values found in: {metrics_path}"
        raise ValueError(msg)
    total_runtime_seconds = (step_indices[-1] + 1) * (avg_step_times_ms[-1] / 1000.0)
    return RunSeries(
        name=name,
        steps=steps,
        throughput=throughput,
        total_runtime_seconds=total_runtime_seconds,
    )


def _percentile(sorted_values: list[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        msg = f"Percentile must be in [0, 1], got {q}."
        raise ValueError(msg)
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = q * (len(sorted_values) - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return sorted_values[low]
    weight_high = idx - low
    return sorted_values[low] * (1.0 - weight_high) + sorted_values[high] * weight_high


def _summarize(throughput: list[float]) -> SummaryStats:
    sorted_values = sorted(throughput)
    median_idx = len(sorted_values) // 2
    if len(sorted_values) % 2 == 0:
        median = (sorted_values[median_idx - 1] + sorted_values[median_idx]) / 2.0
    else:
        median = sorted_values[median_idx]
    mean = sum(sorted_values) / len(sorted_values)
    p10 = _percentile(sorted_values, 0.10)
    p90 = _percentile(sorted_values, 0.90)
    return SummaryStats(
        median=median,
        mean=mean,
        p10=p10,
        p90=p90,
    )


def main() -> None:
    matplotlib.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 10,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    run_1gpu = _read_run(
        "1 GPU",
        Path("outputs_metrics/cifar10_1gpu/metrics.csv"),
        step_divisor=4.0,
    )
    run_4gpu = _read_run(
        "4 GPU", Path("outputs_metrics/cifar10_4gpu_ntask_1/metrics.csv")
    )

    stats_1gpu = _summarize(run_1gpu.throughput)
    stats_4gpu = _summarize(run_4gpu.throughput)
    scaling = stats_4gpu.median / stats_1gpu.median
    efficiency = scaling / 4.0

    colors: dict[str, str] = {
        "1 GPU": "#0072B2",
        "4 GPU": "#D55E00",
    }

    # A4 width is 8.27 inches; height chosen for a compact single-row layout.
    fig, (ax_line, ax_bar, ax_runtime) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(8.27, 3.4),
        gridspec_kw={"width_ratios": [2.0, 1.0, 1.0]},
    )

    ax_line.plot(
        run_1gpu.steps,
        run_1gpu.throughput,
        color=colors["1 GPU"],
        linewidth=1.5,
        alpha=0.9,
        label=f"1 GPU (median={stats_1gpu.median:.1f})",
    )
    ax_line.plot(
        run_4gpu.steps,
        run_4gpu.throughput,
        color=colors["4 GPU"],
        linewidth=1.5,
        alpha=0.9,
        label=f"4 GPU (median={stats_4gpu.median:.1f})",
    )
    ax_line.set_xlabel("Global step")
    ax_line.set_ylabel("Throughput (images/s)")
    ax_line.set_title("Throughput over training")
    ax_line.grid(alpha=0.2, linewidth=0.6)

    labels = ["1 GPU", "4 GPU"]
    medians = [stats_1gpu.median, stats_4gpu.median]
    p10 = [stats_1gpu.p10, stats_4gpu.p10]
    p90 = [stats_1gpu.p90, stats_4gpu.p90]
    yerr = [
        [medians[0] - p10[0], medians[1] - p10[1]],
        [p90[0] - medians[0], p90[1] - medians[1]],
    ]
    ax_bar.bar(
        labels,
        medians,
        yerr=yerr,
        capsize=4,
        color=[colors["1 GPU"], colors["4 GPU"]],
        alpha=0.9,
    )
    ax_bar.set_ylabel("Median throughput (images/s)")
    ax_bar.set_title("Median throughput (full run)")
    ax_bar.grid(axis="y", alpha=0.2, linewidth=0.6)

    runtime_values = [run_1gpu.total_runtime_seconds, run_4gpu.total_runtime_seconds]
    ax_runtime.bar(
        labels,
        runtime_values,
        color=[colors["1 GPU"], colors["4 GPU"]],
        alpha=0.9,
    )
    ax_runtime.set_ylabel("Total runtime (s)")
    ax_runtime.set_title("Total runtime (full run)")
    ax_runtime.grid(axis="y", alpha=0.2, linewidth=0.6)

    for idx, runtime_seconds in enumerate(runtime_values):
        ax_runtime.text(
            idx,
            runtime_seconds,
            f"{runtime_seconds:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    runtime_speedup = run_1gpu.total_runtime_seconds / run_4gpu.total_runtime_seconds
    fig.suptitle(
        (
            f"Throughput comparison | scaling={scaling:.2f}x"
            f" | efficiency={efficiency * 100:.1f}%"
            f" | runtime speedup={runtime_speedup:.2f}x"
        ),
        y=1.02,
    )
    ax_line.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1
    )

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_comparison_a4.svg", bbox_inches="tight")
    fig.savefig(output_dir / "throughput_comparison_a4.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
