from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict, cast

import yaml

PrecisionMode: TypeAlias = Literal[
    "64-true",
    "32-true",
    "16-mixed",
    "bf16-mixed",
]


class DataConfigRaw(TypedDict):
    data_root: str
    batch_size: int
    num_workers: int
    image_size: int


class OptimizerConfigRaw(TypedDict):
    lr: float
    weight_decay: float


class RuntimeConfigRaw(TypedDict):
    seed: int
    accelerator: str
    devices: int
    num_nodes: int
    strategy: str
    precision: PrecisionMode
    max_epochs: int
    max_steps: int
    log_every_n_steps: int
    benchmark: bool
    deterministic: bool
    output_dir: str
    run_name: str


class ModelConfigRaw(TypedDict):
    name: str
    num_classes: int


class TrainConfigRaw(TypedDict):
    data: DataConfigRaw
    optimizer: OptimizerConfigRaw
    runtime: RuntimeConfigRaw
    model: ModelConfigRaw


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    batch_size: int
    num_workers: int
    image_size: int


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    accelerator: str
    devices: int
    num_nodes: int
    strategy: str
    precision: PrecisionMode
    max_epochs: int
    max_steps: int
    log_every_n_steps: int
    benchmark: bool
    deterministic: bool
    output_dir: Path
    run_name: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int


@dataclass(frozen=True)
class TrainConfig:
    data: DataConfig
    optimizer: OptimizerConfig
    runtime: RuntimeConfig
    model: ModelConfig


def load_config(config_path: Path) -> TrainConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        msg = f"Config file '{config_path}' must contain a mapping at top level."
        raise ValueError(msg)
    required_keys = {"data", "optimizer", "runtime", "model"}
    if not required_keys.issubset(set(loaded.keys())):
        msg = (
            f"Config file '{config_path}' must define keys: "
            f"{', '.join(sorted(required_keys))}."
        )
        raise ValueError(msg)
    raw_config: TrainConfigRaw = cast(TrainConfigRaw, loaded)
    data_cfg = raw_config["data"]
    optimizer_cfg = raw_config["optimizer"]
    runtime_cfg = raw_config["runtime"]
    model_cfg = raw_config["model"]

    return TrainConfig(
        data=DataConfig(
            data_root=Path(data_cfg["data_root"]),
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            image_size=data_cfg["image_size"],
        ),
        optimizer=OptimizerConfig(
            lr=optimizer_cfg["lr"],
            weight_decay=optimizer_cfg["weight_decay"],
        ),
        runtime=RuntimeConfig(
            seed=runtime_cfg["seed"],
            accelerator=runtime_cfg["accelerator"],
            devices=runtime_cfg["devices"],
            num_nodes=runtime_cfg["num_nodes"],
            strategy=runtime_cfg["strategy"],
            precision=runtime_cfg["precision"],
            max_epochs=runtime_cfg["max_epochs"],
            max_steps=runtime_cfg["max_steps"],
            log_every_n_steps=runtime_cfg["log_every_n_steps"],
            benchmark=runtime_cfg["benchmark"],
            deterministic=runtime_cfg["deterministic"],
            output_dir=Path(runtime_cfg["output_dir"]),
            run_name=runtime_cfg["run_name"],
        ),
        model=ModelConfig(
            name=model_cfg["name"],
            num_classes=model_cfg["num_classes"],
        ),
    )
