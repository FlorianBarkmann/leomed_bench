from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from leomed_bench.callbacks.throughput import ThroughputCallback
from leomed_bench.config import TrainConfig, load_config
from leomed_bench.data.imagenet import build_data_module
from leomed_bench.models.swin_module import SwinLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Swin benchmark on CIFAR-10 or ImageNet via SLURM."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def build_trainer(config: TrainConfig) -> L.Trainer:
    csv_logger = CSVLogger(
        save_dir=str(config.runtime.output_dir),
        name="",
        version=config.runtime.run_name,
    )
    return L.Trainer(
        accelerator=config.runtime.accelerator,
        devices=config.runtime.devices,
        num_nodes=config.runtime.num_nodes,
        strategy=config.runtime.strategy,
        precision=config.runtime.precision,
        max_epochs=config.runtime.max_epochs,
        log_every_n_steps=config.runtime.log_every_n_steps,
        benchmark=config.runtime.benchmark,
        deterministic=config.runtime.deterministic,
        logger=csv_logger,
        callbacks=[ThroughputCallback()],
    )


def run(config: TrainConfig) -> None:
    L.seed_everything(config.runtime.seed, workers=True)
    data_module = build_data_module(
        dataset=config.data.dataset,
        data_root=config.data.data_root,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
    )
    model = SwinLightningModule(
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    trainer = build_trainer(config)
    trainer.fit(model=model, datamodule=data_module)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
