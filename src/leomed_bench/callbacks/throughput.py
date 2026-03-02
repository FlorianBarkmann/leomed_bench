from __future__ import annotations

import time

import lightning as L
import torch
from torch import Tensor


class ThroughputCallback(L.Callback):
    def __init__(self) -> None:
        super().__init__()
        self._batch_start_time: float = 0.0
        self._elapsed_seconds: float = 0.0
        self._seen_samples: int = 0

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._elapsed_seconds = 0.0
        self._seen_samples = 0

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch_idx
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._elapsed_seconds += time.perf_counter() - self._batch_start_time
        if not isinstance(batch, tuple) or len(batch) != 2:
            msg = "Expected batch to be a tuple of (inputs, targets)."
            raise TypeError(msg)
        inputs, _targets = batch
        if not isinstance(inputs, Tensor):
            msg = "Expected inputs tensor in train batch."
            raise TypeError(msg)
        self._seen_samples += int(inputs.size(0))

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        del pl_module
        if self._elapsed_seconds <= 0.0:
            return
        images_per_second = self._seen_samples / self._elapsed_seconds
        milliseconds_per_step = (
            self._elapsed_seconds / max(trainer.num_training_batches, 1)
        ) * 1000.0
        trainer.callback_metrics["train/throughput_images_per_sec"] = torch.tensor(
            images_per_second
        )
        trainer.callback_metrics["train/step_time_ms"] = torch.tensor(
            milliseconds_per_step
        )
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "train/throughput_images_per_sec": images_per_second,
                    "train/step_time_ms": milliseconds_per_step,
                },
                step=trainer.global_step,
            )
