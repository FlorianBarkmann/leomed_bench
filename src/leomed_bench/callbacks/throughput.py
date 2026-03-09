from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import cast

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
        del outputs, batch_idx
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._elapsed_seconds += time.perf_counter() - self._batch_start_time
        inputs = self._extract_inputs_tensor(batch)
        if inputs is None:
            return
        self._seen_samples += int(inputs.size(0))
        if self._elapsed_seconds <= 0.0:
            return

        world_size = self._world_size(trainer)
        images_per_second = (self._seen_samples / self._elapsed_seconds) * world_size
        milliseconds_per_step = (
            self._elapsed_seconds / max(trainer.global_step, 1)
        ) * 1000.0
        pl_module.log(
            "train/throughput_images_per_sec",
            images_per_second,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            logger=True,
        )
        pl_module.log(
            "train/step_time_ms",
            milliseconds_per_step,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            logger=True,
        )

    def _extract_inputs_tensor(self, batch: object) -> Tensor | None:
        if isinstance(batch, Tensor):
            return batch

        if isinstance(batch, Mapping):
            batch_mapping = cast(Mapping[str, object], batch)
            for key in ("inputs", "input", "x", "image", "images"):
                value = batch_mapping.get(key)
                if isinstance(value, Tensor):
                    return value
            return None

        if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
            if len(batch) == 0:
                return None
            first_item = batch[0]
            if isinstance(first_item, Tensor):
                return first_item
            return None

        return None

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        del pl_module
        if self._elapsed_seconds <= 0.0:
            return
        world_size = self._world_size(trainer)
        images_per_second = (self._seen_samples / self._elapsed_seconds) * world_size
        milliseconds_per_step = (
            self._elapsed_seconds / max(trainer.num_training_batches, 1)
        ) * 1000.0
        trainer.callback_metrics["train/throughput_images_per_sec"] = torch.tensor(
            images_per_second
        )
        trainer.callback_metrics["train/step_time_ms"] = torch.tensor(
            milliseconds_per_step
        )

    def _world_size(self, trainer: L.Trainer) -> int:
        world_size = getattr(trainer, "world_size", 1)
        if not isinstance(world_size, int):
            return 1
        return max(world_size, 1)
