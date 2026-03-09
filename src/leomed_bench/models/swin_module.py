from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics.classification import Accuracy


class SwinLightningModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if model_name != "swin_t":
            msg = (
                f"Unsupported model '{model_name}'. Only 'swin_t' is currently enabled."
            )
            raise ValueError(msg)
        self.model: torch.nn.Module = tv_models.swin_t(num_classes=num_classes)
        self._lr: float = lr
        self._weight_decay: float = weight_decay
        self.train_top1: Accuracy = Accuracy(
            task="multiclass",
            num_classes=num_classes,
            top_k=1,
        )
        self.train_top5: Accuracy = Accuracy(
            task="multiclass",
            num_classes=num_classes,
            top_k=5,
        )
        self.val_top1: Accuracy = Accuracy(
            task="multiclass",
            num_classes=num_classes,
            top_k=1,
        )
        self.val_top5: Accuracy = Accuracy(
            task="multiclass",
            num_classes=num_classes,
            top_k=5,
        )

    def forward(
        self, x: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch classes"]:
        return self.model(x)

    def training_step(
        self,
        batch: tuple[
            Float[Tensor, "batch channels height width"],
            Int[Tensor, "batch"],
        ],
        batch_idx: int,
    ) -> Tensor:
        del batch_idx
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = F.cross_entropy(logits, targets)
        # Keep torchmetrics state graph-free across iterations under DDP.
        metric_logits = logits.detach()
        metric_targets = targets.detach()
        self.train_top1(metric_logits, metric_targets)
        self.train_top5(metric_logits, metric_targets)
        self.log(
            "train/loss",
            loss.detach(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/top1",
            self.train_top1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/top5",
            self.train_top5,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self,
        batch: tuple[
            Float[Tensor, "batch channels height width"],
            Int[Tensor, "batch"],
        ],
        batch_idx: int,
    ) -> None:
        del batch_idx
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = F.cross_entropy(logits, targets)
        metric_logits = logits.detach()
        metric_targets = targets.detach()
        self.val_top1(metric_logits, metric_targets)
        self.val_top5(metric_logits, metric_targets)
        self.log(
            "val/loss",
            loss.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/top1",
            self.val_top1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/top5",
            self.val_top5,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
