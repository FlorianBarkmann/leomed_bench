from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Cifar10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        batch_size: int,
        num_workers: int,
        image_size: int,
    ) -> None:
        super().__init__()
        self._data_root: Path = data_root
        self._batch_size: int = batch_size
        self._num_workers: int = num_workers
        self._image_size: int = image_size
        self._train_dataset: datasets.CIFAR10 | None = None
        self._val_dataset: datasets.CIFAR10 | None = None

    def setup(self, stage: str | None = None) -> None:
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self._image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        val_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self._image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        if stage in (None, "fit"):
            self._train_dataset = datasets.CIFAR10(
                root=str(self._data_root),
                train=True,
                download=True,
                transform=train_transforms,
            )
            self._val_dataset = datasets.CIFAR10(
                root=str(self._data_root),
                train=False,
                download=True,
                transform=val_transforms,
            )

    def train_dataloader(self) -> DataLoader[datasets.CIFAR10]:
        if self._train_dataset is None:
            msg = "CIFAR-10 train dataset is not initialized. Call setup('fit') first."
            raise RuntimeError(msg)
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=self._num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[datasets.CIFAR10]:
        if self._val_dataset is None:
            msg = "CIFAR-10 val dataset is not initialized. Call setup('fit') first."
            raise RuntimeError(msg)
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self._num_workers > 0,
        )
