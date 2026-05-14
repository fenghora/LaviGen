import json
import os
import random
from dataclasses import dataclass, field
from functools import partial

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from ..utils.system_utils.config import (
    instantiate_from_config,
    parse_structured,
)
from ..utils.system_utils.logging import *
from ..utils.typing import *

from .utils import SingleDatasetBatchSampler, safe_dataloader


class BaseDataModule(pl.LightningDataModule):
    @dataclass
    class Config:
        dataset: dict = field(default_factory=dict)
        custom_batch_sampler: Optional[str] = None
        shuffle_train: bool = True

        batch_size: int = 1
        eval_batch_size: int = 1
        num_workers: int = 16

    cfg: Config

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, kwargs)

    def setup(self, stage=None) -> None:
        train_datasets, val_datasets, test_datasets = [], [], []

        for dataset_name in self.cfg.dataset:
            data_cfg = self.cfg.dataset[dataset_name]

            if "params" not in data_cfg:
                data_cfg["params"] = {}

            if stage in [None, "fit"]:
                train_datasets.append(instantiate_from_config(data_cfg, split="train"))
            if stage in [None, "fit", "validate"]:
                val_datasets.append(instantiate_from_config(data_cfg, split="val"))
            if stage in [None, "test", "predict"]:
                test_datasets.append(instantiate_from_config(data_cfg, split="test"))

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        if len(train_datasets) > 0:
            self.train_dataset = ConcatDataset(train_datasets)
        if len(val_datasets) > 0:
            self.val_dataset = ConcatDataset(val_datasets)
        if len(test_datasets) > 0:
            self.test_dataset = ConcatDataset(test_datasets)

    def prepare_data(self):
        pass

    def collate_fn(self, batch, split="train"):
        batch = torch.utils.data.default_collate(batch)
        return batch

    @safe_dataloader("train_dataset")
    def train_dataloader(self) -> DataLoader:
        if not self.cfg.custom_batch_sampler:
            return DataLoader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=partial(self.collate_fn, split="train"),
                shuffle=self.cfg.shuffle_train,
            )

        # Create DistributedSampler (if using DDP)
        distributed_sampler = None
        if hasattr(self.trainer, "world_size") and self.trainer.world_size > 1:
            from torch.utils.data.distributed import DistributedSampler

            distributed_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=self.cfg.shuffle_train,
            )

        train_sampler = SingleDatasetBatchSampler(
            self.train_dataset,
            self.cfg.batch_size,
            shuffle=self.cfg.shuffle_train,
            drop_last=False,
            sampler=distributed_sampler,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            num_workers=self.cfg.num_workers,
            collate_fn=partial(self.collate_fn, split="train"),
        )

    @safe_dataloader("val_dataset")
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=partial(self.collate_fn, split="val"),
            shuffle=False,
        )

    @safe_dataloader("test_dataset")
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=partial(self.collate_fn, split="test"),
            shuffle=False,
        )

    @safe_dataloader("test_dataset")
    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


__all__ = ["BaseDataModule"]
