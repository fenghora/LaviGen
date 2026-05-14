import json
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.system_utils.config import parse_structured
from ..utils.typing import *


@dataclass
class LayoutSynthesisDataModuleConfig:
    data_file: str = "" # jsonl: {"model", "prompt"} 

    # For training
    train_indices: Optional[Tuple[Any, Any]] = None
    val_indices: Optional[Tuple[Any, Any]] = None
    test_indices: Optional[Tuple[Any, Any]] = None

    repeat: int = 1

    batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 16


@dataclass
class LayoutSynthesisValidateDataModuleConfig:
    data_file: str = "" # jsonl: {"model", "prompt"} 

    # For training
    train_indices: Optional[Tuple[Any, Any]] = None
    val_indices: Optional[Tuple[Any, Any]] = None
    test_indices: Optional[Tuple[Any, Any]] = None

    repeat: int = 1

    batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 16


class LayoutSynthesisDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train", dataset_id: int = 0) -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.dataset_id = dataset_id
        self.split = split
        self.cfg = cfg

        self.samples = []
        with open(self.cfg.data_file, "r") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        self.indices = (0, len(self.samples))
        if self.split == "train" and self.cfg.train_indices is not None:
            start, end = self.cfg.train_indices
            self.indices = (
                0 if start is None else start,
                len(self.samples) if end is None else end,
            )
        elif self.split == "val" and self.cfg.val_indices is not None:
            start, end = self.cfg.val_indices
            self.indices = (
                0 if start is None else start,
                len(self.samples) if end is None else end,
            )
        elif self.split == "test" and self.cfg.test_indices is not None:
            start, end = self.cfg.test_indices
            self.indices = (
                0 if start is None else start,
                len(self.samples) if end is None else end,
            )

        self.samples = self.samples[self.indices[0] : self.indices[1]] * self.cfg.repeat

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        model_id = sample["model"]
        prompt = sample["prompt"]

        data = np.load(model_id, allow_pickle=True)
        floor_voxels = data["floor_voxels"]      # floor
        scene_voxels = data["scene_voxels"]      # list of objects in scene
        object_voxels = data["canonical_voxels"] # list of objects

        paired_voxels = list(zip(range(len(scene_voxels)), scene_voxels, object_voxels))
        paired_voxels_sorted = sorted(
            paired_voxels, key=lambda pair: pair[1][:, 2].mean()
        )

        ids_sorted, scene_voxels_sorted, object_voxels_sorted = zip(*paired_voxels_sorted)
        scene_voxels_sorted = list(scene_voxels_sorted)
        object_voxels_sorted = list(object_voxels_sorted)
        ids_sorted = list(ids_sorted)

        num_objects = len(scene_voxels_sorted)
        n = random.randint(0, num_objects - 1)  # 0 <= n < num_objects

        # 构造 condition (floor + 前 n 个物体)
        if n > 0:
            condition = np.vstack([floor_voxels] + scene_voxels_sorted[:n])
        else:
            condition = floor_voxels

        # 构造 voxel (floor + 第 n 个物体)
        voxel = np.vstack([condition, scene_voxels_sorted[n]])
        target_object = object_voxels_sorted[n]  # 对应的 object_voxel

        # 对应的 id_list
        id_list = ids_sorted[:n+1]
        id_list = list(map(int, id_list))
        id_list = ",".join(map(str, id_list))  # 存成字符串 "3,2,1"

        def points_to_dense(points, grid_size=64):
            grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
            idx = np.clip(points.astype(int), 0, grid_size - 1)
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
            return grid[None]  # (1, 64, 64, 64)

        condition = points_to_dense(condition)
        voxel = points_to_dense(voxel)
        target_object = points_to_dense(target_object)

        return {
            "voxel": voxel,
            "condition": condition,
            "object": target_object,
            "prompt": prompt,
            "id_list": id_list,
            "model_id": model_id
        }
        
        
class LayoutSynthesisValidateDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg = cfg

        self.samples = []
        with open(self.cfg.data_file, "r") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        model_id = sample["model"]
        prompt = sample["prompt"]

        data = np.load(model_id, allow_pickle=True)
        floor_voxels = data["floor_voxels"]      # floor
        scene_voxels = data["scene_voxels"]      # list of objects in scene
        object_voxels = data["canonical_voxels"] # list of objects

        # 排序：按 z 坐标平均值
        paired_voxels = list(zip(range(len(scene_voxels)), scene_voxels, object_voxels))
        paired_voxels_sorted = sorted(
            paired_voxels, key=lambda pair: pair[1][:, 2].mean()
        )

        ids_sorted, scene_voxels_sorted, object_voxels_sorted = zip(*paired_voxels_sorted)
        scene_voxels_sorted = list(scene_voxels_sorted)
        object_voxels_sorted = list(object_voxels_sorted)
        ids_sorted = list(ids_sorted)

        def points_to_dense(points, grid_size=64):
            grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
            idx = np.clip(points.astype(int), 0, grid_size - 1)
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
            return grid[None]  # (1, 64, 64, 64)

        results = []
        condition_points = floor_voxels

        for n in range(len(scene_voxels_sorted)):
            # 构造 condition (floor + 前 n 个物体)
            condition_dense = points_to_dense(condition_points)

            # 构造 voxel (condition + 第 n 个物体)
            voxel_points = np.vstack([condition_points, scene_voxels_sorted[n]])
            voxel_dense = points_to_dense(voxel_points)

            # 构造 target_object
            target_object_dense = points_to_dense(object_voxels_sorted[n])

            # 构造 id_list (floor + 前 n 个物体的索引)
            id_list = ids_sorted[:n+1]
            id_list = ",".join(map(str, map(int, id_list)))

            results.append({
                "voxel": voxel_dense,
                "condition": condition_dense,
                "object": target_object_dense,
                "prompt": prompt,
                "id_list": id_list,
                "model_id": model_id
            })

            # 更新 condition_points，加上当前物体
            condition_points = voxel_points

        return results


class LayoutSynthesisDataModule(pl.LightningDataModule):
    cfg: LayoutSynthesisDataModuleConfig

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cfg = parse_structured(LayoutSynthesisDataModuleConfig, {**kwargs})

    def setup(self, stage=None) -> None:
        print(self.cfg)
        if stage in [None, "fit"]:
            self.train_dataset = LayoutSynthesisDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = LayoutSynthesisDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = LayoutSynthesisDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
    
    
class LayoutSynthesisValidateDataModule(pl.LightningDataModule):
    cfg: LayoutSynthesisValidateDataModuleConfig

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cfg = parse_structured(LayoutSynthesisValidateDataModuleConfig, {**kwargs})

    def setup(self, stage=None) -> None:
        print(self.cfg)
        if stage in [None, "fit"]:
            self.train_dataset = LayoutSynthesisValidateDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = LayoutSynthesisValidateDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = LayoutSynthesisValidateDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
