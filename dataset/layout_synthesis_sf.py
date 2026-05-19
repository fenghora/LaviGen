import json
import os
import random
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.system_utils.config import parse_structured
from ..utils.typing import *


@dataclass
class LayoutSynthesisSFDataModuleConfig:
    data_file: str = ""
    train_indices: Optional[Tuple[Any, Any]] = None
    val_indices: Optional[Tuple[Any, Any]] = None
    test_indices: Optional[Tuple[Any, Any]] = None
    repeat: int = 1
    batch_size: int = 1
    eval_batch_size: int = 1
    num_workers: int = 16
    min_objects: int = 1
    max_objects: int = 10
    
    pretrained_qwen_processor_name_or_path: Optional[str] = None
    qwen_prompt_max_length: int = 1024
    qwen_prompt_template_start_idx: int = 64
    qwen_prompt_template: str = (
        "<|im_start|>system\nYou are a helpful and creative assistant for generating 3D models. Your task is to analyze the user's input, which may include text, an image, or both. Your goal is to provide a comprehensive, detailed, and imaginative description for creating a 3D asset.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    cond_drop_prob: float = 0.1
    cond_drop_type: str = "empty"


class LayoutSynthesisSFDataset(Dataset):
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
        floor_voxels = data["floor_voxels"]
        scene_voxels = data["scene_voxels"]
        object_voxels = data["canonical_voxels"]

        paired_voxels = list(zip(range(len(scene_voxels)), scene_voxels, object_voxels))
        paired_voxels_sorted = sorted(
            paired_voxels, key=lambda pair: pair[1][:, 2].mean()
        )

        ids_sorted, scene_voxels_sorted, object_voxels_sorted = zip(*paired_voxels_sorted)
        scene_voxels_sorted = list(scene_voxels_sorted)
        object_voxels_sorted = list(object_voxels_sorted)
        ids_sorted = list(ids_sorted)

        num_objects = len(scene_voxels_sorted)
        n_start = random.randint(0, num_objects - 1)
        max_generate = min(self.cfg.max_objects, num_objects - n_start)
        num_to_generate = random.randint(
            min(self.cfg.min_objects, max_generate),
            max_generate
        )

        def points_to_dense(points, grid_size=64):
            grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
            idx = np.clip(points.astype(int), 0, grid_size - 1)
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
            return grid[None]

        if n_start > 0:
            condition_points = np.vstack([floor_voxels] + scene_voxels_sorted[:n_start])
        else:
            condition_points = floor_voxels

        condition = points_to_dense(condition_points)

        voxel_points = np.vstack([condition_points] + scene_voxels_sorted[n_start:n_start+num_to_generate])
        voxel = points_to_dense(voxel_points)

        target_objects = [points_to_dense(object_voxels_sorted[i]) for i in range(n_start, n_start+num_to_generate)]
        
        gt_conditions = []
        current_points = condition_points
        for i in range(n_start, n_start+num_to_generate):
            current_points = np.vstack([current_points, scene_voxels_sorted[i]])
            gt_conditions.append(points_to_dense(current_points))

        id_list = [int(x) for x in ids_sorted[:n_start+num_to_generate]]
        id_list = ",".join(map(str, id_list))

        return {
            "voxel": voxel,
            "condition": condition,
            "object": target_objects,
            "gt_conditions": gt_conditions,
            "num_objects": num_to_generate,
            "prompt": prompt,
            "id_list": id_list,
            "model_id": model_id
        }


class LayoutSynthesisSFValidateDataset(Dataset):
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
        floor_voxels = data["floor_voxels"]
        scene_voxels = data["scene_voxels"]
        object_voxels = data["canonical_voxels"]

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
            return grid[None]

        results = []
        condition_points = floor_voxels

        for n in range(len(scene_voxels_sorted)):
            condition_dense = points_to_dense(condition_points)
            voxel_points = np.vstack([condition_points, scene_voxels_sorted[n]])
            voxel_dense = points_to_dense(voxel_points)
            target_object_dense = points_to_dense(object_voxels_sorted[n])
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

            condition_points = voxel_points

        return results


def collate_sf_batch(batch, qwen_processor=None, cfg=None, split="train"):
    batch_collated = torch.utils.data.default_collate(batch)

    if qwen_processor is not None:
        if "prompt" in batch_collated:
            prompts = batch_collated["prompt"]
        else:
            prompts = [""] * len(batch)

        if split == "train" and cfg.cond_drop_type == "empty":
            for i, prompt in enumerate(prompts):
                if random.random() < cfg.cond_drop_prob:
                    prompts[i] = " "
        batch_collated["cond_drop_type"] = cfg.cond_drop_type
        batch_collated["cond_drop_prob"] = cfg.cond_drop_prob

        text_inputs = [f"{cfg.qwen_prompt_template.format(prompt)}" for prompt in prompts]
        features = qwen_processor(
            text=text_inputs,
            images=None,
            max_length=cfg.qwen_prompt_max_length + cfg.qwen_prompt_template_start_idx,
            padding=True,
            padding_side="left",
            truncation=True,
            return_tensors="pt",
        )
        batch_collated["llm_features"] = features
        batch_collated["cond_drop_idx"] = cfg.qwen_prompt_template_start_idx

    return batch_collated


class LayoutSynthesisSFDataModule(pl.LightningDataModule):
    cfg: LayoutSynthesisSFDataModuleConfig

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cfg = parse_structured(LayoutSynthesisSFDataModuleConfig, {**kwargs})
        
        if self.cfg.pretrained_qwen_processor_name_or_path is not None:
            from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor

            self.qwen_processor = Qwen2_5_VLProcessor.from_pretrained(
                self.cfg.pretrained_qwen_processor_name_or_path
            )
        else:
            self.qwen_processor = None

    def setup(self, stage=None) -> None:
        print(self.cfg)
        if stage in [None, "fit"]:
            self.train_dataset = LayoutSynthesisSFDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = LayoutSynthesisSFValidateDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = LayoutSynthesisSFValidateDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=partial(collate_sf_batch, qwen_processor=self.qwen_processor, cfg=self.cfg, split="train"),
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
