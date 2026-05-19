import random
from dataclasses import dataclass, field
from functools import partial

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..utils.system_utils.config import instantiate_from_config, parse_structured
from ..utils.typing import *
from .utils import safe_dataloader


class SparseStructureDataModule(pl.LightningDataModule):
    @dataclass
    class Config:
        dataset: dict = field(default_factory=dict)
        custom_batch_sampler: Optional[dict] = None
        shuffle_train: bool = True

        # Preprocess Qwen inputs in collate_fn.
        pretrained_qwen_processor_name_or_path: Optional[str] = None
        qwen_prompt_max_length: int = 1024
        qwen_prompt_template_start_idx: int = 64
        qwen_prompt_template: str = (
            "<|im_start|>system\nYou are a helpful and creative assistant for generating 3D models. "
            "Your task is to analyze the user's input, which may include text, an image, or both. "
            "Your goal is to provide a comprehensive, detailed, and imaginative description for "
            "creating a 3D asset.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        cond_drop_prob: float = 0.1
        cond_drop_type: str = "empty"

        batch_size: int = 1
        eval_batch_size: int = 1
        num_workers: int = 16

    cfg: Config

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, kwargs)

        if self.cfg.pretrained_qwen_processor_name_or_path is not None:
            from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor

            self.qwen_processor = Qwen2_5_VLProcessor.from_pretrained(
                self.cfg.pretrained_qwen_processor_name_or_path
            )
        else:
            self.qwen_processor = None

    def setup(self, stage=None) -> None:
        train_datasets, val_datasets, test_datasets = [], [], []

        self.dataset_name_to_id = {
            dataset_name: idx for idx, dataset_name in enumerate(self.cfg.dataset.keys())
        }

        for dataset_name in self.cfg.dataset:
            data_cfg = self.cfg.dataset[dataset_name]

            if "params" not in data_cfg:
                data_cfg["params"] = {}
            data_cfg["params"]["dataset_id"] = self.dataset_name_to_id[dataset_name]

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
        batch_collated = torch.utils.data.default_collate(batch)

        if self.qwen_processor is not None:
            if "prompt" in batch_collated:
                prompts = list(batch_collated["prompt"])
            else:
                prompts = [""] * len(batch)

            if "image" in batch_collated:
                for i, prompt in enumerate(prompts):
                    prompts[i] = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
                image_inputs = batch_collated["image"] * 255
            else:
                image_inputs = None

            if split == "train" and self.cfg.cond_drop_type == "empty":
                for i, _prompt in enumerate(prompts):
                    if random.random() < self.cfg.cond_drop_prob and image_inputs is None:
                        prompts[i] = " "

            batch_collated["cond_drop_type"] = self.cfg.cond_drop_type
            batch_collated["cond_drop_prob"] = self.cfg.cond_drop_prob

            text_inputs = [
                self.cfg.qwen_prompt_template.format(prompt) for prompt in prompts
            ]
            features = self.qwen_processor(
                text=text_inputs,
                images=image_inputs,
                max_length=(
                    self.cfg.qwen_prompt_max_length
                    + self.cfg.qwen_prompt_template_start_idx
                ),
                padding=True,
                padding_side="left",
                truncation=True,
                return_tensors="pt",
            )
            batch_collated["llm_features"] = features
            batch_collated["cond_drop_idx"] = self.cfg.qwen_prompt_template_start_idx

        return batch_collated

    @safe_dataloader("train_dataset")
    def train_dataloader(self) -> DataLoader:
        if self.cfg.custom_batch_sampler is None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=partial(self.collate_fn, split="train"),
                shuffle=self.cfg.shuffle_train,
            )

        distributed_sampler = None
        if hasattr(self.trainer, "world_size") and self.trainer.world_size > 1:
            from torch.utils.data.distributed import DistributedSampler

            distributed_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=self.cfg.shuffle_train,
            )

        trainer_sampler = instantiate_from_config(
            self.cfg.custom_batch_sampler,
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle_train,
            drop_last=False,
            sampler=distributed_sampler,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=trainer_sampler,
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


__all__ = ["SparseStructureDataModule"]
