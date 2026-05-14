import copy
import inspect
import math
import os
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import PIL
import PIL.Image
import torch
import utils3d
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from einops import rearrange
from omegaconf import OmegaConf
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

from ..models.autoencoder_kl_voxel import AutoencoderKLVoxel
from ..models.transformer_qwenvoxel_layout_synthesis import QwenVoxelTransformer3DModelLayoutSynthesis

from ..pipeline.pipeline_qwenvoxel_layout_synthesis import QwenVoxelPipelineLayoutSynthesis

from ..utils.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from ..utils.system_utils.logging import debug, info, warn
from ..utils.system_utils.misc import set_trainable_parameters
from ..utils.typing import *
from .base import BaseSystem


class QwenVoxelSystemLayoutSynthesis(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        """
        Examples:

        transformer_config:
            attention_head_dim: 128
            axes_dims_rope: [8, 40, 40, 40]
            guidance_embeds: false
            in_channels: 8
            joint_attention_dim: 2048
            num_attention_heads: 16
            num_layers: 28
            out_channels: 8
            patch_size: 1

        noise_scheduler_config:
            num_train_timesteps: 1000
            shift: 1.0
            use_dynamic_shifting: false
        """

        # Model
        base_model: str = "" 
        # pretrained_qwen_model_name_or_path: str = ""
        # pretrained_vae_model_name_or_path: str = ""
        # pretrained_adapter_model_name_or_path: Optional[str] = None

        transformer_config: Dict[str, Any] = field(default_factory=dict)
        noise_scheduler_config: Dict[str, Any] = field(default_factory=dict)

        # Training
        trainable_modules: List[str] = field(default_factory=list)
        non_trainable_modules: List[str] = field(default_factory=list)
        gradient_checkpointing: bool = False
        weighting_scheme: str = "logit_normal"

        resolution: int = 64

        # Evaluation
        eval_num_inference_steps: int = 50
        eval_seed: int = 42
        eval_guidance_scale: float = 7.0

    cfg: Config

    def configure(self):
        super().configure()

        # Get base pipeline
        # base_pipe = QwenVoxelPipeline.from_pretrained(self.cfg.base_model)

        # Prepare models
        transformer = QwenVoxelTransformer3DModelLayoutSynthesis.from_pretrained(
            self.cfg.base_model,
            subfolder="transformer",
        )
        vae = AutoencoderKLVoxel.from_pretrained(
            self.cfg.base_model,
            subfolder="vae"
        )
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.cfg.base_model,
            subfolder="scheduler"
        )
        feature_extractor = Qwen2_5_VLProcessor.from_pretrained(
            self.cfg.base_model,
            subfolder="feature_extractor"
        )
        condition_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.cfg.base_model,
            subfolder="condition_encoder"
        ) 

        # transformer: QwenVoxelTransformer3DModelLayoutSynthesis = QwenVoxelTransformer3DModelLayoutSynthesis(
        #     **OmegaConf.to_container(self.cfg.transformer_config)
        # )
        # vae: AutoencoderKLVoxel = AutoencoderKLVoxel.from_pretrained(
        #     self.cfg.pretrained_vae_model_name_or_path
        # )
        # noise_scheduler: FlowMatchEulerDiscreteScheduler = (
        #     FlowMatchEulerDiscreteScheduler(
        #         **OmegaConf.to_container(self.cfg.noise_scheduler_config)
        #     )
        # )

        # feature_extractor: Qwen2_5_VLProcessor = Qwen2_5_VLProcessor.from_pretrained(
        #     self.cfg.pretrained_qwen_model_name_or_path
        # )
        # condition_encoder: Qwen2_5_VLForConditionalGeneration = (
        #     Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #         self.cfg.pretrained_qwen_model_name_or_path
        #     )
        # )

        # Prepare pipeline
        pipeline: QwenVoxelPipelineLayoutSynthesis = QwenVoxelPipelineLayoutSynthesis(
            vae=vae,
            transformer=transformer,
            scheduler=copy.deepcopy(noise_scheduler),
            condition_encoder=condition_encoder,
            feature_extractor=feature_extractor,
        )
        pipeline.init_custom_adapter()

        # if self.cfg.pretrained_adapter_model_name_or_path:
        #     adapter_dir = os.path.dirname(
        #         self.cfg.pretrained_adapter_model_name_or_path
        #     )
        #     adapter_name = os.path.basename(
        #         self.cfg.pretrained_adapter_model_name_or_path
        #     )
        #     pipeline.load_custom_adapter(adapter_dir, adapter_name)

        # Prepare models in system
        self.register_non_module("pipeline", pipeline)  # Avoid duplicate parameters
        self.transformer = transformer
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.feature_extractor = feature_extractor
        self.condition_encoder = condition_encoder
        self.connector = pipeline.connector

        # Set transformer to training mode
        set_trainable_parameters(
            self, self.cfg.trainable_modules, self.cfg.non_trainable_modules
        )

        # Set connector to training mode
        self.vae.requires_grad_(False)
        self.condition_encoder.requires_grad_(False)
        self.connector.requires_grad_(True)

        self.vae.eval()
        self.condition_encoder.eval()
        self.transformer.train()
        self.connector.train()

        if self.cfg.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            # self.condition_encoder.gradient_checkpointing_enable()

        # Other parameters
        self.patch_size = self.non_module("pipeline").patch_size
        self.vae_scale_factor = self.non_module("pipeline").vae_scale_factor

        self._pack_latents = self.non_module("pipeline")._pack_latents
        self._unpack_latents = self.non_module("pipeline")._unpack_latents

    def on_fit_start(self):
        # Log for manual dtype casting for deepspeed
        self.my_dtype = next(self.transformer.parameters()).dtype

    def forward(self, batch):
        ############################################################
        voxel = batch["voxel"]
        condition = batch["condition"]
        target_object = batch["object"]
        ############################################################
        # VAE encode
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            latents = self.vae.encode(voxel.to(self.vae.dtype)).latent_dist.sample()
            num_channels_latents = latents.shape[1]
            ############################################################
            condition = self.vae.encode(condition.to(self.vae.dtype)).latent_dist.sample()
            target_object = self.vae.encode(target_object.to(self.vae.dtype)).latent_dist.sample()
            ############################################################

        batch_size = latents.shape[0]

        # Sample random timesteps for each sample
        sigmas = compute_density_for_timestep_sampling(
            self.cfg.weighting_scheme,
            batch_size,
            logit_mean=0.0,
            logit_std=1.0,
        ).to(latents.device)
        sigmas = self.noise_scheduler._time_shift_constant(sigmas)
        timesteps = self.noise_scheduler._sigma_to_t(sigmas)

        # Add noise to latents according to flow matching
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.scale_noise(
            latents, timesteps, noise, sigma=sigmas
        )

        # Pack latents
        resolution = self.cfg.resolution
        latent_resolution = 2 * (resolution // (self.vae_scale_factor * 2))
        noisy_latents = self._pack_latents(
            noisy_latents,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            resolution=latent_resolution,
            patch_size=self.patch_size,
        )

        ############################################################
        condition = self._pack_latents(
            condition,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            resolution=latent_resolution,
            patch_size=self.patch_size,
        )

        target_object = self._pack_latents(
            target_object,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            resolution=latent_resolution,
            patch_size=self.patch_size,
        )
        ############################################################

        # Prepare text conditions
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            cond_embeds = self.condition_encoder(
                **batch["llm_features"], output_hidden_states=True
            ).hidden_states[-1]
            if "cond_drop_idx" in batch:
                cond_embeds = cond_embeds[:, batch["cond_drop_idx"] :]

            # zero padding to match the length of the prompt
            for bid in range(cond_embeds.shape[0]):
                cond_embeds[
                    bid, : (batch["llm_features"]["attention_mask"][bid] == 0).sum(), :
                ] = 0

        cond_embeds = self.connector(cond_embeds)
        ############################################################
        # Prepare shapes and lens for positional encoding x 3
        vox_shapes = [
            [
                (
                    3,
                    latent_resolution // self.patch_size,
                    latent_resolution // self.patch_size,
                    latent_resolution // self.patch_size,
                )
            ]
        ] * batch_size
        # FIXME: we padding zero to match the length of the prompt
        # so we set positional encoding length to the total length of the prompt including the padding.
        ############################################################
        txt_seq_lens = [cond_embeds.shape[1]] * batch_size

        # Cast input to the same dtype as transformer to avoid dtype mismatch
        noisy_latents = noisy_latents.to(self.my_dtype)
        timesteps = timesteps.to(self.my_dtype)
        cond_embeds = cond_embeds.to(self.my_dtype)

        model_input = noisy_latents
        timesteps = timesteps / self.noise_scheduler.config.num_train_timesteps

        model_pred = self.transformer(
            hidden_states=model_input,
            encoder_hidden_states=cond_embeds,
            timestep=timesteps,
            vox_shapes=vox_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
            ############################################################
            condition=condition,
            target_object=target_object
            ############################################################
        )[0]

        # unpack latents
        model_pred = self._unpack_latents(
            model_pred,
            resolution=self.cfg.resolution,
            patch_size=self.patch_size,
            vae_scale_factor=self.vae_scale_factor,
        )

        # Flow matching loss
        target = noise - latents

        weighting = compute_loss_weighting_for_sd3(self.cfg.weighting_scheme, sigmas)
        loss = torch.mean(
            (
                weighting.float()[:, None, None, None, None]
                * (model_pred.float() - target.float()) ** 2
            ).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)

        self.log("train/loss", loss, prog_bar=True)

        # will execute self.on_check_train every self.cfg.check_train_every_n_steps steps
        self.check_train(batch, batch_idx)

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    @torch.no_grad()
    @torch.amp.autocast("cuda", enabled=False)
    def visualization_step(self, batch, batch_idx=0, step_name="train", step=None):

        if "id_list" in batch and "prompt" in batch and "model_id" in batch:
            import json
            for i, (id_list, prompt, model_id) in enumerate(zip(batch["id_list"], batch["prompt"], batch["model_id"])):
                if step is None:
                    save_path = self.get_save_path(
                        f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_info_{i}.jsonl"
                    )
                else:
                    save_path = self.get_save_path(
                        f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_info_{i}_step_{step}.jsonl"
                    )

                if isinstance(id_list, str):
                    id_list = [int(x) for x in id_list.split(",") if x]

                record = {
                    "id_list": id_list,
                    "prompt": prompt,
                    "model_id": model_id
                }
                with open(save_path, "w") as f:
                    f.write(json.dumps(record) + "\n")

        elif "prompt" in batch:
            save_path = self.get_save_path(
                f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_prompt.txt"
            )
            with open(save_path, "w") as f:
                f.write("\n".join(batch["prompt"]))

        if "voxel" in batch:
            for i in range(batch["voxel"].shape[0]):
                voxel = batch["voxel"][i]
                if step is None:
                    save_path = self.get_save_path(
                        f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_voxel_{i}.ply"
                    )
                else:
                    save_path = self.get_save_path(
                        f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_voxel_{i}_step_{step}.ply"
                    )
                coords = torch.argwhere(voxel > 0)[:, [1, 2, 3]].int()
                coords_np = (coords.cpu().numpy() + 0.5) / self.cfg.resolution - 0.5
                utils3d.io.write_ply(save_path, coords_np)

        if "condition" in batch:
            for i in range(batch["condition"].shape[0]):
                voxel = batch["condition"][i]
                if step is None:
                    save_path = self.get_save_path(
                        f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_condition_{i}.ply"
                    )
                else:
                    save_path = self.get_save_path(
                        f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_condition_{i}_step_{step}.ply"
                    )
                coords = torch.argwhere(voxel > 0)[:, [1, 2, 3]].int()
                coords_np = (coords.cpu().numpy() + 0.5) / self.cfg.resolution - 0.5
                utils3d.io.write_ply(save_path, coords_np)

        if "object" in batch:
            for i in range(batch["object"].shape[0]):
                voxel = batch["object"][i]
                if step is None:
                    save_path = self.get_save_path(
                        f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_object_{i}.ply"
                    )
                else:
                    save_path = self.get_save_path(
                        f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_object_{i}_step_{step}.ply"
                    )
                coords = torch.argwhere(voxel > 0)[:, [1, 2, 3]].int()
                coords_np = (coords.cpu().numpy() + 0.5) / self.cfg.resolution - 0.5
                utils3d.io.write_ply(save_path, coords_np)


    def on_check_train(self, batch, batch_idx):
        self.visualization_step(batch, batch_idx, step_name="train")

    def on_save_checkpoint(self, checkpoint):
        if self.global_rank == 0:
            save_dir = os.path.join(os.path.dirname(self.get_save_dir()), "pipeline")
            os.makedirs(save_dir, exist_ok=True)
            if os.path.exists(os.path.join(save_dir, "condition_encoder")):
                self.transformer.save_pretrained(os.path.join(save_dir, "transformer"))
            else:
                self.non_module("pipeline").save_pretrained(save_dir)
            self.non_module("pipeline").save_custom_adapter(
                save_dir, "custom_adapter.safetensors", safe_serialization=True
            )

    @torch.no_grad()
    def inference(self, batch):
        # bsz = 1
        pcd = self.non_module("pipeline")(
            prompt=batch["llm_features"],
            negative_prompt=" ",
            num_inference_steps=self.cfg.eval_num_inference_steps,
            resolution=self.cfg.resolution,
            guidance_scale=self.cfg.eval_guidance_scale,
            condition=batch["condition"],
            target_object=batch["object"]
        ).pcds[0]

        return pcd

    def validation_step(self, batch, batch_idx):
        def points_to_dense_from_normed(points, resolution=64):
            # [-0.5,0.5] → [0,resolution)
            coords = (points + 0.5) * resolution - 0.5
            coords = np.clip(coords, 0, resolution - 1).astype(int)

            grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
            grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
            return grid[None, None]  # (1,1,64,64,64)
        
        if isinstance(batch, dict):
            output_pcd = self.inference(batch)

            if self.cfg.check_val_limit_rank == -1 or (
                self.cfg.check_val_limit_rank > 0
                and self.global_rank < self.cfg.check_val_limit_rank
            ):
                self.visualization_step(batch, batch_idx, step_name="validation")
                save_name = (
                    f"it{self.true_global_step}-validation-{self.global_rank}_{batch_idx}"
                )
                utils3d.io.write_ply(
                    self.get_save_path(f"{save_name}_output.ply"), output_pcd
                )

        elif isinstance(batch, list) and all(isinstance(b, dict) for b in batch):
            prev_output = None
            for i, b in enumerate(batch):
                if prev_output is not None:
                    b = dict(b)
                    b["condition"] = prev_output

                output_pcd = self.inference(b)
                if (
                    self.cfg.check_val_limit_rank > 0
                    and self.global_rank < self.cfg.check_val_limit_rank
                ):
                    self.visualization_step(b, batch_idx, step_name="validation", step=i)
                    save_name = (
                        f"it{self.true_global_step}-validation-"
                        f"{self.global_rank}_{batch_idx}_step_{i}"
                    )
                    utils3d.io.write_ply(
                        self.get_save_path(f"{save_name}_output.ply"), output_pcd
                    )
                dense_grid = points_to_dense_from_normed(output_pcd, resolution=64)
                prev_output = torch.from_numpy(dense_grid).float().to(self.device)

    def test_step(self, batch, batch_idx):
        pass

    def on_test_end(self):
        pass
