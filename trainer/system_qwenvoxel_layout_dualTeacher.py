import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import NoneType
from typing import Any, Dict, List, Optional

import numpy as np
import PIL
import PIL.Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import utils3d
from einops import rearrange
from omegaconf import OmegaConf
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

from ..models.autoencoder_kl_voxel import AutoencoderKLVoxel
from ..models.transformer_qwenvoxel import QwenVoxelTransformer3DModel
from ..models.transformer_qwenvoxel_layout_synthesis import (
    QwenVoxelTransformer3DModelLayoutSynthesis,
)
from ..pipeline.pipeline_qwenvoxel_layout_synthesis import (
    QwenVoxelPipelineLayoutSynthesis,
)
from ..utils.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from ..utils.system_utils.logging import warn
from ..utils.typing import *
from .base import BaseSystem


class QwenVoxelSystemLayoutSynthesisSF(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        # Model
        base_model: str = ""
        fake_score_pretrained_model_name_or_path: str = ""
        teacher_pretrained_model_name_or_path: str = ""
        scene_teacher_pretrained_model_name_or_path: str = ""

        # Training
        trainable_modules: List[str] = field(default_factory=list)
        non_trainable_modules: List[str] = field(default_factory=list)
        gradient_checkpointing: bool = False
        resolution: int = 64

        # Self-Forcing
        sf_num_inference_steps: int = 4
        sf_same_step_across_objects: bool = True
        sf_last_step_only: bool = False

        # DMD
        dmd_enabled: bool = True
        dmd_weight: float = 1.0
        real_guidance_scale: float = 3.0
        fake_guidance_scale: float = 0.0
        denoising_loss_type: str = "flow"
        
        # Teacher settings
        teacher_flow_shift: float = 1.0

        # Timesteps
        denoising_step_list: List[int] = field(default_factory=list)
        num_train_timestep: int = 1000
        min_score_timestep: int = 0

        # Alternating training
        dfake_gen_update_ratio: int = 5

        # EMA
        ema_weight: float = 0.99
        ema_start_step: int = 200
        ema_offload_cpu: bool = True

        # Evaluation
        eval_num_inference_steps: int = 50
        eval_seed: int = 42
        eval_guidance_scale: float = 7.0

    cfg: Config

    def configure(self):
        super().configure()

        # Load models
        transformer = QwenVoxelTransformer3DModelLayoutSynthesis.from_pretrained(
            self.cfg.base_model, subfolder="transformer"
        )
        vae = AutoencoderKLVoxel.from_pretrained(
            self.cfg.base_model, subfolder="vae"
        )
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.cfg.base_model, subfolder="scheduler"
        )
        feature_extractor = Qwen2_5_VLProcessor.from_pretrained(
            self.cfg.base_model, subfolder="feature_extractor"
        )
        condition_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.cfg.base_model, subfolder="condition_encoder"
        )

        pipeline = QwenVoxelPipelineLayoutSynthesis(
            vae=vae,
            transformer=transformer,
            scheduler=noise_scheduler,
            condition_encoder=condition_encoder,
            feature_extractor=feature_extractor,
        )
        pipeline.init_custom_adapter()

        # Fake score (critic)
        fake_score_model_path = (
            self.cfg.fake_score_pretrained_model_name_or_path
            if self.cfg.fake_score_pretrained_model_name_or_path
            else self.cfg.base_model
        )
        self.fake_score = QwenVoxelTransformer3DModelLayoutSynthesis.from_pretrained(
            fake_score_model_path, subfolder="transformer"
        )

        # Teacher (object-level)
        self.teacher_noise_scheduler = None
        if self.cfg.dmd_enabled:
            teacher_model_id = (
                self.cfg.teacher_pretrained_model_name_or_path
                if self.cfg.teacher_pretrained_model_name_or_path
                else self.cfg.base_model
            )
            teacher_transformer = QwenVoxelTransformer3DModelLayoutSynthesis.from_pretrained(
                teacher_model_id, subfolder="transformer"
            )
            teacher_transformer.requires_grad_(False)
            teacher_transformer.eval()
            self.__dict__['teacher_transformer'] = teacher_transformer.cpu()
            self.teacher_noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=self.cfg.teacher_flow_shift,
            )
            
            # Scene-level teacher
            scene_teacher_model_id = (
                self.cfg.scene_teacher_pretrained_model_name_or_path
                if self.cfg.scene_teacher_pretrained_model_name_or_path
                else self.cfg.base_model
            )
            scene_teacher_transformer = QwenVoxelTransformer3DModel.from_pretrained(
                scene_teacher_model_id, subfolder="transformer"
            )
            scene_teacher_transformer.requires_grad_(False)
            scene_teacher_transformer.eval()
            self.__dict__['scene_teacher_transformer'] = scene_teacher_transformer.cpu()
        else:
            self.__dict__['teacher_transformer'] = None
            self.__dict__['scene_teacher_transformer'] = None

        # Disable automatic optimization
        self.automatic_optimization = False

        # Register models
        self.register_non_module("pipeline", pipeline)
        self.transformer = transformer
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.feature_extractor = feature_extractor
        self.connector = pipeline.connector
        
        # Keep condition_encoder on GPU but exclude from DeepSpeed management
        # DeepSpeed Zero-3 hangs when managing Qwen2.5-VL
        # Use __dict__ to prevent it from being registered as a submodule
        condition_encoder.requires_grad_(False)
        condition_encoder.eval()
        # Move to GPU immediately if available
        if torch.cuda.is_available():
            condition_encoder = condition_encoder.to('cuda')
        self.__dict__['condition_encoder'] = condition_encoder

        # Set training mode
        self.vae.requires_grad_(False)
        self.connector.requires_grad_(True)
        self.transformer.requires_grad_(True)
        self.fake_score.requires_grad_(True)

        self.vae.eval()
        self.transformer.train()
        self.connector.train()
        self.fake_score.train()

        if self.cfg.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # Other parameters
        self.patch_size = self.non_module("pipeline").patch_size
        self.vae_scale_factor = self.non_module("pipeline").vae_scale_factor
        self._pack_latents = self.non_module("pipeline")._pack_latents
        self._unpack_latents = self.non_module("pipeline")._unpack_latents

        # Build timesteps
        if self.cfg.denoising_step_list and len(self.cfg.denoising_step_list) > 0:
            self.noise_scheduler.set_timesteps(self.cfg.sf_num_inference_steps, device=self.device)
            timesteps_list = []
            for idx in self.cfg.denoising_step_list:
                if idx == -1:
                    timesteps_list.append(0.0)
                else:
                    timesteps_list.append(self.noise_scheduler.timesteps[idx].item())
            self.sf_timesteps = torch.tensor(timesteps_list, dtype=torch.float32, device=self.device)
            
            if self.teacher_noise_scheduler is not None:
                self.teacher_noise_scheduler.set_timesteps(self.cfg.sf_num_inference_steps, device=self.device)
                teacher_timesteps_list = []
                for idx in self.cfg.denoising_step_list:
                    if idx == -1:
                        teacher_timesteps_list.append(0.0)
                    else:
                        teacher_timesteps_list.append(self.teacher_noise_scheduler.timesteps[idx].item())
                self.teacher_sf_timesteps = torch.tensor(teacher_timesteps_list, dtype=torch.float32, device=self.device)
        else:
            self.noise_scheduler.set_timesteps(self.cfg.sf_num_inference_steps, device=self.device)
            self.sf_timesteps = self.noise_scheduler.timesteps.to(device=self.device)
            if self.teacher_noise_scheduler is not None:
                self.teacher_noise_scheduler.set_timesteps(self.cfg.sf_num_inference_steps, device=self.device)
                self.teacher_sf_timesteps = self.teacher_noise_scheduler.timesteps.to(device=self.device)

        self.min_step = int(0.02 * self.cfg.num_train_timestep)
        self.max_step = int(0.3 * self.cfg.num_train_timestep)
        self._ema_state = {}

    def on_fit_start(self):
        self.my_dtype = next(self.transformer.parameters()).dtype
        self.sf_timesteps = self.sf_timesteps.to(device=self.device)
        if self.teacher_noise_scheduler is not None:
            self.teacher_sf_timesteps = self.teacher_sf_timesteps.to(device=self.device)
        
        # Ensure condition_encoder is on the correct GPU device
        if self.condition_encoder is not None and hasattr(self.condition_encoder, 'device'):
            current_device = next(self.condition_encoder.parameters()).device
            if current_device != self.device:
                self.__dict__['condition_encoder'] = self.condition_encoder.to(self.device)
        
        # Ensure teachers stay on CPU to save VRAM (use __dict__ to avoid DeepSpeed)
        if self.teacher_transformer is not None:
            self.__dict__['teacher_transformer'] = self.teacher_transformer.cpu()
        if self.scene_teacher_transformer is not None:
            self.__dict__['scene_teacher_transformer'] = self.scene_teacher_transformer.cpu()
        if (
            self.cfg.ema_weight > 0.0
            and self.global_step >= int(getattr(self.cfg, "ema_start_step", 0))
        ):
            self._init_ema()

    def _toggle_requires_grad(self, module: torch.nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def _init_ema(self):
        self._ema_state = {}
        for name, p in self.transformer.named_parameters():
            if not p.requires_grad:
                continue
            state = p.detach().clone()
            if getattr(self.cfg, "ema_offload_cpu", False):
                state = state.to("cpu")
            self._ema_state[name] = state

    def _update_ema(self):
        if float(getattr(self.cfg, "ema_weight", 0.0)) <= 0.0:
            return
        if self.global_step < int(getattr(self.cfg, "ema_start_step", 0)):
            return
        decay = float(self.cfg.ema_weight)
        for name, p in self.transformer.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self._ema_state:
                state = p.detach().clone()
                if getattr(self.cfg, "ema_offload_cpu", False):
                    state = state.to("cpu")
                self._ema_state[name] = state
                continue
            if getattr(self.cfg, "ema_offload_cpu", False):
                self._ema_state[name].mul_(decay).add_(p.detach().to("cpu"), alpha=1.0 - decay)
            else:
                self._ema_state[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    def _sample_sigma(self, t_vec: torch.Tensor, n_dim: int):
        timesteps_flat = t_vec.flatten()
        schedule_timesteps = self.noise_scheduler.timesteps.to(t_vec.device)
        schedule_sigmas = self.noise_scheduler.sigmas.to(t_vec.device)
        timestep_id = torch.argmin(
            (schedule_timesteps.unsqueeze(0) - timesteps_flat.unsqueeze(1)).abs(), dim=1
        )
        sigmas = schedule_sigmas[timestep_id].reshape(t_vec.shape)
        if n_dim == 4:
            sigmas = sigmas.view(-1, 1, 1, 1)
        elif n_dim == 5:
            sigmas = sigmas.view(-1, 1, 1, 1, 1)
        return sigmas.float()

    def _add_noise(self, clean: torch.Tensor, noise: torch.Tensor, t_vec: torch.Tensor):
        sigmas = self._sample_sigma(t_vec, clean.ndim)
        clean_fp32 = clean.float()
        noise_fp32 = noise.float()
        noisy = sigmas * noise_fp32 + (1.0 - sigmas) * clean_fp32
        return noisy.to(dtype=clean.dtype)

    def _generate_and_sync_step_indices(self, num_objects, num_steps, device):
        """Generate bp_step_index for each object (like system_wan_sf.py)"""
        if num_objects <= 0:
            return []
        
        if self.cfg.sf_last_step_only:
            return [num_steps - 1] * num_objects
        
        use_ddp = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if use_ddp else 0
        
        if self.cfg.sf_same_step_across_objects:
            # Same step for all objects
            if rank == 0:
                idx_tensor = torch.randint(0, num_steps, (1,), device=device, dtype=torch.long)
            else:
                idx_tensor = torch.empty(1, device=device, dtype=torch.long)
            if use_ddp:
                dist.broadcast(idx_tensor, src=0)
            return [int(idx_tensor.item())] * num_objects
        else:
            # Different step for each object
            if rank == 0:
                idx_tensor = torch.randint(0, num_steps, (num_objects,), device=device, dtype=torch.long)
            else:
                idx_tensor = torch.empty(num_objects, device=device, dtype=torch.long)
            if use_ddp:
                dist.broadcast(idx_tensor, src=0)
            return [int(v) for v in idx_tensor.tolist()]


    def _run_generator_autoregressive(
        self,
        batch_size,
        initial_condition,
        num_objects_per_sample,
        target_objects_list,
        cond_embeds,
        vox_shapes,
        txt_seq_lens,
        return_all_stages=False,
    ):
        num_steps = len(self.sf_timesteps)
        max_num_objects = max(num_objects_per_sample)
        
        noise_per_object = []
        for obj_idx in range(max_num_objects):
            noise_obj = torch.randn(
                batch_size,
                initial_condition.shape[1],
                initial_condition.shape[2],
                initial_condition.shape[3],
                initial_condition.shape[4],
                device=self.device,
                dtype=initial_condition.dtype
            )
            noise_per_object.append(noise_obj)
        
        bp_step_indices = self._generate_and_sync_step_indices(
            max_num_objects, num_steps, self.device
        )
        
        current_scene = initial_condition.clone()
        all_stages = []
        
        for obj_idx in range(max_num_objects):
            condition_packed = self._pack_latents(
                current_scene,
                batch_size=batch_size,
                num_channels_latents=current_scene.shape[1],
                resolution=current_scene.shape[2],
                patch_size=self.patch_size,
            )
            
            target_object = target_objects_list[obj_idx]
            target_object_packed = self._pack_latents(
                target_object,
                batch_size=batch_size,
                num_channels_latents=target_object.shape[1],
                resolution=target_object.shape[2],
                patch_size=self.patch_size,
            )
            
            bp_step_index = bp_step_indices[obj_idx]
            
            next_scene_packed = self._multi_step_denoising(
                noisy_init=noise_per_object[obj_idx],
                condition=condition_packed,
                target_object=target_object_packed,
                cond_embeds=cond_embeds,
                vox_shapes=vox_shapes,
                txt_seq_lens=txt_seq_lens,
                bp_step_index=bp_step_index,
            )
            
            next_scene = self._unpack_latents(
                next_scene_packed,
                resolution=self.cfg.resolution,
                patch_size=self.patch_size,
                vae_scale_factor=self.vae_scale_factor,
            )
            
            current_scene = next_scene
            if return_all_stages:
                all_stages.append(current_scene)
            
            del condition_packed, target_object_packed, next_scene_packed
            if obj_idx < max_num_objects - 1:
                torch.cuda.empty_cache()
        
        return all_stages if return_all_stages else current_scene

    def _multi_step_denoising(
        self,
        noisy_init,
        condition,
        target_object,
        cond_embeds,
        vox_shapes,
        txt_seq_lens,
        bp_step_index,
    ):
        batch_size = noisy_init.shape[0]
        
        noisy = self._pack_latents(
            noisy_init,
            batch_size=batch_size,
            num_channels_latents=noisy_init.shape[1],
            resolution=noisy_init.shape[2],
            patch_size=self.patch_size,
        )
        
        x_pred = None
        
        for si, t in enumerate(self.sf_timesteps):
            should_bp = si == bp_step_index
            cm = nullcontext() if should_bp else torch.no_grad()
            
            with cm:
                timestep = t.expand(batch_size) / self.noise_scheduler.config.num_train_timesteps
                
                model_pred = self.transformer(
                    hidden_states=noisy,
                    encoder_hidden_states=cond_embeds,
                    timestep=timestep,
                    vox_shapes=vox_shapes,
                    txt_seq_lens=txt_seq_lens,
                    condition=condition,
                    target_object=target_object,
                )[0]
                
                sigma = self._sample_sigma(t.expand(batch_size), noisy.ndim)
                x_pred = (noisy.float() - sigma * model_pred.float()).to(noisy.dtype)
                
                if should_bp:
                    break
                
                if si < len(self.sf_timesteps) - 1:
                    t_next = self.sf_timesteps[si + 1]
                    eps_next = torch.randn_like(x_pred)
                    noisy = self._add_noise(x_pred, eps_next, t_next.expand(batch_size))
        
        return x_pred

    def _compute_dmd_loss_for_scene(self, scene_pred, gt_condition_latent, target_object_latent, 
                                     batch_size, cond_embeds, uncond_embeds, 
                                     txt_seq_lens, uncond_txt_seq_lens, vox_shapes, latent_resolution):
        timestep = torch.randint(self.min_step, self.max_step, (batch_size,), device=self.device)
        eps_dmd = torch.randn_like(scene_pred)
        
        with torch.no_grad():
            scene_detached = scene_pred.detach()
            noisy_dmd = self._add_noise(scene_detached, eps_dmd, timestep).detach()
            
            noisy_packed = self._pack_latents(
                noisy_dmd, batch_size=batch_size, num_channels_latents=noisy_dmd.shape[1],
                resolution=latent_resolution, patch_size=self.patch_size,
            )
            timestep_norm = timestep.float() / self.noise_scheduler.config.num_train_timesteps
            
            gt_condition_packed = self._pack_latents(
                gt_condition_latent, batch_size=batch_size, num_channels_latents=gt_condition_latent.shape[1],
                resolution=gt_condition_latent.shape[2], patch_size=self.patch_size,
            )
            target_object_packed = self._pack_latents(
                target_object_latent, batch_size=batch_size, num_channels_latents=target_object_latent.shape[1],
                resolution=target_object_latent.shape[2], patch_size=self.patch_size,
            )
            
            teacher_sigma = self._sample_sigma(timestep, noisy_dmd.ndim)
            noisy_packed_cpu = noisy_packed.cpu().float()
            cond_embeds_cpu = cond_embeds.cpu().float()
            uncond_embeds_cpu = uncond_embeds.cpu().float()
            timestep_norm_cpu = timestep_norm.cpu().float()
            gt_condition_packed_cpu = gt_condition_packed.cpu().float()
            target_object_packed_cpu = target_object_packed.cpu().float()
            
            mp_t = self.teacher_transformer(
                hidden_states=noisy_packed_cpu, encoder_hidden_states=cond_embeds_cpu,
                timestep=timestep_norm_cpu, vox_shapes=vox_shapes, txt_seq_lens=txt_seq_lens,
                condition=gt_condition_packed_cpu, target_object=target_object_packed_cpu,
            )[0]
            mp_t = self._unpack_latents(
                mp_t.to(noisy_dmd.device), resolution=self.cfg.resolution,
                patch_size=self.patch_size, vae_scale_factor=self.vae_scale_factor,
            )
            x0_t = noisy_dmd.float() - teacher_sigma * mp_t.float()
            
            mp_t_u = self.teacher_transformer(
                hidden_states=noisy_packed_cpu, encoder_hidden_states=uncond_embeds_cpu,
                timestep=timestep_norm_cpu, vox_shapes=vox_shapes, txt_seq_lens=uncond_txt_seq_lens,
                condition=gt_condition_packed_cpu, target_object=target_object_packed_cpu,
            )[0]
            mp_t_u = self._unpack_latents(
                mp_t_u.to(noisy_dmd.device), resolution=self.cfg.resolution,
                patch_size=self.patch_size, vae_scale_factor=self.vae_scale_factor,
            )
            x0_t_u = noisy_dmd.float() - teacher_sigma * mp_t_u.float()
            x0_t_cfg = x0_t + (x0_t - x0_t_u) * self.cfg.real_guidance_scale
            
            del mp_t, mp_t_u, x0_t, x0_t_u, noisy_packed_cpu, cond_embeds_cpu, uncond_embeds_cpu, timestep_norm_cpu
            del gt_condition_packed_cpu, target_object_packed_cpu
            torch.cuda.empty_cache()
            
            sigma_dmd = self._sample_sigma(timestep, noisy_dmd.ndim)
            mp_f = self.fake_score(
                hidden_states=noisy_packed, encoder_hidden_states=cond_embeds,
                timestep=timestep_norm, vox_shapes=vox_shapes, txt_seq_lens=txt_seq_lens,
                condition=gt_condition_packed, target_object=target_object_packed,
            )[0]
            mp_f = self._unpack_latents(
                mp_f, resolution=self.cfg.resolution,
                patch_size=self.patch_size, vae_scale_factor=self.vae_scale_factor,
            )
            x0_f = noisy_dmd.float() - sigma_dmd * mp_f.float()
            
            if self.cfg.fake_guidance_scale != 0.0:
                mp_f_u = self.fake_score(
                    hidden_states=noisy_packed, encoder_hidden_states=uncond_embeds,
                    timestep=timestep_norm, vox_shapes=vox_shapes, txt_seq_lens=uncond_txt_seq_lens,
                    condition=gt_condition_packed, target_object=target_object_packed,
                )[0]
                mp_f_u = self._unpack_latents(
                    mp_f_u, resolution=self.cfg.resolution,
                    patch_size=self.patch_size, vae_scale_factor=self.vae_scale_factor,
                )
                x0_f_u = noisy_dmd.float() - sigma_dmd * mp_f_u.float()
                x0_f_cfg = x0_f + (x0_f - x0_f_u) * self.cfg.fake_guidance_scale
            else:
                x0_f_cfg = x0_f
            
            grad = scene_pred - x0_t_cfg
            normalizer = (scene_detached - x0_t_cfg).abs().mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / (normalizer + 1e-8)
            grad = torch.nan_to_num(grad)
            
            del mp_f, x0_f, x0_f_cfg, noisy_packed, noisy_dmd
            del gt_condition_packed, target_object_packed
            if self.cfg.fake_guidance_scale != 0.0:
                del mp_f_u, x0_f_u
        
        loss_dmd = 0.5 * F.mse_loss(scene_pred.float(), (scene_pred.float() - grad.float()).detach())
        return loss_dmd

    def _compute_scene_dmd_loss(self, final_scene_pred, batch_size, cond_embeds, uncond_embeds,
                                 txt_seq_lens, uncond_txt_seq_lens, vox_shapes, vox_shapes_single, latent_resolution):
        timestep = torch.randint(self.min_step, self.max_step, (batch_size,), device=self.device)
        eps_dmd = torch.randn_like(final_scene_pred)
        
        with torch.no_grad():
            final_scene_detached = final_scene_pred.detach()
            noisy_dmd = self._add_noise(final_scene_detached, eps_dmd, timestep).detach()
            
            noisy_packed = self._pack_latents(
                noisy_dmd, batch_size=batch_size, num_channels_latents=noisy_dmd.shape[1],
                resolution=latent_resolution, patch_size=self.patch_size,
            )
            timestep_norm = timestep.float() / self.noise_scheduler.config.num_train_timesteps
            
            teacher_sigma = self._sample_sigma(timestep, noisy_dmd.ndim)
            noisy_packed_cpu = noisy_packed.cpu().float()
            cond_embeds_cpu = cond_embeds.cpu().float()
            uncond_embeds_cpu = uncond_embeds.cpu().float()
            timestep_norm_cpu = timestep_norm.cpu().float()
            
            mp_t = self.scene_teacher_transformer(
                hidden_states=noisy_packed_cpu, encoder_hidden_states=cond_embeds_cpu,
                timestep=timestep_norm_cpu, vox_shapes=vox_shapes_single, txt_seq_lens=txt_seq_lens,
            )[0]
            mp_t = self._unpack_latents(
                mp_t.to(noisy_dmd.device), resolution=self.cfg.resolution,
                patch_size=self.patch_size, vae_scale_factor=self.vae_scale_factor,
            )
            x0_t = noisy_dmd.float() - teacher_sigma * mp_t.float()
            
            mp_t_u = self.scene_teacher_transformer(
                hidden_states=noisy_packed_cpu, encoder_hidden_states=uncond_embeds_cpu,
                timestep=timestep_norm_cpu, vox_shapes=vox_shapes_single, txt_seq_lens=uncond_txt_seq_lens,
            )[0]
            mp_t_u = self._unpack_latents(
                mp_t_u.to(noisy_dmd.device), resolution=self.cfg.resolution,
                patch_size=self.patch_size, vae_scale_factor=self.vae_scale_factor,
            )
            x0_t_u = noisy_dmd.float() - teacher_sigma * mp_t_u.float()
            x0_t_cfg = x0_t + (x0_t - x0_t_u) * self.cfg.real_guidance_scale
            
            grad = final_scene_pred - x0_t_cfg
            normalizer = (final_scene_detached - x0_t_cfg).abs().mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / (normalizer + 1e-8)
            grad = torch.nan_to_num(grad)
            
            del mp_t, mp_t_u, x0_t, x0_t_u, noisy_packed_cpu, cond_embeds_cpu, uncond_embeds_cpu, timestep_norm_cpu
            del noisy_packed, noisy_dmd
        
        loss_dmd = 0.5 * F.mse_loss(final_scene_pred.detach().float(), (final_scene_pred.float() - grad.float()).detach())
        return loss_dmd

    def forward(self, batch):
        batch_size = batch["condition"].shape[0]
        # if self.global_rank == 0:
        #     print("="*6)
        #     print(batch['prompt'])
        #     print("="*6)
        
        with torch.no_grad():
            condition_latent = self.vae.encode(batch["condition"].to(self.vae.dtype)).latent_dist.sample()
            gt_voxel_latent = self.vae.encode(batch["voxel"].to(self.vae.dtype)).latent_dist.sample()
            if self.global_step % 10 == 0:
                self._save_debug_output(condition_latent, "condition", batch_size)
                if "voxel" in batch:
                    self._save_debug_output(gt_voxel_latent, "gt_voxel", batch_size)
            
            target_objects_latent = []
            for target_obj in batch["object"]:
                target_obj_latent = self.vae.encode(target_obj.to(self.vae.dtype)).latent_dist.sample()
                target_objects_latent.append(target_obj_latent)
            
            gt_conditions_latent = []
            for gt_cond in batch["gt_conditions"]:
                gt_cond_latent = self.vae.encode(gt_cond.to(self.vae.dtype)).latent_dist.sample()
                gt_conditions_latent.append(gt_cond_latent)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            cond_embeds = self.condition_encoder(**batch["llm_features"], output_hidden_states=True).hidden_states[-1]
            if "cond_drop_idx" in batch:
                cond_embeds = cond_embeds[:, batch["cond_drop_idx"]:]
            for bid in range(cond_embeds.shape[0]):
                cond_embeds[bid, :(batch["llm_features"]["attention_mask"][bid] == 0).sum(), :] = 0

        cond_embeds = self.connector(cond_embeds.to(self.my_dtype))
        txt_seq_lens = [cond_embeds.shape[1]] * batch_size

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            pipeline = self.non_module("pipeline")
            negative_prompts = [" "] * batch_size
            from transformers import BatchFeature
            neg_text_inputs = [f"{pipeline.prompt_template.format(p)}" for p in negative_prompts]
            neg_features = pipeline.feature_extractor(
                text=neg_text_inputs, images=None,
                max_length=pipeline.prompt_max_length + pipeline.prompt_template_start_idx,
                padding=True, padding_side="left", truncation=True, return_tensors="pt",
            ).to(device=self.device)
            
            uncond_embeds = self.condition_encoder(**neg_features, output_hidden_states=True).hidden_states[-1]
            uncond_embeds = uncond_embeds[:, pipeline.prompt_template_start_idx:]
            for bid in range(uncond_embeds.shape[0]):
                uncond_embeds[bid, :(neg_features["attention_mask"][bid] == 0).sum(), :] = 0

        uncond_embeds = self.connector(uncond_embeds.to(self.my_dtype))
        uncond_txt_seq_lens = [uncond_embeds.shape[1]] * batch_size
        
        latent_resolution = 2 * (self.cfg.resolution // (self.vae_scale_factor * 2))
        vox_shapes = [
            [(3, latent_resolution // self.patch_size, latent_resolution // self.patch_size, latent_resolution // self.patch_size)]
        ] * batch_size
        vox_shapes_single = [
            [(1, latent_resolution // self.patch_size, latent_resolution // self.patch_size, latent_resolution // self.patch_size)]
        ] * batch_size
        
        all_scenes = self._run_generator_autoregressive(
            batch_size=batch_size, initial_condition=condition_latent,
            num_objects_per_sample=batch["num_objects"], target_objects_list=target_objects_latent,
            cond_embeds=cond_embeds, vox_shapes=vox_shapes, txt_seq_lens=txt_seq_lens,
            return_all_stages=True,
        )
        
        if self.global_step % 10 == 0:
            self._save_debug_output(all_scenes[-1], "final_scene_pred", batch_size)
        
        if not self.cfg.dmd_enabled:
            return torch.zeros((), device=self.device)
        
        total_loss = 0.0
        # object-level dmd loss
        for idx, scene_pred in enumerate(all_scenes):
            loss = self._compute_dmd_loss_for_scene(
                scene_pred, gt_conditions_latent[idx], target_objects_latent[idx],
                batch_size, cond_embeds, uncond_embeds,
                txt_seq_lens, uncond_txt_seq_lens, vox_shapes, latent_resolution
            )
            total_loss = total_loss + loss
        
        # scene-level dmd loss  
        # scene_loss = self._compute_scene_dmd_loss(
        #     all_scenes[-1], batch_size, cond_embeds, uncond_embeds,
        #     txt_seq_lens, uncond_txt_seq_lens, vox_shapes, vox_shapes_single, latent_resolution
        # )
        # total_loss = total_loss + scene_loss

        return self.cfg.dmd_weight * total_loss

    def compute_critic_loss(self, batch):
        batch_size = batch["condition"].shape[0]
        
        with torch.no_grad():
            condition_latent = self.vae.encode(batch["condition"].to(self.vae.dtype)).latent_dist.sample()
            target_objects_latent = []
            for target_obj in batch["object"]:
                target_obj_latent = self.vae.encode(target_obj.to(self.vae.dtype)).latent_dist.sample()
                target_objects_latent.append(target_obj_latent)
            
            gt_conditions_latent = []
            for gt_cond in batch["gt_conditions"]:
                gt_cond_latent = self.vae.encode(gt_cond.to(self.vae.dtype)).latent_dist.sample()
                gt_conditions_latent.append(gt_cond_latent)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            cond_embeds = self.condition_encoder(**batch["llm_features"], output_hidden_states=True).hidden_states[-1]
            if "cond_drop_idx" in batch:
                cond_embeds = cond_embeds[:, batch["cond_drop_idx"]:]
            for bid in range(cond_embeds.shape[0]):
                cond_embeds[bid, :(batch["llm_features"]["attention_mask"][bid] == 0).sum(), :] = 0

        cond_embeds = self.connector(cond_embeds.to(self.my_dtype))
        txt_seq_lens = [cond_embeds.shape[1]] * batch_size
        
        latent_resolution = 2 * (self.cfg.resolution // (self.vae_scale_factor * 2))
        vox_shapes = [
            [(3, latent_resolution // self.patch_size, latent_resolution // self.patch_size, latent_resolution // self.patch_size)]
        ] * batch_size
        
        with torch.no_grad():
            all_scenes = self._run_generator_autoregressive(
                batch_size=batch_size, initial_condition=condition_latent,
                num_objects_per_sample=batch["num_objects"], target_objects_list=target_objects_latent,
                cond_embeds=cond_embeds, vox_shapes=vox_shapes, txt_seq_lens=txt_seq_lens,
                return_all_stages=True,
            )
        
        total_loss = 0.0
        for idx, scene_student in enumerate(all_scenes):
            scene_student = scene_student.detach()
            timestep = torch.randint(self.min_step, self.max_step, (batch_size,), device=self.device)
            sigma = self._sample_sigma(timestep, scene_student.ndim)
            critic_noise = torch.randn_like(scene_student)
            noisy_gen = self._add_noise(scene_student, critic_noise, timestep)
            
            noisy_packed = self._pack_latents(
                noisy_gen, batch_size=batch_size, num_channels_latents=noisy_gen.shape[1],
                resolution=noisy_gen.shape[2], patch_size=self.patch_size,
            )
            timestep_norm = timestep.float() / self.noise_scheduler.config.num_train_timesteps
            
            gt_condition_packed = self._pack_latents(
                gt_conditions_latent[idx], batch_size=batch_size, 
                num_channels_latents=gt_conditions_latent[idx].shape[1],
                resolution=gt_conditions_latent[idx].shape[2], patch_size=self.patch_size,
            )
            target_object_packed = self._pack_latents(
                target_objects_latent[idx], batch_size=batch_size,
                num_channels_latents=target_objects_latent[idx].shape[1],
                resolution=target_objects_latent[idx].shape[2], patch_size=self.patch_size,
            )
            
            model_pred = self.fake_score(
                hidden_states=noisy_packed, encoder_hidden_states=cond_embeds,
                timestep=timestep_norm, vox_shapes=vox_shapes, txt_seq_lens=txt_seq_lens,
                condition=gt_condition_packed, target_object=target_object_packed,
            )[0]
            
            model_pred = self._unpack_latents(
                model_pred, resolution=self.cfg.resolution,
                patch_size=self.patch_size, vae_scale_factor=self.vae_scale_factor,
            )
            x0_pred = noisy_gen - sigma * model_pred
            
            if self.cfg.denoising_loss_type == "flow":
                flow_pred = (noisy_gen - x0_pred) / sigma
                flow_tgt = critic_noise - scene_student
                loss = F.mse_loss(flow_pred.float(), flow_tgt.float())
            else:
                noise_pred = (noisy_gen - x0_pred) / sigma
                loss = F.mse_loss(noise_pred.float(), critic_noise.float())
            
            total_loss = total_loss + loss
        
        return total_loss

    def configure_optimizers(self):
        opt_cfg = getattr(self.cfg, "optimizer", None) or {}
        args = opt_cfg.get("args", {})
        base_lr = float(args.get("lr", 1e-4))
        betas = tuple(args.get("betas", (0.9, 0.999)))
        weight_decay = float(args.get("weight_decay", 0.01))
        
        # Get lr for student (transformer + connector) and fake_score separately
        params_cfg = opt_cfg.get("params", {})
        lr_student = float(params_cfg.get("student", {}).get("lr", base_lr))
        lr_critic = float(params_cfg.get("fake_score", {}).get("lr", base_lr))
        
        param_groups = [
            {"params": list(self.transformer.parameters()) + list(self.connector.parameters()), "lr": lr_student},
            {"params": list(self.fake_score.parameters()), "lr": lr_critic},
        ]
        optimizer = optim.AdamW(param_groups, lr=base_lr, betas=betas, weight_decay=weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Ensure teacher is always frozen and on CPU (use __dict__ to avoid DeepSpeed)
        if self.teacher_transformer is not None:
            self.__dict__['teacher_transformer'] = self.teacher_transformer.cpu()
            self.teacher_transformer.eval()
            for p in self.teacher_transformer.parameters():
                p.requires_grad = False
        if self.scene_teacher_transformer is not None:
            self.__dict__['scene_teacher_transformer'] = self.scene_teacher_transformer.cpu()
            self.scene_teacher_transformer.eval()
            for p in self.scene_teacher_transformer.parameters():
                p.requires_grad = False
        
        opt = self.optimizers()
        TRAIN_GENERATOR = (self.global_step % int(self.cfg.dfake_gen_update_ratio)) == 0
        # TRAIN_GENERATOR = True
        
        if TRAIN_GENERATOR:
            self._toggle_requires_grad(self.transformer, True)
            self._toggle_requires_grad(self.connector, True)
            self._toggle_requires_grad(self.fake_score, False)
            opt.zero_grad(set_to_none=True)
            g_loss = self(batch)
            self.manual_backward(g_loss)
            torch.nn.utils.clip_grad_norm_(
                list(self.transformer.parameters()) + list(self.connector.parameters()),
                max_norm=10.0,
            )
            opt.step()
            
            if (
                self.global_step >= int(getattr(self.cfg, "ema_start_step", 0))
                and self.cfg.ema_weight > 0.0
            ):
                if not self._ema_state:
                    self._init_ema()
                self._update_ema()
            
            self.log("train/g_loss", g_loss, prog_bar=True, on_step=True, on_epoch=False)
            self.check_train(batch, batch_idx)
            return {"loss": g_loss.detach()}
        else:
            self._toggle_requires_grad(self.transformer, False)
            self._toggle_requires_grad(self.connector, False)
            self._toggle_requires_grad(self.fake_score, True)
            opt.zero_grad(set_to_none=True)
            d_loss = self.compute_critic_loss(batch)
            self.manual_backward(d_loss)
            torch.nn.utils.clip_grad_norm_(self.fake_score.parameters(), max_norm=10.0)
            opt.step()
            
            self.log("train/d_loss", d_loss, prog_bar=True, on_step=True, on_epoch=False)
            self.check_train(batch, batch_idx)
            return {"loss": d_loss.detach()}

    def _save_debug_output(self, latents, name, batch_size):
        """保存latent space的tensor为ply文件用于调试"""
        import os
        import utils3d
        
        # 只在rank 0保存，避免多卡覆盖
        if hasattr(self, 'global_rank') and self.global_rank != 0:
            return
        
        # 创建debug目录
        debug_dir = "./debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        with torch.no_grad():
            # Decode latents to voxel space
            samples = self.vae.decode(latents.to(self.vae.dtype)).sample
            
            # 对每个batch保存
            for idx in range(samples.shape[0]):
                coords = torch.argwhere(samples[idx] > 0)[:, [1, 2, 3]].int()
                coords_np = (coords.cpu().numpy() + 0.5) / self.cfg.resolution - 0.5
                
                save_path = os.path.join(
                    debug_dir, 
                    f"step{self.global_step}_{name}_batch{idx}.ply"
                )
                utils3d.io.write_ply(save_path, coords_np)
    
    @torch.no_grad()
    def visualization_step(self, batch, batch_idx=0, step_name="train"):
        if "prompt" in batch:
            save_path = self.get_save_path(
                f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_prompt.txt"
            )
            with open(save_path, "w") as f:
                prompts = batch["prompt"]
                if isinstance(prompts, list):
                    f.write("\n".join(prompts))
                else:
                    f.write(str(prompts))
        
        if "voxel" in batch:
            for i in range(batch["voxel"].shape[0]):
                voxel_data = batch["voxel"][i]
                save_path = self.get_save_path(
                    f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_voxel_{i}.ply"
                )
                coords = torch.argwhere(voxel_data > 0)[:, [1, 2, 3]].int()
                coords_np = (coords.cpu().numpy() + 0.5) / self.cfg.resolution - 0.5
                utils3d.io.write_ply(save_path, coords_np)
        
        if "condition" in batch:
            for i in range(batch["condition"].shape[0]):
                condition_data = batch["condition"][i]
                save_path = self.get_save_path(
                    f"it{self.true_global_step}-{step_name}-{self.global_rank}_{batch_idx}_condition_{i}.ply"
                )
                coords = torch.argwhere(condition_data > 0)[:, [1, 2, 3]].int()
                coords_np = (coords.cpu().numpy() + 0.5) / self.cfg.resolution - 0.5
                utils3d.io.write_ply(save_path, coords_np)

    def on_check_train(self, batch, batch_idx):
        self.visualization_step(batch, batch_idx, step_name="train")

    def on_save_checkpoint(self, checkpoint):
        if self.global_rank == 0:
            training_backup = {}
            if self._ema_state:
                for name, p in self.transformer.named_parameters():
                    if name in self._ema_state:
                        training_backup[name] = p.data.clone()
                        ema_param = self._ema_state[name]
                        if getattr(self.cfg, "ema_offload_cpu", False):
                            ema_param = ema_param.to(p.device)
                        p.data.copy_(ema_param)
            
            save_dir = os.path.join(os.path.dirname(self.get_save_dir()), "pipeline")
            os.makedirs(save_dir, exist_ok=True)
            if os.path.exists(os.path.join(save_dir, "condition_encoder")):
                self.transformer.save_pretrained(os.path.join(save_dir, "transformer"))
            else:
                self.non_module("pipeline").save_pretrained(save_dir)
            self.non_module("pipeline").save_custom_adapter(
                save_dir, "custom_adapter.safetensors", safe_serialization=True
            )
            
            if training_backup:
                for name, p in self.transformer.named_parameters():
                    if name in training_backup:
                        p.data.copy_(training_backup[name])

    def _setup_inference_scheduler(self):
        pipeline = self.non_module("pipeline")
        pipeline.scheduler.config.stochastic_sampling = True
        
        if self.cfg.denoising_step_list and len(self.cfg.denoising_step_list) > 0:
            pipeline.scheduler.set_timesteps(self.cfg.sf_num_inference_steps, device=self.device)
            
            selected_timesteps = []
            selected_sigmas = []
            for idx in self.cfg.denoising_step_list:
                if idx == -1:
                    selected_sigmas.append(torch.tensor(0.0, device=self.device))
                else:
                    selected_timesteps.append(pipeline.scheduler.timesteps[idx])
                    selected_sigmas.append(pipeline.scheduler.sigmas[idx])
            
            pipeline.scheduler.timesteps = torch.stack(selected_timesteps)
            pipeline.scheduler.sigmas = torch.stack(selected_sigmas)
            pipeline.scheduler.num_inference_steps = len(selected_timesteps)
        else:
            pipeline.scheduler.set_timesteps(
                num_inference_steps=self.cfg.eval_num_inference_steps,
                device=self.device
            )

    @torch.no_grad()
    def inference(self, batch):
        # self._setup_inference_scheduler()
        pipeline = self.non_module("pipeline")
        pipeline.scheduler.config.stochastic_sampling = True
        
        # Prepare prompt - use raw prompt string if available
        if "prompt" in batch:
            prompt = batch["prompt"]
        else:
            # If only llm_features available, convert back to text (shouldn't happen in normal validation)
            prompt = " "
        pcd = pipeline(
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=len(pipeline.scheduler.timesteps),
            resolution=self.cfg.resolution,
            guidance_scale=self.cfg.eval_guidance_scale,
            condition=batch["condition"],
            target_object=batch["object"],
        ).pcds[0]
        return pcd

    def validation_step(self, batch, batch_idx):
        def points_to_dense_from_normed(points, resolution=64):
            coords = (points + 0.5) * resolution - 0.5
            coords = np.clip(coords, 0, resolution - 1).astype(int)
            grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
            grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
            return grid[None, None]
        
        if isinstance(batch, list) and all(isinstance(b, dict) for b in batch):
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

