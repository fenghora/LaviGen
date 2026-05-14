import inspect
import math
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL
import PIL.Image
import torch
import trimesh
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders.lora_pipeline import QwenImageLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from transformers import Qwen2Tokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

from ..utils.custom_adapter import CustomAdapterMixin
from ..models.autoencoder_kl_voxel import AutoencoderKLVoxel
from ..models.transformer_qwenvoxel_layout_synthesis import QwenVoxelTransformer3DModelLayoutSynthesis
from ..utils.qwen_vision_process import fetch_image
from .pipeline_utils import TransformerDiffusionMixin
from .pipeline_shapediff_output import SparseStructurePipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class QwenVoxelPipelineLayoutSynthesis(
    DiffusionPipeline,
    TransformerDiffusionMixin,
    QwenImageLoraLoaderMixin,
    CustomAdapterMixin,
):
    """
    Pipeline for sparse structure generation.
    """

    model_cpu_offload_seq = "condition_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLVoxel,
        transformer: QwenVoxelTransformer3DModelLayoutSynthesis,
        scheduler: FlowMatchEulerDiscreteScheduler,
        condition_encoder: Qwen2_5_VLForConditionalGeneration,
        feature_extractor: Qwen2_5_VLProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            condition_encoder=condition_encoder,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.decoder_channels) - 1)
        self.patch_size = self.transformer.config.patch_size

        self.prompt_template = "<|im_start|>system\nYou are a helpful and creative assistant for generating 3D models. Your task is to analyze the user's input, which may include text, an image, or both. Your goal is to provide a comprehensive, detailed, and imaginative description for creating a 3D asset.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_start_idx = 64
        self.prompt_max_length = 1024

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(self, prompt, image, device):
        device = device or self._execution_device
        dtype = next(self.condition_encoder.parameters()).dtype

        prompt_template = self.prompt_template
        drop_idx = self.prompt_template_start_idx

        if isinstance(prompt, BatchFeature):
            inputs = prompt
        else:
            prompt = prompt if prompt is not None else ""
            prompt = [prompt] if isinstance(prompt, str) else prompt

            if image is not None:
                for i, p in enumerate(prompt):
                    prompt[i] = f"<|vision_start|><|image_pad|><|vision_end|>{p}"
                image_inputs = [fetch_image({"image": image})]
            else:
                image_inputs = None

            text_input = [f"{prompt_template.format(p)}" for p in prompt]
            inputs = self.feature_extractor(
                text=text_input,
                images=image_inputs,
                max_length=self.prompt_max_length + drop_idx,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        inputs = inputs.to(device=device, dtype=dtype)
        encoder_hidden_states = self.condition_encoder(
            **inputs, output_hidden_states=True
        )
        prompt_embeds = encoder_hidden_states.hidden_states[-1]

        # Prepare attention mask
        prompt_embeds = prompt_embeds[:, drop_idx:]
        attention_mask = inputs.attention_mask[:, drop_idx:]

        if hasattr(self, "connector"):
            # Convert to connector's dtype to avoid dtype mismatch
            # (condition_encoder may output Float32 while connector uses BFloat16)
            connector_dtype = next(self.connector.parameters()).dtype
            prompt_embeds = self.connector(prompt_embeds.to(connector_dtype))

        return prompt_embeds, attention_mask

    def encode_prompt(
        self,
        prompt,
        image,
        device,
        num_shapes_per_prompt,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        device = device or self._execution_device

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt, image, device
            )

        # NOTE not used for now
        # prompt_embeds = prompt_embeds[:, :max_sequence_length]
        # prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        prompt_embeds = prompt_embeds.repeat_interleave(num_shapes_per_prompt, dim=0)
        prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(
            num_shapes_per_prompt, dim=0
        )

        return prompt_embeds, prompt_embeds_mask

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        resolution,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 4x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        resolution = 2 * (int(resolution) // (self.vae_scale_factor * 2))

        shape = (
            batch_size,
            1,
            num_channels_latents,
            resolution,
            resolution,
            resolution,
        )

        if latents is not None:
            return latents.to(device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, resolution, self.patch_size
        )

        return latents

    @staticmethod
    def _pack_latents(
        latents, batch_size, num_channels_latents, resolution, patch_size
    ):
        latents = latents.view(
            batch_size,
            num_channels_latents,
            resolution // patch_size,
            patch_size,
            resolution // patch_size,
            patch_size,
            resolution // patch_size,
            patch_size,
        )

        # Pack latents into a single dimension
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
        latents = latents.reshape(
            batch_size,
            (resolution // patch_size) ** 3,
            num_channels_latents * patch_size**3,
        )
        return latents

    @staticmethod
    def _unpack_latents(latents, resolution, patch_size, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 4x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        resolution = 2 * (int(resolution) // (vae_scale_factor * 2))

        latents = latents.view(
            batch_size,
            resolution // patch_size,
            resolution // patch_size,
            resolution // patch_size,
            channels // patch_size**3,
            patch_size,
            patch_size,
            patch_size,
        )
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        latents = latents.reshape(
            batch_size, channels // (patch_size**3), resolution, resolution, resolution
        )

        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Any] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        resolution: int = 64,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "pcd",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        # -----------------------------------------
        condition = None,
        target_object = None
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if image is None and prompt is None:
            raise ValueError("Either image or prompt must be provided")

        if prompt is not None:  # use prompt first
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, torch.Tensor):
                batch_size = prompt.shape[0]
            elif isinstance(prompt, BatchFeature):
                batch_size = prompt["input_ids"].shape[0]
            else:
                raise ValueError("Invalid input type for prompt")
        else:
            if isinstance(image, PIL.Image.Image):
                batch_size = 1
            elif isinstance(image, list):
                batch_size = len(image)
            elif isinstance(image, torch.Tensor):
                batch_size = image.shape[0]
            elif isinstance(image, BatchFeature):
                batch_size = image["input_ids"].shape[0]
            else:
                raise ValueError("Invalid input type for image")

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None

        if self.do_classifier_free_guidance and not has_neg_prompt:
            logger.warning(
                f"guidance scale is passed as {guidance_scale}, but classifier-free guidance is not enabled since no negative prompt is provided."
            )
        elif not self.do_classifier_free_guidance and has_neg_prompt:
            logger.warning(
                f"negative prompt is provided, but classifier-free guidance is not enabled."
            )

        do_true_cfg = self.do_classifier_free_guidance and has_neg_prompt

        # 3. Encode condition
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt,
            image,
            device,
            num_shapes_per_prompt,
            prompt_embeds,
            prompt_embeds_mask,
        )

        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                negative_prompt,
                None,
                device,
                num_shapes_per_prompt,  # Now neg image is not supported
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // (
            self.patch_size**3
        )
        latents = self.prepare_latents(
            batch_size * num_shapes_per_prompt,
            num_channels_latents,
            resolution,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        ############################################################
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            condition = self.vae.encode(condition.to(self.vae.dtype)).latent_dist.sample()
            target_object = self.vae.encode(target_object.to(self.vae.dtype)).latent_dist.sample()

        condition = self._pack_latents(
            condition,
            batch_size=batch_size * num_shapes_per_prompt,
            num_channels_latents=num_channels_latents,
            resolution=2 * (int(resolution) // (self.vae_scale_factor * 2)),
            patch_size=self.patch_size,
        )

        target_object = self._pack_latents(
            target_object,
            batch_size=batch_size * num_shapes_per_prompt,
            num_channels_latents=num_channels_latents,
            resolution=2 * (int(resolution) // (self.vae_scale_factor * 2)),
            patch_size=self.patch_size,
        )
        ############################################################

        # trible pos emb
        vox_shapes = [
            [
                (
                    3,
                    resolution // self.vae_scale_factor // self.patch_size,
                    resolution // self.vae_scale_factor // self.patch_size,
                    resolution // self.vae_scale_factor // self.patch_size,
                )
            ]
        ] * batch_size
        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else None
        )
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if negative_prompt_embeds_mask is not None
            else None
        )

        ##########################sf##########################
        # self.scheduler.config.stochastic_sampling = True
        # timesteps = torch.tensor([1000.0, 935.0, 833.3, 635]).to(device)
        # num_inference_steps = len(timesteps)
        ##########################sf##########################

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0])
                timestep = timestep / self.scheduler.config.num_train_timesteps

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        timestep=timestep,
                        vox_shapes=vox_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        ######################################################
                        condition=condition,
                        target_object=target_object
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            timestep=timestep,
                            vox_shapes=vox_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            ######################################################
                            condition=condition,
                            target_object=target_object
                        )[0]

                    noise_pred = neg_noise_pred + self.guidance_scale * (
                        noise_pred - neg_noise_pred
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            output = latents
            coords = None
        else:
            # 7. decode sparse structure
            latents = self._unpack_latents(
                latents, resolution, self.patch_size, self.vae_scale_factor
            )
            samples = self.vae.decode(latents).sample

            output = []
            for idx in range(samples.shape[0]):
                coords = torch.argwhere(samples[idx] > 0)[:, [1, 2, 3]].int()
                coords_np = (
                    coords.cpu().numpy() + 0.5
                ) / resolution - 0.5  # Why Trellis do it?
                output.append(coords_np)

            coords = torch.argwhere(samples > 0)[:, [0, 2, 3, 4]].int()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, output, coords)

        return SparseStructurePipelineOutput(samples=output, pcds=output, coords=coords)

    def _init_custom_adapter(self):
        # Initialize connector for transformer
        joint_attention_dim = self.transformer.config.joint_attention_dim
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(
                self.condition_encoder.config.hidden_size,
                joint_attention_dim,
            ),
            torch.nn.LayerNorm(joint_attention_dim),
        ).to(self.device, self.dtype)

    def _load_custom_adapter(self, state_dict):
        # Load connector for transformer
        connector_state_dict = {
            k.replace("connector.", ""): v
            for k, v in state_dict.items()
            if k.startswith("connector.")
        }
        self.connector.load_state_dict(connector_state_dict)

    def _save_custom_adapter(self):
        state_dict = {}
        # Save connector for transformer
        state_dict.update(
            {f"connector.{k}": v for k, v in self.connector.state_dict().items()}
        )

        return state_dict
