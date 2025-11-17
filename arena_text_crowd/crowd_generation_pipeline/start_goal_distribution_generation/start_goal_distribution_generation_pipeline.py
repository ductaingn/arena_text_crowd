import os
import copy
from typing import List
from tqdm.auto import tqdm

import attrs

import numpy as np

import torch

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import DDPMScheduler, UNet2DConditionModel


@attrs.define
class StartGoalDistrGenerationPipelineConfig:
    unet_dir: str
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    revision = None
    logging_dir: str = "logs"
    tracker_project_name: str = "SgDistr"
    report_to: str = "tensorboard"
    seed: int = 0

    map_size: int = 64
    smap_channels: int = 9
    sg_distr_channels: int = 2

    mixed_precision: str = "no"

    num_inference_steps: int = 20
    guidance_scale: int = 1


@attrs.define
class StartGoalDistrGenerationPipeline:
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    scheduler: DDPMScheduler
    unet: UNet2DConditionModel
    config: StartGoalDistrGenerationPipelineConfig

    @staticmethod
    def add_background_class(init_sem_map):
        new_added_dim = np.zeros((init_sem_map.shape[0], init_sem_map.shape[1], 1))
        bg_idx = np.argwhere(np.sum(np.array(init_sem_map), axis=2) == 0)
        new_added_dim[bg_idx[:, 0], bg_idx[:, 1]] = np.array([1])
        return np.concatenate((np.array(init_sem_map), new_added_dim), axis=2)

    def inference(
        self,
        smaps,
        prompts,
        num_inference_steps,
        guidance_scale,
        save_path=None,
        show=False,
    ):
        assert self.scheduler.config.prediction_type == "epsilon"
        if save_path is not None and not os.path.exists(save_path):
            os.mkdir(save_path)
        torch_device = "cuda"
        weight_dtype = torch.float32
        if self.config.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.config.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.text_encoder.to(torch_device, dtype=weight_dtype)
        self.unet.to(torch_device, dtype=weight_dtype)

        prompts_ = prompts

        smaps_full = [self.add_background_class(np.array(smap)) for smap in smaps]
        smaps_ = torch.from_numpy(np.array(smaps_full)).to(
            torch_device, dtype=weight_dtype
        )
        smaps_ = torch.permute(smaps_, (0, 3, 1, 2))

        generator = torch.Generator().manual_seed(self.config.seed)
        batch_size_ = len(prompts_)

        text_input = self.tokenizer(
            prompts_,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(torch_device))[
                0
            ]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size_,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(torch_device))[
            0
        ]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (
                batch_size_,
                self.config.sg_distr_channels,
                self.config.map_size,
                self.config.map_size,
            ),
            generator=generator,
        )
        latents = latents.to(torch_device, dtype=weight_dtype)
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat((latents, smaps_), dim=1)
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latent_model_input] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        sgdistrs_all = []
        for dt_id in range(batch_size_):
            sg_distr_i = np.array(
                torch.permute(latents[dt_id].clone().detach().cpu(), (1, 2, 0))
            )
            sgdistrs_all.append(copy.deepcopy(sg_distr_i))

        return np.array(sgdistrs_all)
