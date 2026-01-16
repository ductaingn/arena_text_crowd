import os
import copy
from typing import List
from tqdm.auto import tqdm

import attrs

import cv2

import numpy as np

import torch

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import UNet2DConditionModel, DDPMScheduler

from ..utils import cv_visual_map, cv_visual_field


@attrs.define
class VelocityFieldGenerationPipelineConfig:
    unet_dir: str
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    revision = None
    logging_dir: str = "logs"
    tracker_project_name: str = "Field"
    report_to: str = "tensorboard"
    seed: int = 0

    map_size: int = 64
    smap_channels: int = 9
    sg_distr_channels: int = 2
    field_channels: int = 2

    mixed_precision: str = "no"

    num_inference_steps: int = 20
    guidance_scale: int = 1


@attrs.define
class VelocityFieldGenerationPipeline:
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    scheduler: DDPMScheduler
    unet: UNet2DConditionModel
    config: VelocityFieldGenerationPipelineConfig

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
        sg_distrs,
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

        smaps_full = []
        for smp_id in range(len(smaps)):
            smaps_full.append(self.add_background_class(np.array(smaps[smp_id])))
        smaps_ = torch.from_numpy(np.array(smaps_full)).to(
            torch_device, dtype=weight_dtype
        )
        smaps_ = torch.permute(smaps_, (0, 3, 1, 2))
        sg_distrs_ = torch.from_numpy(np.array(sg_distrs)).to(
            torch_device, dtype=weight_dtype
        )
        sg_distrs_ = torch.permute(sg_distrs_, (0, 3, 1, 2))

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
                self.config.field_channels,
                self.config.map_size,
                self.config.map_size,
            ),
            generator=generator,
        )
        latents = latents.to(torch_device, dtype=weight_dtype)
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat((latents, smaps_, sg_distrs_), dim=1)
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

        fields_all = []
        for dt_id in range(batch_size_):
            smap_i = np.array(
                torch.permute(smaps_[dt_id].clone().detach().cpu(), (1, 2, 0))
            )
            sg_distr_i = np.array(
                torch.permute(sg_distrs_[dt_id].clone().detach().cpu(), (1, 2, 0))
            )
            field_i = np.array(
                torch.permute(latents[dt_id].clone().detach().cpu(), (1, 2, 0))
            )
            fields_all.append(copy.deepcopy(field_i))
            text_i = prompts_[dt_id]

            sg_colors = np.array([[0, 255, 0], [0, 0, 255]])
            bg_color = np.array([[0, 0, 0]])
            map_colors = np.concatenate(
                [
                    np.random.randint(0, 255, (len(smap_i[0][0]) - 3, 3)),
                    sg_colors,
                    bg_color,
                ],
                axis=0,
            )

            smap_cvimg = cv_visual_map(
                smap_i, colors=map_colors, save_nm=None, show=False
            )
            distr_cvimg = cv_visual_map(
                sg_distr_i, colors=sg_colors, save_nm=None, show=False
            )
            field_cvimg = cv_visual_field(
                field_i, grid_width=10, save_nm=None, show=False
            )

            enlarge_scale = int(len(field_cvimg) / len(smap_cvimg))
            smap_large = cv2.resize(
                smap_cvimg,
                None,
                fx=enlarge_scale,
                fy=enlarge_scale,
                interpolation=cv2.INTER_CUBIC,
            )
            distr_large = cv2.resize(
                distr_cvimg,
                None,
                fx=enlarge_scale,
                fy=enlarge_scale,
                interpolation=cv2.INTER_CUBIC,
            )

            interval = np.ones((len(smap_large), 20, 3)) * 255
            cat_img = cv2.hconcat(
                [
                    interval,
                    smap_large,
                    interval,
                    distr_large,
                    interval,
                    field_cvimg,
                    interval,
                ]
            )

            if show:
                print(text_i)
                cv2.imshow("img", cat_img / 255)
                cv2.waitKey(0)
            if save_path is not None:
                cv2.imwrite(
                    os.path.join(save_path, "dt_" + str(dt_id) + ".jpg"), cat_img
                )
        if save_path is not None:
            np.save(os.path.join(save_path, "texts.npy"), {"texts": prompts_})

        return np.array(fields_all)


if __name__ == "__main__":
    from pathlib import Path
    from arena_simulation_setup.tree.World import World

    from arena_text_crowd.crowd_generation_pipeline.input_models.prompt import (
        PromptCanonicalizer,
    )
    from arena_text_crowd.converters import arena_world_to_text_crowd_scenario

    # Create Text-Crowd scenario from Arena World
    world_path = Path(
        "/home/linh/ductai_nguyen_ws/Arena_ws/install/arena_simulation_setup/share/arena_simulation_setup/worlds/hospital_1"
    )
    arena_world = World(path=world_path)
    scenario = arena_world_to_text_crowd_scenario(
        arena_world=arena_world, scenario_size=(1024, 1024), wall_thickness=1.0
    )

    # Initialize velocity generation model
    vel_field_gen_config = VelocityFieldGenerationPipelineConfig(
        unet_dir="/home/linh/ductai_nguyen_ws/Text-Crowd/text_crowd/Language_Crowd_Animation/Models_Server_ForTest/Field-Full-V2/checkpoint-270000/unet"
    )
    vel_field_text_encoder = CLIPTextModel.from_pretrained(
        vel_field_gen_config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=vel_field_gen_config.revision,
        use_safetensors=True,
    )
    sg_tokenizer = CLIPTokenizer.from_pretrained(
        vel_field_gen_config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=vel_field_gen_config.revision,
    )
    sg_noise_scheduler = DDPMScheduler.from_pretrained(
        vel_field_gen_config.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    sg_unet = UNet2DConditionModel.from_pretrained(
        vel_field_gen_config.unet_dir,
        subfolder="unet",
        use_safetensors=True,
    )
    sg_unet.set_attention_slice("max")

    vel_field_gen_pipeline = VelocityFieldGenerationPipeline(
        text_encoder=vel_field_text_encoder,
        tokenizer=sg_tokenizer,
        scheduler=sg_noise_scheduler,
        unet=sg_unet,
        config=vel_field_gen_config,
    )

    prompt = "Two groups enter from the main entrance in the top right corner and both of them walk through the upper right passage first. Afterward, one group leaves at the bottom left exit. Another group follows a different path where they  exit through the bottom gate"
    prompt_canonicalizer = PromptCanonicalizer()
    canonicalized_descriptions = prompt_canonicalizer.canonicalize(prompt)
    group_sizes = prompt_canonicalizer.get_group_size(canonicalized_descriptions)
    group_n = len(group_sizes)

    semantic_map = scenario.get_semantic_map()
    smaps = []
    for _ in range(group_n):
        smaps.append(copy.deepcopy(semantic_map))

    pred_group_sgdistrs = torch.randn(size=(1, 64, 64, 2))

    pred_group_fields = vel_field_gen_pipeline.inference(
        smaps=np.array(smaps),
        prompts=canonicalized_descriptions,
        sg_distrs=copy.deepcopy(pred_group_sgdistrs),
        num_inference_steps=vel_field_gen_config.num_inference_steps,
        guidance_scale=vel_field_gen_config.guidance_scale,
        save_path=None,
        show=True,
    )
    print("Output shape: ", pred_group_fields.shape)
