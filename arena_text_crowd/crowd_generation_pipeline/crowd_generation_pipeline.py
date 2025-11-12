import copy
from typing import List
import random

import attrs

import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import DDPMScheduler, UNet2DConditionModel

from .start_goal_distribution_generation.start_goal_distribution_generation_pipeline import (
    StartGoalDistrGenerationPipeline as SGDGP,
    StartGoalDistrGenerationPipelineConfig as SGDGPConfig,
)
from .velocity_field_generation.velocity_field_generation_pipeline import (
    VelocityFieldGenerationPipeline as VFGP,
    VelocityFieldGenerationPipelineConfig as VFGPConfig,
)
from .utils.field import Field
from .utils.utils import collision_checker, get_box_ll
from .input_models.scenario import Scenario
from .input_models.prompt import PromptCanonicalizer
from .input_models.semantic.semantic_object import Rectangle
from .input_models.constants import AllSemanticObjects
from .simulator.agent import Agent
from .simulator.field_env import FieldEnv, GroupField


@attrs.define
class CrowdGenerationPipelineConfig:
    # dataset
    data_path: str = "./Dataset/Data_Full_V2/"
    obj_nums: List = [0, 1, 2, 3, 4, 5]
    group_nums: List = [1, 2, 3]
    data_scale: float = 1.0

    # 2d sim
    visual: bool = True
    warm_up_steps: int = 20
    agent_radius: int = 5
    agent_prefV: int = 4
    max_path_len_ratio: float = 1.5
    pos_rand_range: int = 5
    reach_dis: int = 60

    # metrics
    path_bound: int = 80
    strict_agent_success_ratio: float = 0.7
    strict_group_success_ratio: float = 0.8
    relaxed_agent_success_ratio: float = 0.5
    relaxed_group_success_ratio: float = 0.7

    use_complete_text: bool = True


@attrs.define
class CrowdGenerationPipeline:
    generation_pipeline_config: CrowdGenerationPipelineConfig
    sg_distr_gen_config: SGDGPConfig
    vel_field_gen_config: VFGPConfig

    sg_distr_gen_pipeline: SGDGP = attrs.field(init=False)
    vel_field_gen_pipeline: VFGP = attrs.field(init=False)
    canonicalizer: PromptCanonicalizer = attrs.field(
        init=False, default=PromptCanonicalizer()
    )

    @sg_distr_gen_pipeline.default
    def _sg_distr_gen_pipeline_factory(self):
        print("Loading models for start-goal-distribution generation...")

        sg_text_encoder = CLIPTextModel.from_pretrained(
            self.sg_distr_gen_config.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.sg_distr_gen_config.revision,
            use_safetensors=True,
        )
        sg_tokenizer = CLIPTokenizer.from_pretrained(
            self.sg_distr_gen_config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.sg_distr_gen_config.revision,
        )
        sg_noise_scheduler = DDPMScheduler.from_pretrained(
            self.sg_distr_gen_config.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        sg_unet = UNet2DConditionModel.from_pretrained(
            self.sg_distr_gen_config.output_dir, subfolder="unet", use_safetensors=True
        )
        sg_unet.set_attention_slice("max")

        return SGDGP(
            text_encoder=sg_text_encoder,
            tokenizer=sg_tokenizer,
            scheduler=sg_noise_scheduler,
            unet=sg_unet,
            config=self.sg_distr_gen_config,
        )

    @vel_field_gen_pipeline.default
    def _vel_field_gen_pipeline_factory(self):
        print("Loading models for field generation...")

        vel_field_text_encoder = CLIPTextModel.from_pretrained(
            self.vel_field_gen_config.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.vel_field_gen_config.revision,
            use_safetensors=True,
        )
        sg_tokenizer = CLIPTokenizer.from_pretrained(
            self.vel_field_gen_config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.vel_field_gen_config.revision,
        )
        sg_noise_scheduler = DDPMScheduler.from_pretrained(
            self.vel_field_gen_config.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        sg_unet = UNet2DConditionModel.from_pretrained(
            self.vel_field_gen_config.output_dir,
            subfolder="unet",
            use_safetensors=True,
        )
        sg_unet.set_attention_slice("max")

        return VFGP(
            text_encoder=vel_field_text_encoder,
            tokenizer=sg_tokenizer,
            scheduler=sg_noise_scheduler,
            unet=sg_unet,
            config=self.vel_field_gen_config,
        )

    def get_canonicalized_des(self, prompt: str):
        """
        Get the canonicalized description text of groups
        """
        return self.canonicalizer.canonicalize(prompt)

    def get_group_size(self, canonicalized_des: str):
        return self.canonicalizer.get_group_size(canonicalized_des)

    def generate(self, scenario: Scenario, prompt: str):
        semantic_map = scenario.get_semantic_map()
        canonicalized_descriptions = self.get_canonicalized_des(prompt)
        group_sizes = self.get_group_size(canonicalized_descriptions)
        group_n = len(group_sizes)

        print("Inferring start and goal distributions...")
        pred_group_sgdistrs = self.sg_distr_gen_pipeline.inference(
            smaps=copy.deepcopy(np.array(semantic_map)),
            prompts=copy.deepcopy(canonicalized_descriptions),
            num_inference_steps=self.sg_distr_gen_config.num_inference_steps,
            guidance_scale=self.sg_distr_gen_config.guidance_scale,
            save_path=None,
            show=False,
        )
        pred_group_sgdistrs = np.clip(pred_group_sgdistrs, a_min=0.0, a_max=1.0)
        pred_group_sgdistrs[pred_group_sgdistrs < 0.2] = 0.0
        pred_group_sgdistrs[pred_group_sgdistrs > 0.8] = 1.0

        print("Inferring fields...")
        pred_group_fields = self.vel_field_gen_pipeline.inference(
            smaps=copy.deepcopy(np.array(semantic_map)),
            prompts=copy.deepcopy(canonicalized_descriptions),
            sg_distrs=copy.deepcopy(pred_group_sgdistrs),
            num_inference_steps=self.vel_field_gen_config.num_inference_steps,
            guidance_scale=self.vel_field_gen_config.guidance_scale,
            save_path=None,
            show=False,
        )

        # normalize the fields and remove vectors in obstacle
        grid_size = [
            self.vel_field_gen_config.map_size,
            self.vel_field_gen_config.map_size,
        ]
        grid_width = semantic_map.window_size[0] / grid_size[0]
        base_field = Field(scenario=..., grid_width=grid_width)
        grid_map = base_field.grid.grid_map
        obs_coords = np.argwhere(grid_map == 1)
        for group_id in range(group_n):
            pred_group_fields[group_id] = np.nan_to_num(
                pred_group_fields[group_id]
                / (
                    np.linalg.norm(pred_group_fields[group_id], axis=2).reshape(
                        (grid_size[0], grid_size[1], 1)
                    )
                ),
                0,
            )
            pred_group_fields[group_id][obs_coords[:, 0], obs_coords[:, 1]] = np.array(
                [0.0, 0.0]
            )

        # do simulation
        return self.get_agent_trajectories(
            scenario=scenario,
            group_distrs=pred_group_sgdistrs,
            group_sizes=group_sizes,
            group_fields=pred_group_fields,
        )

    def get_agent_trajectories(
        self,
        scenario: Scenario,
        group_distrs,
        group_sizes,
        group_fields,
    ):
        agent_n = np.sum(np.array(group_sizes))
        agent_radius = self.generation_pipeline_config.agent_radius
        agent_prefV = self.generation_pipeline_config.agent_prefV
        group_n = len(group_sizes)
        groups_colors = [
            [255, 128, 0],
            [135, 200, 240],
            [0, 255, 0],
            [124, 79, 13],
            [255, 192, 203],
            [128, 0, 128],
        ][
            0:group_n
        ]  # TODO: Remove

        grid_width = scenario["wind_size"][0] / len(group_fields[0])
        base_field = Field(scenario, grid_width)

        def sample_poses(
            current_poses,
            safe_dis,
            distr,
            grid_width,
            rand_buffer,
            boundary,
            obs_list,
            sample_num,
        ):
            distr_1d = np.array(distr).reshape(1, -1)[0]
            pos_list = []
            for smp_id in range(sample_num):
                while 1:
                    pos_loc = random.choices(
                        list(range(len(distr_1d))), weights=distr_1d, k=1
                    )[0]
                    pos = [
                        int(pos_loc / len(distr)) * grid_width
                        + grid_width / 2
                        + random.uniform(-rand_buffer, rand_buffer),
                        int(pos_loc % len(distr)) * grid_width
                        + grid_width / 2
                        + random.uniform(-rand_buffer, rand_buffer),
                    ]
                    pos[0] = 0 if pos[0] < 0 else pos[0]
                    pos[0] = boundary[0] if pos[0] > boundary[0] else pos[0]
                    pos[1] = 0 if pos[1] < 0 else pos[1]
                    pos[1] = boundary[1] if pos[1] > boundary[1] else pos[1]
                    if len(current_poses) == 0:
                        distances = np.array([np.inf])
                    else:
                        distances = np.linalg.norm(
                            np.array(current_poses) - np.array(pos), axis=1
                        )
                    if len(pos_list) == 0:
                        distances_ = np.array([np.inf])
                    else:
                        distances_ = np.linalg.norm(
                            np.array(pos_list) - np.array(pos), axis=1
                        )
                    if (
                        np.min(distances) >= safe_dis
                        and np.min(distances_) >= safe_dis
                        and collision_checker(pos=pos, obs_list=obs_list, buf=safe_dis)
                    ):
                        pos_list.append(pos)
                        break
            return pos_list

        # set group fields
        gfields_for_ctrl: List[GroupField] = []
        tp = 0
        for gid in range(group_n):
            gfields_for_ctrl.append(
                GroupField(
                    agent_ids=list(range(tp, tp + group_sizes[gid])),
                    field=group_fields[gid],
                    grid=base_field.grid,
                )
            )
            tp += group_sizes[gid]

        # Set all agents' params and reset the scenario
        # Add boundary to scenario_bound
        scenario_bound = copy.deepcopy(scenario)
        window_size = copy.deepcopy(scenario_bound.scenario_config.window_size)
        thick = 50
        scenario_bound.obstacle_dict[AllSemanticObjects.RECTANGLE].append(
            Rectangle(
                width=window_size[0] + thick * 2,
                height=thick,
                vertexes=get_box_ll(
                    x=window_size[0] + thick * 2,
                    y=thick,
                    lowerleft=(-thick, -thick),
                ),
            )
        )
        scenario_bound.obstacle_dict[AllSemanticObjects.RECTANGLE].append(
            Rectangle(
                width=window_size[0] + thick * 2,
                height=thick,
                vertexes=get_box_ll(
                    x=window_size[0] + thick * 2,
                    y=thick,
                    lowerleft=(-thick, window_size[1]),
                ),
            )
        )
        scenario_bound.obstacle_dict[AllSemanticObjects.RECTANGLE].append(
            Rectangle(
                width=thick,
                height=window_size[1] + thick * 2,
                vertexes=get_box_ll(
                    x=thick,
                    y=window_size[1] + thick * 2,
                    lowerleft=(-thick, -thick),
                ),
            )
        )
        scenario_bound.obstacle_dict[AllSemanticObjects.RECTANGLE].append(
            Rectangle(
                width=thick,
                height=window_size[1] + thick * 2,
                vertexes=get_box_ll(
                    x=thick,
                    y=window_size[1] + thick * 2,
                    lowerleft=(window_size[0], -thick),
                ),
            )
        )
        all_obs = scenario_bound.get_all_obstacles()

        agent_list: List[Agent] = []
        for gid in range(group_n):
            add_agent_n = group_sizes[gid]
            agent_pos_list = sample_poses(
                [],
                safe_dis=0,
                distr=group_distrs[gid][:, :, 0],
                grid_width=grid_width,
                rand_buffer=self.generation_pipeline_config.pos_rand_range,
                boundary=scenario.scenario_config.window_size,
                obs_list=copy.deepcopy(all_obs),
                sample_num=add_agent_n,
            )
            agent_goal_list = sample_poses(
                [],
                safe_dis=0,
                distr=group_distrs[gid][:, :, 1],
                grid_width=grid_width,
                rand_buffer=self.generation_pipeline_config.pos_rand_range,
                boundary=scenario.scenario_config.window_size,
                obs_list=copy.deepcopy(all_obs),
                sample_num=add_agent_n,
            )
            for idx in range(add_agent_n):
                agent = Agent(
                    pos=agent_pos_list[idx],
                    goal_pos=agent_goal_list[idx],
                    radius=agent_radius,
                    pref_speed=agent_prefV,
                    color=groups_colors[gid],
                )
                agent_list.append(copy.deepcopy(agent))

        ### Start simulation
        fld_env = FieldEnv(scenario=scenario, agent_num=agent_n)
        fld_env.reset(scenario=copy.deepcopy(scenario_bound))

        # warm up
        for _ in range(self.generation_pipeline_config.warm_up_steps):
            fld_env.perform_action_ORCAEnv(np.zeros((agent_n, 2)).tolist())
        for agt_id in range(agent_n):
            fld_env.agent_dict[agt_id].traj_history = fld_env.agent_dict[
                agt_id
            ].traj_history[: -self.generation_pipeline_config.warm_up_steps]
            assert len(fld_env.agent_dict[agt_id].traj_history) == 0

        step = 0
        all_agent_trajs = [[] * group_n]
        removed_agent_n = 0
        while True:
            # remove agents that reach the goal or get out of the scenario, and record its trajectory
            for gid in range(group_n):
                for aid in gfields_for_ctrl[gid].agent_ids:
                    if (
                        np.linalg.norm(
                            np.array(fld_env.agent_dict[aid].pos)
                            - np.array(fld_env.agent_dict[aid].goal_pos)
                        )
                        < self.generation_pipeline_config.reach_dis
                        or fld_env.agent_dict[aid].pos[0] <= 0
                        or fld_env.agent_dict[aid].pos[0]
                        >= scenario.scenario_config.window_size[0]
                        or fld_env.agent_dict[aid].pos[1] <= 0
                        or fld_env.agent_dict[aid].pos[1]
                        >= scenario.scenario_config.window_size[1]
                    ):
                        all_agent_trajs[gid].append(
                            {
                                "agent_id": aid,
                                "agent_trajs": np.array(
                                    fld_env.agent_dict[aid].traj_history
                                )[:, 0, :],
                            }
                        )
                        fld_env.set_agent_position(aid, [-1e5, -1e5])
                        gfields_for_ctrl[gid]["agent_ids"].remove(aid)
                        removed_agent_n += 1

            fld_env.perform_action_fast(gfields_for_ctrl)

            step += 1
            if removed_agent_n >= agent_n:
                break

        if self.generation_pipeline_config.visual and not fld_env.viewer.closed:
            fld_env.viewer.close()

        # handle the rest agents
        for gid in range(group_n):
            for aid in gfields_for_ctrl[gid].agent_ids:
                all_agent_trajs[gid].append(
                    {
                        "agent_id": aid,
                        "agent_trajs": np.array(fld_env.agent_dict[aid].traj_history)[
                            :, 0, :
                        ],
                    }
                )

        return all_agent_trajs
