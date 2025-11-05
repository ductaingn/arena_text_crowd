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

from .

from .utils.utils import collision_checker, get_box


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

    def generate(self, semantic_map, prompt):
        print("Inferring start and goal distributions...")
        pred_group_sgdistrs = self.sg_distr_gen_pipeline.inference(
            smaps=copy.deepcopy(np.array(semantic_map)),
            prompts=copy.deepcopy(prompt),
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
            prompts=copy.deepcopy(prompt),
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
        grid_width = gt_scenario["wind_size"][0] / grid_size[0]
        base_field = Base_Field()
        base_field.reset(scenario=copy.deepcopy(gt_scenario), grid_width=grid_width)
        grid_map = base_field.grid["grid_map"]
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
        return self.Sim2D_WithField(
            scenario=copy.deepcopy(gt_scenario),
            group_distrs=pred_group_sgdistrs,
            group_sizes=gt_group_sizes,
            group_fields=pred_group_fields,
            group_paths=gt_group_paths,
            show_fields=False,
            show_paths=True,
            record_video_path=None,
            save_sim_path=None,
        )

    def Sim2D_WithField(
        self,
        scenario,
        group_distrs,
        group_sizes,
        group_fields,
        group_paths,
        show_fields=True,
        show_paths=True,
        record_video_path=None,
        save_sim_path=None,
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
        ][0:group_n]

        grid_width = scenario["wind_size"][0] / len(group_fields[0])
        grid_size = [len(group_fields[0]), len(group_fields[0][0])]
        base_field = Base_Field()
        base_field.reset(scenario, grid_width)
        # visualize the fields
        if self.generation_pipeline_config.visual and show_fields:
            for grp_id in range(group_n):
                base_field.field_visualization(
                    field=group_fields[grp_id], guidance=None
                )

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
        gfields_for_ctrl = []
        tp = 0
        for gid in range(group_n):
            gfields_for_ctrl.append(
                {
                    "agent_ids": list(range(tp, tp + group_sizes[gid])),
                    "field": copy.deepcopy(group_fields[gid]),
                    "grid": copy.deepcopy(base_field.grid),
                }
            )
            tp += group_sizes[gid]

        # set all agents' params and reset the scenario
        fld_env = Field_Env(
            agent_num=agent_n,
            visual=self.generation_pipeline_config.visual,
            draw_scale=1.0,
        )
        # Add boundary to scenario_bound
        scenario_bound = copy.deepcopy(scenario)
        if "obs_list" not in scenario_bound.keys():
            scenario_bound["obs_list"] = []
        wind_size = copy.deepcopy(scenario_bound["wind_size"])
        thick = 50
        scenario_bound["obs_list"].append(
            {
                "type": "rectangle",
                "params": {
                    "vertexes": get_box_ll(
                        x=wind_size[0] + thick * 2, y=thick, lowerleft=(-thick, -thick)
                    )
                },
                "attributes": {},
            }
        )
        scenario_bound["obs_list"].append(
            {
                "type": "rectangle",
                "params": {
                    "vertexes": get_box_ll(
                        x=wind_size[0] + thick * 2,
                        y=thick,
                        lowerleft=(-thick, wind_size[1]),
                    )
                },
                "attributes": {},
            }
        )
        scenario_bound["obs_list"].append(
            {
                "type": "rectangle",
                "params": {
                    "vertexes": get_box_ll(
                        x=thick, y=wind_size[1] + thick * 2, lowerleft=(-thick, -thick)
                    )
                },
                "attributes": {},
            }
        )
        scenario_bound["obs_list"].append(
            {
                "type": "rectangle",
                "params": {
                    "vertexes": get_box_ll(
                        x=thick,
                        y=wind_size[1] + thick * 2,
                        lowerleft=(wind_size[0], -thick),
                    )
                },
                "attributes": {},
            }
        )
        all_obs = fld_env.get_all_obstacles_from_scenario(copy.deepcopy(scenario_bound))

        agent_params = {"init_agent_params": []}
        for gid in range(group_n):
            add_agent_n = group_sizes[gid]
            agent_pos_list = sample_poses(
                [],
                safe_dis=0,
                distr=group_distrs[gid][:, :, 0],
                grid_width=grid_width,
                rand_buffer=self.generation_pipeline_config.pos_rand_range,
                boundary=scenario["wind_size"],
                obs_list=copy.deepcopy(all_obs),
                sample_num=add_agent_n,
            )
            agent_goal_list = sample_poses(
                [],
                safe_dis=0,
                distr=group_distrs[gid][:, :, 1],
                grid_width=grid_width,
                rand_buffer=self.generation_pipeline_config.pos_rand_range,
                boundary=scenario["wind_size"],
                obs_list=copy.deepcopy(all_obs),
                sample_num=add_agent_n,
            )
            for idx in range(add_agent_n):
                ai_params = copy.deepcopy(AGNET_PARAM_DEFAULT)
                ai_params["pos"] = agent_pos_list[idx]
                ai_params["goal_pos"] = agent_goal_list[idx]
                ai_params["radius"] = agent_radius
                ai_params["pref_speed"] = agent_prefV
                ai_params["color"] = groups_colors[gid]
                agent_params["init_agent_params"].append(copy.deepcopy(ai_params))

        ### Start simulation
        fld_env.reset(
            scenario=copy.deepcopy(scenario_bound),
            agent_setting=copy.deepcopy(agent_params),
        )
        max_steps = -1
        for gid, (group_path_v_i, group_path_e_i) in enumerate(group_paths):
            path_len_group_i = 0
            for edge_i in group_path_e_i:
                line_p1 = copy.deepcopy(
                    scenario["roadmap"]["vertexes"][edge_i["edge"][0]]
                )
                line_p2 = copy.deepcopy(
                    scenario["roadmap"]["vertexes"][edge_i["edge"][1]]
                )
                path_len_group_i += np.linalg.norm(
                    np.array(line_p2) - np.array(line_p1)
                )
                if self.generation_pipeline_config.visual and show_paths:
                    fld_env.viewer.add_line(
                        p1=line_p1, p2=line_p2, color=groups_colors[gid]
                    )
            max_steps = max(
                max_steps,
                int(
                    path_len_group_i
                    * self.generation_pipeline_config.max_path_len_ratio
                    / self.generation_pipeline_config.agent_prefV
                ),
            )

        # # wait at beginning
        # if self.generation_pipeline_config.visual:
        #     while(1):
        #         fld_env.render()
        #         if fld_env.viewer.entered:
        #             break

        # warm up
        for stp_id in range(self.generation_pipeline_config.warm_up_steps):
            if self.generation_pipeline_config.visual:
                fld_env.render()
                # time.sleep(0.003)
            super(Field_Env, fld_env).perform_action(np.zeros((agent_n, 2)).tolist())
        for agt_id in range(agent_n):
            fld_env.agent_current_infor[agt_id]["traj_history"] = (
                fld_env.agent_current_infor[agt_id]["traj_history"][
                    : -self.generation_pipeline_config.warm_up_steps
                ]
            )
            assert len(fld_env.agent_current_infor[agt_id]["traj_history"]) == 0

        video_frms = []
        step = 0
        all_agent_trajs = [[] for i_ in range(group_n)]
        removed_agent_n = 0
        while 1:
            if self.generation_pipeline_config.visual:
                fld_env.render()
                time.sleep(0.006)
                if record_video_path is not None and not fld_env.viewer.closed:
                    loc = fld_env.viewer.get_location()
                    im_cv2 = np.array(
                        screenshot.Screenshot().capture(
                            (
                                loc[0],
                                loc[1],
                                fld_env.viewer.width,
                                fld_env.viewer.height,
                            )
                        )
                    )[:, :, 0:3]
                    video_frms.append(im_cv2)

            # remove agents that reach the goal or get out of the scenario, and record its trajectory
            for gid in range(group_n):
                for aid in gfields_for_ctrl[gid]["agent_ids"]:
                    if (
                        np.linalg.norm(
                            np.array(fld_env.agent_current_infor[aid]["pos"])
                            - np.array(fld_env.agent_current_infor[aid]["goal_pos"])
                        )
                        < self.generation_pipeline_config.reach_dis
                        or fld_env.agent_current_infor[aid]["pos"][0] <= 0
                        or fld_env.agent_current_infor[aid]["pos"][0]
                        >= scenario["wind_size"][0]
                        or fld_env.agent_current_infor[aid]["pos"][1] <= 0
                        or fld_env.agent_current_infor[aid]["pos"][1]
                        >= scenario["wind_size"][1]
                    ):
                        all_agent_trajs[gid].append(
                            {
                                "agent_id": aid,
                                "agent_trajs": np.array(
                                    fld_env.agent_current_infor[aid]["traj_history"]
                                )[:, 0, :],
                            }
                        )
                        fld_env.set_agent_position(aid, [-1e5, -1e5])
                        gfields_for_ctrl[gid]["agent_ids"].remove(aid)
                        removed_agent_n += 1

            fld_env.perform_action_fast(gfields_for_ctrl)

            step += 1
            if removed_agent_n >= agent_n or step >= max_steps:
                break

        if self.generation_pipeline_config.visual and not fld_env.viewer.closed:
            fld_env.viewer.close()

        # handle the rest agents
        for gid in range(group_n):
            for aid in gfields_for_ctrl[gid]["agent_ids"]:
                all_agent_trajs[gid].append(
                    {
                        "agent_id": aid,
                        "agent_trajs": np.array(
                            fld_env.agent_current_infor[aid]["traj_history"]
                        )[:, 0, :],
                    }
                )

        # save record video
        if self.generation_pipeline_config.visual and record_video_path is not None:
            video_wt = cv2.VideoWriter(
                record_video_path,
                cv2.VideoWriter_fourcc("P", "I", "M", "1"),
                280 / agent_prefV,
                (fld_env.viewer.width, fld_env.viewer.height),
            )
            for frm in video_frms:
                video_wt.write(frm)
            video_wt.release()
        # save simulation data
        if save_sim_path is not None:
            sim_data = {
                "scenario": scenario,
                "agent_num": agent_n,
                "agent_infor": fld_env.agent_current_infor,
            }
            np.save(save_sim_path, sim_data)

        return all_agent_trajs
