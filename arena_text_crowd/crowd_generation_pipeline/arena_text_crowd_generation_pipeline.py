import copy
from typing import Dict, List

from arena_simulation_setup.tree.World import WorldDescription
import attrs

import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import DDPMScheduler, UNet2DConditionModel

from arena_text_crowd.crowd_generation_pipeline.input_models.prompt.start_goal_distr_llm_inference_client import (
    LLMResponse,
)
from arena_text_crowd.crowd_generation_pipeline.start_goal_distribution_generation.start_goal_distribution_llm_generation_pipeline import (
    StartGoalDistrLLMGenerationPipeline as SGDLLMGP,
)
from arena_text_crowd.crowd_generation_pipeline.input_models.prompt import (
    StartGoalDistrLLMInferenceClient,
)
from arena_text_crowd.crowd_generation_pipeline.velocity_field_generation.velocity_field_generation_pipeline import (
    VelocityFieldGenerationPipeline as VFGP,
    VelocityFieldGenerationPipelineConfig as VFGPConfig,
)
from arena_text_crowd.crowd_generation_pipeline.utils.field import Field
from arena_text_crowd.crowd_generation_pipeline.input_models.scenario import (
    Scenario,
)
from arena_text_crowd.crowd_generation_pipeline.input_models.prompt import (
    PromptCanonicalizer,
)


@attrs.define
class ArenaTextCrowdGenerationPipelineConfig:
    # 2d sim
    visual: bool = True

    # LLM configuration
    model: str = "gemini-2.5-flash"
    top_p: float = 0.8
    temparature: float = 0.2
    top_k: int = 40


@attrs.define
class CrowdGenerationPipeline:
    generation_pipeline_config: ArenaTextCrowdGenerationPipelineConfig
    vel_field_gen_config: VFGPConfig

    sg_distr_gen_pipeline: SGDLLMGP = attrs.field(init=False)
    vel_field_gen_pipeline: VFGP = attrs.field(init=False)
    canonicalizer: PromptCanonicalizer = attrs.field(init=False)

    @sg_distr_gen_pipeline.default
    def _sg_distr_gen_pipeline_factory(self):
        sg_distr_llm_inference_client = StartGoalDistrLLMInferenceClient(
            self.generation_pipeline_config.model, self.generation_pipeline_config.top_p
        )
        return SGDLLMGP(sg_distr_llm_inference_client, grid_width=16)

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
            self.vel_field_gen_config.unet_dir,
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

    @canonicalizer.default
    def canonicalizer_factory(self):
        return PromptCanonicalizer(
            self.generation_pipeline_config.model, self.generation_pipeline_config.top_p
        )

    def get_canonicalized_des(self, prompt: str, llm_response: LLMResponse):
        """
        Get the canonicalized description text of groups, knowing the start and goal zones
        """
        return self.canonicalizer.canonicalize_from_zones(prompt, llm_response)

    def sample_pedestrians(
        self, llm_response: LLMResponse, arena_world_description: WorldDescription
    ) -> List[Dict]:
        pedestrians = []
        for g_id, ped_group in enumerate(llm_response.pedestrian_groups):
            n_peds = ped_group.num_pedestrians
            x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
            found_zone = False

            # 1. Find the zone and calculate bounds
            for zone in arena_world_description.zones:
                if zone.name == ped_group.start.name:
                    x_min = min(x_min, *(corner.x for corner in zone.corners))
                    y_min = min(y_min, *(corner.y for corner in zone.corners))
                    x_max = max(x_max, *(corner.x for corner in zone.corners))
                    y_max = max(y_max, *(corner.y for corner in zone.corners))
                    found_zone = True
                    break  # Stop looking once the zone is found

            # 2. Sample only if a valid zone was found
            if found_zone:
                x_pos = np.random.uniform(low=x_min, high=x_max, size=n_peds)
                y_pos = np.random.uniform(low=y_min, high=y_max, size=n_peds)

                for p_id, (x, y) in enumerate(zip(x_pos, y_pos)):
                    pedestrians.append(
                        {
                            "name": f"hunav_{p_id}_group_{g_id}",
                            "group_id": g_id,
                            "pos": [x, y, 0.0],
                        }
                    )
            else:
                raise ValueError(
                    f"Zone {ped_group.start.name} not found in world description."
                )

        return pedestrians

    def generate(
        self,
        prompt: str,
        scenario: Scenario,
        arena_world_description: WorldDescription,
        show: bool = False,
    ):
        print("Inferring start and goal distributions...")
        pred_group_sgdistrs, llm_response = self.sg_distr_gen_pipeline.inference(
            prompt=prompt,
            text_crowd_scenario=scenario,
            arena_world_description=arena_world_description,
            show=show,
        )
        sampled_pedestrians = self.sample_pedestrians(
            llm_response, arena_world_description
        )

        canonicalized_descriptions = self.get_canonicalized_des(prompt, llm_response)
        group_n = len(canonicalized_descriptions)
        assert group_n == len(llm_response.pedestrian_groups), (
            f"Size of canonicalized_descriptions and pred_group_sgdistrs mismatch, got: {group_n} and {len(llm_response.pedestrian_groups)}, respectively."
        )

        semantic_map = scenario.get_semantic_map()
        # Create a copy of semantic map for each group
        smaps = []
        for _ in range(group_n):
            smaps.append(copy.deepcopy(semantic_map))

        print("Inferring fields...")
        pred_group_fields = self.vel_field_gen_pipeline.inference(
            smaps=np.array(smaps),
            prompts=copy.deepcopy(canonicalized_descriptions),
            sg_distrs=copy.deepcopy(pred_group_sgdistrs),
            num_inference_steps=self.vel_field_gen_config.num_inference_steps,
            guidance_scale=self.vel_field_gen_config.guidance_scale,
            save_path=None,
            show=show,
        )

        # normalize the fields and remove vectors in obstacle
        grid_size = [
            self.vel_field_gen_config.map_size,
            self.vel_field_gen_config.map_size,
        ]
        grid_width = scenario.scenario_config.window_size[0] // grid_size[0]
        base_field = Field(scenario=scenario, grid_width=grid_width)
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
                nan=0,
            )
            pred_group_fields[group_id][obs_coords[:, 0], obs_coords[:, 1]] = np.array(
                [0.0, 0.0]
            )

        return pred_group_fields, sampled_pedestrians


if __name__ == "__main__":
    import os
    from pathlib import Path
    from arena_simulation_setup.tree.World import World
    from arena_text_crowd.converters import arena_world_to_text_crowd_scenario

    # Create Text-Crowd scenario from Arena World
    world_path = Path(
        "/home/linh/ductai_nguyen_ws/Arena_ws/install/arena_simulation_setup/share/arena_simulation_setup/worlds/hospital_1"
    )
    arena_world = World(path=world_path)
    scenario = arena_world_to_text_crowd_scenario(
        arena_world=arena_world, scenario_size=(1024, 1024), wall_thickness=1.0
    )

    # Generate agents trajectories
    models_path = Path(
        "/home/linh/ductai_nguyen_ws/Text-Crowd/text_crowd/Language_Crowd_Animation/Models_Server_ForTest"
    )
    crowd_generation_pipeline = CrowdGenerationPipeline(
        ArenaTextCrowdGenerationPipelineConfig(),
        VFGPConfig(
            unet_dir=os.path.join(models_path, "Field-Full-V2/checkpoint-270000/unet")
        ),
    )

    prompt = "Two groups enter from the main entrance in the top right corner and both of them walk through the upper right passage first. Afterward, one group leaves at the bottom left exit. Another group follows a different path where they  exit through the bottom gate"
    pred_velocity_field = crowd_generation_pipeline.generate(
        prompt=prompt,
        scenario=scenario,
        arena_world_description=arena_world.load(),
        show=True,
    )
    print(pred_velocity_field.shape)
