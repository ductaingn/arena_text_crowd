from typing import List

import attrs

import numpy as np

from arena_simulation_setup.tree.World import WorldDescription
from arena_text_crowd.crowd_generation_pipeline.input_models.prompt import StartGoalDistrLLMGeneration


@attrs.defind
class StartGoalDistrLLMGenerationPipeline:
    inference_client: StartGoalDistrLLMGeneration

    def inference(self, prompts: List[str], arena_world_description: WorldDescription):
        sgdistrs_all = self.inference_client.get_start_goal_zones(prompts, arena_world_description)

        return sgdistrs_all
