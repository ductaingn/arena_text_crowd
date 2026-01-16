from typing import List, Tuple
import copy

import attrs

import numpy as np

import cv2

from rasterio import features, transform

from shapely import geometry as geom

from arena_simulation_setup.tree.World import WorldDescription
from arena_text_crowd.crowd_generation_pipeline.input_models.prompt import (
    StartGoalDistrLLMInferenceClient,
    StartGoalPair,
)
from arena_text_crowd.crowd_generation_pipeline.input_models.scenario import Scenario
from arena_text_crowd.crowd_generation_pipeline.input_models.semantic.semantic_object import (
    Entrance,
)
from arena_text_crowd.crowd_generation_pipeline.input_models.constants import (
    AllSemanticObjects,
)
from arena_text_crowd.crowd_generation_pipeline.utils.utils import cv_visual_map


@attrs.define
class StartGoalDistrLLMGenerationPipeline:
    inference_client: StartGoalDistrLLMInferenceClient
    grid_width: int
    sgdistr_size: Tuple[int, int] = attrs.field(
        init=False,
        default=(64, 64),
        metadata={"description": "Equal to the original work"},
    )

    def get_sg_distr(self, start_goal_zone, text_crowd_scenario):
        sg_distr = np.zeros((*self.sgdistr_size, 2), dtype=np.float32)

        transform_ = transform.from_bounds(
            0, 0,
            text_crowd_scenario.scenario_config.window_size[0],
            text_crowd_scenario.scenario_config.window_size[1],
            self.sgdistr_size[0],
            self.sgdistr_size[1],
        )

        def mask(area, channel):
            poly = geom.Polygon(copy.deepcopy(area.whole_box))
            mask = features.geometry_mask(
                [poly],
                out_shape=self.sgdistr_size,
                transform=transform_,
                all_touched=True,
            )
            mask = np.flip(mask, axis=0).transpose(1, 0)
            sg_distr[~mask, channel] = 1.0

        # --- START ---
        start_area = next(
            a for a in text_crowd_scenario.areas_dict[AllSemanticObjects.ENTRANCE]
            if a.name == start_goal_zone.start.name
        )
        mask(start_area, channel=0)

        # --- GOAL ---
        goal_area = next(
            a for a in text_crowd_scenario.areas_dict[AllSemanticObjects.EXIT]
            if a.name == start_goal_zone.goal.name
        )
        mask(goal_area, channel=1)

        return sg_distr


    def inference(
        self,
        prompt: str,
        text_crowd_scenario: Scenario,
        arena_world_description: WorldDescription,
        show: bool = False,
    ):
        """
        Parameters
        ----------
        prompt: str
            Unlike in Text-Crowd original pipeline, this method take in raw user prompt instead of list of canonicalized group description.
        """
        if (
            self.grid_width * self.sgdistr_size[0]
            != text_crowd_scenario.scenario_config.window_size[0]
            or self.grid_width * self.sgdistr_size[1]
            != text_crowd_scenario.scenario_config.window_size[1]
        ):
            raise ValueError(
                "Invalid grid width! Valid grid width is scenario window size // 64"
            )
        llm_response = self.inference_client.inference(prompt, arena_world_description)
        smap = text_crowd_scenario.get_semantic_map()

        sgdistr_all: List[np.ndarray] = []
        for start_goal_zone in llm_response.start_goal_zones:
            sgdistr = self.get_sg_distr(start_goal_zone, text_crowd_scenario)
            sgdistr_all.append(sgdistr)

            if show:
                smap = copy.deepcopy(smap)
                sg_colors = np.array([[0, 255, 0], [0, 0, 255]])
                map_colors = np.concatenate(
                    [np.random.randint(0, 255, (len(smap[0, 0]) - 2, 3)), sg_colors]
                )
                smap_cvimg = cv_visual_map(
                    smap, colors=map_colors, save_nm=None, show=False
                )
                distr_cvimg = cv_visual_map(
                    sgdistr, colors=sg_colors, save_nm=None, show=False
                )
                interval = np.ones((len(smap_cvimg), 5, 3)) * 255
                cat_img = cv2.hconcat(
                    [interval, smap_cvimg, interval, distr_cvimg, interval]
                )
                cat_img_resize = cv2.resize(
                    cat_img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC
                )
                cv2.imshow("img", cat_img_resize / 255)
                cv2.waitKey(0)

        return sgdistr_all


if __name__ == "__main__":
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

    semantic_map = np.array([scenario.get_semantic_map()])

    inference_client = StartGoalDistrLLMInferenceClient()

    sg_distr_llm_gen_pipeline = StartGoalDistrLLMGenerationPipeline(
        inference_client=inference_client,
        grid_width=16,
    )

    prompt = "Two groups enter from the main entrance in the top right corner and both of them walk through the upper right passage first. Afterward, one group leaves at the bottom left exit. Another group follows a different path where they  exit through the bottom gate"

    pred_group_sgdistr = sg_distr_llm_gen_pipeline.inference(
        prompt=prompt,
        text_crowd_scenario=scenario,
        arena_world_description=arena_world.load(),
        show=True,
    )
