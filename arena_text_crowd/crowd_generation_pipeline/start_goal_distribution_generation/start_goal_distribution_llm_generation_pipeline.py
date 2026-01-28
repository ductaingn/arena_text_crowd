from typing import List, Tuple
import copy

import attrs

import numpy as np

import cv2

from rasterio import features, transform

from shapely import geometry as geom

from arena_simulation_setup.tree.World import WorldDescription
from arena_text_crowd.converters.arena_world_to_text_crowd_scenario import clamp
from arena_text_crowd.crowd_generation_pipeline.input_models.prompt import (
    StartGoalDistrLLMInferenceClient,
)
from arena_text_crowd.crowd_generation_pipeline.input_models.prompt.start_goal_distr_llm_inference_client import (
    LLMResponse,
    PedestrianGroup,
)
from arena_text_crowd.crowd_generation_pipeline.input_models.scenario import Scenario
from arena_text_crowd.crowd_generation_pipeline.input_models.constants import (
    AllSemanticObjects,
)
from arena_text_crowd.crowd_generation_pipeline.input_models.semantic.semantic_object import (
    Entrance,
    Exit,
)
from arena_text_crowd.crowd_generation_pipeline.utils.utils import (
    cv_visual_map,
    get_box,
    vectors_rotation,
)


@attrs.define
class StartGoalDistrLLMGenerationPipeline:
    inference_client: StartGoalDistrLLMInferenceClient
    grid_width: int
    sgdistr_size: Tuple[int, int] = attrs.field(
        init=False,
        default=(64, 64),
        metadata={"description": "Equal to the original work"},
    )

    def get_sg_distr(
        self,
        ped_group: PedestrianGroup,
        text_crowd_scenario: Scenario,
        arena_world_description: WorldDescription,
    ):
        sg_distr = np.zeros((*self.sgdistr_size, 2), dtype=np.float32)
        scenario_size = text_crowd_scenario.scenario_config.window_size

        # Get Arena World size
        x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
        for zone in arena_world_description.zones:
            x_min, y_min, x_max, y_max = (
                min(x_min, *(corner.x for corner in zone.corners)),
                min(y_min, *(corner.y for corner in zone.corners)),
                max(x_max, *(corner.x for corner in zone.corners)),
                max(y_max, *(corner.y for corner in zone.corners)),
            )
        arena_world_size = (x_max - x_min, y_max - y_min)

        transform_ = transform.from_bounds(
            0,
            0,
            scenario_size[0],
            scenario_size[1],
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
        scenario_entrance_range = text_crowd_scenario.scenario_config.objects[
            AllSemanticObjects.ENTRANCE
        ].size_range

        width, height = (
            clamp(
                ped_group.start.size[0] * scenario_size[0] / arena_world_size[0],
                scenario_entrance_range[0],
                scenario_entrance_range[1],
            ),
            clamp(
                ped_group.start.size[1] * scenario_size[1] / arena_world_size[1],
                scenario_entrance_range[0],
                scenario_entrance_range[1],
            ),
        )
        ctr = [
            ped_group.start.size[0] * scenario_size[0] / arena_world_size[0],
            ped_group.start.size[1] * scenario_size[1] / arena_world_size[1],
        ]
        box = vectors_rotation(
            np.array(get_box(width, height, [0.0, 0.0])).reshape(-1, 2).tolist(),
            0,
        )
        box = (np.array(box) + np.array(ctr)).tolist()
        obj_poly = geom.Polygon([[p[0], p[1]] for p in box])

        entrance = Entrance(
            width=width,
            height=height,
            center=ctr,
            rotation=0,
            polygon=obj_poly,
            name=zone.name,
        )
        entrance.set_infor()
        entrance.set_obj_graph()
        scenario.add_object(entrance)

        mask(entrance, channel=0)

        # --- GOAL ---
        scenario_exit_range = text_crowd_scenario.scenario_config.objects[
            AllSemanticObjects.EXIT
        ].size_range

        width, height = (
            clamp(
                ped_group.goal.size[0] * scenario_size[0] / arena_world_size[0],
                scenario_exit_range[0],
                scenario_exit_range[1],
            ),
            clamp(
                ped_group.goal.size[1] * scenario_size[1] / arena_world_size[1],
                scenario_exit_range[0],
                scenario_exit_range[1],
            ),
        )
        ctr = [
            ped_group.goal.size[0] * scenario_size[0] / arena_world_size[0],
            ped_group.goal.size[1] * scenario_size[1] / arena_world_size[1],
        ]
        box = vectors_rotation(
            np.array(get_box(width, height, [0.0, 0.0])).reshape(-1, 2).tolist(),
            0,
        )
        box = (np.array(box) + np.array(ctr)).tolist()
        obj_poly = geom.Polygon([[p[0], p[1]] for p in box])

        exit_ = Exit(
            width=width,
            height=height,
            center=ctr,
            rotation=0,
            polygon=obj_poly,
            name=zone.name,
        )
        exit_.set_infor()
        exit_.set_obj_graph()
        scenario.add_object(exit_)

        mask(exit_, channel=1)

        return sg_distr

    def inference(
        self,
        prompt: str,
        text_crowd_scenario: Scenario,
        arena_world_description: WorldDescription,
        show: bool = False,
    ) -> Tuple[np.ndarray, LLMResponse, Scenario]:
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

        sgdistr_all: List[np.ndarray] = []
        for ped_group in llm_response.pedestrian_groups:
            sgdistr = self.get_sg_distr(
                ped_group, text_crowd_scenario, arena_world_description
            )
            sgdistr_all.append(sgdistr)

        if show:
            smap = text_crowd_scenario.get_semantic_map()
            for sgdistr in sgdistr_all:
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

        return np.array(sgdistr_all), llm_response, text_crowd_scenario


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

    inference_client = StartGoalDistrLLMInferenceClient()

    sg_distr_llm_gen_pipeline = StartGoalDistrLLMGenerationPipeline(
        inference_client=inference_client,
        grid_width=16,
    )

    prompt = "Two groups enter from the main entrance in the top right corner and both of them walk through the upper right passage first. Afterward, one group leaves at the bottom left exit. Another group follows a different path where they  exit through the bottom gate"

    pred_group_sgdistr, _, _ = sg_distr_llm_gen_pipeline.inference(
        prompt=prompt,
        text_crowd_scenario=scenario,
        arena_world_description=arena_world.load(),
        show=True,
    )
