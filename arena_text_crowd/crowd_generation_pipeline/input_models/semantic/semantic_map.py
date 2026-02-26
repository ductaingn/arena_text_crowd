from typing import Tuple, List, Dict
import copy

import attrs

import numpy as np

import shapely.geometry as geom

from rasterio import features, transform  # Because rasterio is stupid

from ..constants import AllSemanticObjects, EnvironmentParams
from .semantic_object import (
    Rectangle,
    Triangle,
    Circle,
    ZebraCrossing,
    Passage,
    Entrance,
    Exit,
)


@attrs.define
class SemanticMap:
    """
    Occupancy grid map
    """

    window_size: Tuple[int, int]
    obstacle_dict: Dict[AllSemanticObjects, List[Rectangle | Triangle | Circle]]
    zebra_crossing_list: List[ZebraCrossing]
    passages_list: List[Passage]
    areas_dict: Dict[AllSemanticObjects, List[Entrance | Exit]]

    def get_feature(self, obj_semantic_name: str | AllSemanticObjects):
        if isinstance(obj_semantic_name, AllSemanticObjects):
            obj_semantic_name = obj_semantic_name.value
        all_obj_smtcs = [obj.value for obj in AllSemanticObjects]
        all_obj_smtcs = copy.deepcopy(all_obj_smtcs)

        # Find index of the given semantic name
        if obj_semantic_name not in all_obj_smtcs:
            raise ValueError(f"Unknown semantic object: {obj_semantic_name}")

        objsm_id = all_obj_smtcs.index(obj_semantic_name)

        return objsm_id

    def mask(self, poly: geom.Polygon, semantic_map_: np.ndarray, grid_size: List[int], transform_: transform.Affine, semantic_object: AllSemanticObjects):
        mask = features.geometry_mask(
            [poly],
            out_shape=(grid_size[1], grid_size[0]),
            transform=transform_,
            all_touched=True,
        )
        mask = np.flip(mask, axis=0).transpose(1, 0)
        objsm_id = self.get_feature(semantic_object)
        # Create one-hot feature vector for the semantic object (support multi-class semantic map)
        semantic_map_[~mask, objsm_id] = 1
    
    def from_arena_world(self): ...

    def from_text_crowd_scenario(self) -> np.ndarray:
        grid_size = [
            int(self.window_size[0] / EnvironmentParams.grid_width_map),
            int(self.window_size[1] / EnvironmentParams.grid_width_map),
        ]

        semantic_map_ = np.zeros(
            (grid_size[0], grid_size[1], len(AllSemanticObjects) - 1)
        )  # Because I added PASSAGE, in the original work there was only passage_free and passage_obstacles

        transform_ = transform.from_bounds(
            0,
            0,
            self.window_size[0],
            self.window_size[1],
            grid_size[0],
            grid_size[1],
        )

        for passage in self.passages_list:
            poly_free = geom.Polygon(copy.deepcopy(passage.free_space))
            self.mask(
                poly_free, 
                semantic_map_, 
                grid_size, 
                transform_, 
                AllSemanticObjects.PASSAGE_FREE
            )
            for obs_id in range(2):
                poly_obs = geom.Polygon(copy.deepcopy(passage.obstacles[obs_id]))
                self.mask(
                    poly_obs, 
                    semantic_map_, 
                    grid_size, 
                    transform_, 
                    AllSemanticObjects.PASSAGE_OBSTACLE
                )

        for obstacle in (
            self.obstacle_dict[AllSemanticObjects.RECTANGLE]
            + self.obstacle_dict[AllSemanticObjects.TRIANGLE]
        ):
            poly_ = geom.Polygon(copy.deepcopy(obstacle.vertexes))
            self.mask(
                poly_, 
                semantic_map_, 
                grid_size, 
                transform_, 
                AllSemanticObjects.RECTANGLE if isinstance(obstacle, Rectangle) else AllSemanticObjects.TRIANGLE
            )

        for circle in self.obstacle_dict[AllSemanticObjects.CIRCLE]:
            poly_ = geom.Polygon(copy.deepcopy(circle.edges))
            self.mask(
                poly_, 
                semantic_map_, 
                grid_size, 
                transform_, AllSemanticObjects.CIRCLE
            )

        for zebra_crossing in self.zebra_crossing_list:
            poly_ = geom.Polygon(copy.deepcopy(zebra_crossing.whole_box))
            self.mask(
                poly_, 
                semantic_map_, 
                grid_size, 
                transform_, 
                AllSemanticObjects.ZEBRA_CROSSING
            )

        for area in (
            self.areas_dict[AllSemanticObjects.ENTRANCE]
            + self.areas_dict[AllSemanticObjects.EXIT]
        ):
            poly_ = geom.Polygon(copy.deepcopy(area.whole_box))
            self.mask(
                poly_, 
                semantic_map_, 
                grid_size, 
                transform_, 
                AllSemanticObjects.ENTRANCE if isinstance(area, Entrance) else AllSemanticObjects.EXIT
            )

        return semantic_map_
