import copy
from typing import List, Tuple, Dict, Optional

import attrs

import numpy as np

import shapely.geometry as geom

from ..utils.utils import get_box_ll, poly_collision_check
from .constants import AllSemanticObjects
from .semantic.semantic_map import SemanticMap
from .semantic.semantic_object import (
    SemanticObject,
    Rectangle,
    Triangle,
    Circle,
    ZebraCrossing,
    Passage,
    Entrance,
    Exit,
)


@attrs.define
class ObjectConfig:
    num: int = 1
    size_range: Tuple[int, int] = (40, 60)
    graph_buffer: Optional[int] = 20


@attrs.define
class ScenarioConfig:
    window_size: Tuple[int, int] = (1024, 1024)
    bound_srk_scale: float = 1 / 5
    safe_dis: float = 80.0
    objects: Dict[AllSemanticObjects, ObjectConfig] = {
        AllSemanticObjects.RECTANGLE: ObjectConfig(1, (40, 60), 20),
        AllSemanticObjects.TRIANGLE: ObjectConfig(1, (40, 60), 20),
        AllSemanticObjects.CIRCLE: ObjectConfig(1, (40, 60), 20),
        AllSemanticObjects.ZEBRA_CROSSING: ObjectConfig(1, (180, 230), 20),
        AllSemanticObjects.PASSAGE: ObjectConfig(1, (160, 180), 20),
        AllSemanticObjects.ENTRANCE: ObjectConfig(2, (60, 60)),
        AllSemanticObjects.EXIT: ObjectConfig(1, (60, 60)),
    }


@attrs.define
class Scenario:
    scenario_config: ScenarioConfig
    obstacle_dict: Dict[AllSemanticObjects, List[Rectangle | Triangle | Circle]] = {
        AllSemanticObjects.RECTANGLE: [],
        AllSemanticObjects.TRIANGLE: [],
        AllSemanticObjects.CIRCLE: [],
    }
    zebra_crossing_list: List[ZebraCrossing] = []
    passages_list: List[Passage] = []
    areas_dict: Dict[AllSemanticObjects, List[Entrance | Exit]] = {
        AllSemanticObjects.ENTRANCE: [],
        AllSemanticObjects.EXIT: [],
    }
    extended: bool = attrs.field(init=False, default=False)
    window_size_sub: Tuple[Tuple[float, float], Tuple[float, float]] = attrs.field(
        init=False
    )

    @window_size_sub.default
    def _window_size_sub_factory(self):
        window_size = copy.deepcopy(self.scenario_config.window_size)
        w_rg = [
            window_size[0] * self.scenario_config.bound_srk_scale / 2,
            window_size[0] * (1 - self.scenario_config.bound_srk_scale / 2),
        ]
        h_rg = [
            window_size[1] * self.scenario_config.bound_srk_scale / 2,
            window_size[1] * (1 - self.scenario_config.bound_srk_scale / 2),
        ]

        return (w_rg, h_rg)

    @classmethod
    def random(cls, scenario_config: ScenarioConfig) -> "Scenario":
        scenario_ = Scenario(scenario_config)

        boundary = geom.LineString(
            geom.Polygon(
                get_box_ll(
                    x=scenario_config.window_size[0]
                    * (1 - scenario_config.bound_srk_scale),
                    y=scenario_config.window_size[1]
                    * (1 - scenario_config.bound_srk_scale),
                    lowerleft=(
                        scenario_.window_size_sub[0][0],
                        scenario_.window_size_sub[1][0],
                    ),
                )
            ).exterior.coords
        ).buffer(1, cap_style=3, join_style=2)
        obj_polys_list = [boundary]

        for obj_type, obj_config in scenario_config.objects.items():
            obj_num = obj_config.num
            obj_size_range = copy.deepcopy(obj_config.size_range)
            for r_i in range(obj_num):
                # get obj (except for attributes)
                loop_cnt = 0
                while True:
                    loop_cnt += 1
                    if loop_cnt >= 30000:
                        return None
                    if (
                        obj_type == AllSemanticObjects.ENTRANCE
                        or obj_type == AllSemanticObjects.EXIT
                    ):
                        semantic_obj: Entrance | Exit = scenario_.get_rand_obj(
                            obj_type,
                            obj_size_range,
                            [
                                copy.deepcopy(scenario_.window_size_sub[0]),
                                copy.deepcopy(scenario_.window_size_sub[1]),
                            ],
                        )
                        jd = True
                        for area in (
                            scenario_.areas_dict[AllSemanticObjects.ENTRANCE]
                            + scenario_.areas_dict[AllSemanticObjects.EXIT]
                        ):
                            area_type = (
                                AllSemanticObjects.ENTRANCE
                                if isinstance(area, Entrance)
                                else AllSemanticObjects.EXIT
                            )
                            dis_lim = (
                                (
                                    scenario_config.window_size[0]
                                    * (1 - scenario_config.bound_srk_scale)
                                    + scenario_config.window_size[1]
                                    * (1 - scenario_config.bound_srk_scale)
                                )
                                / 2
                            ) / (
                                max(
                                    scenario_config.objects[
                                        AllSemanticObjects.ENTRANCE
                                    ].num,
                                    scenario_config.objects[
                                        AllSemanticObjects.EXIT
                                    ].num,
                                )
                            )
                            dis_lim = max(
                                dis_lim,
                                scenario_config.objects[area_type].size_range[1] / 2
                                + scenario_config.objects[obj_type].size_range[1] / 2,
                            )
                            if (
                                np.linalg.norm(
                                    np.array(semantic_obj.center)
                                    - np.array(area.center)
                                )
                                < dis_lim
                            ):
                                jd = False
                                break
                        if jd:
                            break
                    else:
                        semantic_obj = scenario_.get_rand_obj(
                            obj_type,
                            copy.deepcopy(obj_size_range),
                            [
                                copy.deepcopy(scenario_.window_size_sub[0]),
                                copy.deepcopy(scenario_.window_size_sub[1]),
                            ],
                        )
                        if poly_collision_check(
                            obj_polys_list,
                            semantic_obj.polygon.buffer(scenario_config.safe_dis),
                        ):
                            break

                # set obj's attributes
                semantic_obj.set_infor()
                semantic_obj.set_obj_graph()
                obj_polys_list.append(semantic_obj.polygon)

                scenario_.add_object(semantic_obj)

        return scenario_

    @classmethod
    def get_scenario_bound(cls, scenario: "Scenario") -> "Scenario":
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

        return scenario_bound

    def get_rand_obj(
        self,
        obj_type: AllSemanticObjects,
        objsize_range: Tuple[float, float],
        area_: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> SemanticObject:
        if obj_type == AllSemanticObjects.RECTANGLE:
            return Rectangle.random(size_range=objsize_range, area=area_)
        elif obj_type == AllSemanticObjects.TRIANGLE:
            return Triangle.random(size_range=objsize_range, area=area_)
        elif obj_type == AllSemanticObjects.CIRCLE:
            return Circle.random(size_range=objsize_range, area=area_)
        elif obj_type == AllSemanticObjects.ZEBRA_CROSSING:
            return ZebraCrossing.random(size_range=objsize_range, area=area_)
        elif obj_type == AllSemanticObjects.PASSAGE:
            return Passage.random(size_range=objsize_range, area=area_)
        elif obj_type == AllSemanticObjects.ENTRANCE:
            return Entrance.random(size_range=objsize_range, area=area_)
        elif obj_type == AllSemanticObjects.EXIT:
            return Exit.random(size_range=objsize_range, area=area_)
        else:
            raise NotImplementedError

    def add_object(self, semantic_obj: SemanticObject):
        if isinstance(semantic_obj, Rectangle):
            self.obstacle_dict[AllSemanticObjects.RECTANGLE].append(semantic_obj)
        elif isinstance(semantic_obj, Triangle):
            self.obstacle_dict[AllSemanticObjects.TRIANGLE].append(semantic_obj)
        elif isinstance(semantic_obj, Circle):
            self.obstacle_dict[AllSemanticObjects.CIRCLE].append(semantic_obj)
        elif isinstance(semantic_obj, ZebraCrossing):
            self.zebra_crossing_list.append(semantic_obj)
        elif isinstance(semantic_obj, Passage):
            self.passages_list.append(semantic_obj)
        elif isinstance(semantic_obj, Entrance):
            self.areas_dict[AllSemanticObjects.ENTRANCE].append(semantic_obj)
        elif isinstance(semantic_obj, Exit):
            self.areas_dict[AllSemanticObjects.EXIT].append(semantic_obj)
        else:
            raise NotImplementedError

    def get_semantic_map(self) -> np.ndarray:
        if not self.extended:
            self.extend()

        return SemanticMap(
            self.scenario_config.window_size,
            self.obstacle_dict,
            self.zebra_crossing_list,
            self.passages_list,
            self.areas_dict,
        ).from_text_crowd_scenario()

    def extend(self):
        if self.extended is True:
            print("The scenario has already been extended.")

        for obs in (
            self.obstacle_dict[AllSemanticObjects.RECTANGLE]
            + self.obstacle_dict[AllSemanticObjects.TRIANGLE]
            + self.obstacle_dict[AllSemanticObjects.CIRCLE]
        ):
            obs.set_infor()
        for zebra_crossing in self.zebra_crossing_list:
            zebra_crossing.set_infor()
        for passage in self.passages_list:
            passage.set_infor()
        for area in (
            self.areas_dict[AllSemanticObjects.ENTRANCE]
            + self.areas_dict[AllSemanticObjects.EXIT]
        ):
            area.set_infor()

        self.extended = True

    def get_all_obstacles(self):
        if self.extended is False:
            self.extend()

        all_obstacles_list = []
        for obs in (
            self.obstacle_dict[AllSemanticObjects.RECTANGLE]
            + self.obstacle_dict[AllSemanticObjects.TRIANGLE]
        ):
            all_obstacles_list.append(copy.deepcopy(np.array(obs.vertexes)).tolist())
        for obs in self.obstacle_dict[AllSemanticObjects.CIRCLE]:
            all_obstacles_list.append(copy.deepcopy(np.array(obs.edges)).tolist())
        for psg in self.passages_list:
            for psg_obs in psg.obstacles:
                all_obstacles_list.append(copy.deepcopy(np.array(psg_obs)).tolist())

        return all_obstacles_list

    # def get_group_paths(self):
    #     if not self.extended:
    #         self.extend()

    #     groups_paths = []
    #     for group_id in self.
    #         roadmap = Roadmap(
    #             obstacle_dict=self.obstacle_dict,
    #             zebra_crossing_list=self.zebra_crossing_list,
    #             passages_list=self.passages_list,
    #             areas_dict=self.areas_dict,
    #             window_size_sub=self.window_size_sub,
    #         )
    #         start_p = self.areas_dict[]copy.deepcopy(
    #             scenario_["areas_list"][etc_id]["attributes"]["graph"]["points_idx_inRM"][0]
    #         )
    #         goal_p = copy.deepcopy(
    #             scenario_["areas_list"][exit_id]["attributes"]["graph"]["points_idx_inRM"][
    #                 0
    #             ]
    #         )
    #         # sample a path randomly and postprocess it
    #         path_vs_init, path_es_init = self.roadmap.sample_path(
    #             constrains=PathConstrains(ParametersMode.SIMPLE)
    #         )
    #         if path_vs_init is None or path_es_init is None:
    #             valid_path = False
    #             break
    #         path_vs, path_es = roadmap.path_postprocess(
    #             path_v=copy.deepcopy(path_vs_init),
    #             path_e=copy.deepcopy(path_es_init),
    #             rlx_dis=param_["path_relax_dis"],
    #         )
    #     groups_paths.append([copy.deepcopy(path_vs), copy.deepcopy(path_es)])

    #     return groups_paths
