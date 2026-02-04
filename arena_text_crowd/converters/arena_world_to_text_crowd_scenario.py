from typing import Dict, List, Tuple

import numpy as np

import shapely.geometry as geom

from arena_simulation_setup.tree.World import World, WorldDescription
# from arena_models.impl.build.ObjectDatabaseBuilder import ObjectAnnotation

from arena_text_crowd.crowd_generation_pipeline.input_models.scenario import (
    Scenario,
    ScenarioConfig,
)
from arena_text_crowd.crowd_generation_pipeline.input_models.semantic.semantic_object import (
    Entrance,
    Exit,
    Passage,
    Rectangle,
    ZebraCrossing,
)
from arena_text_crowd.crowd_generation_pipeline.input_models.constants import (
    AllSemanticObjects,
)
from arena_text_crowd.crowd_generation_pipeline.utils.utils import (
    vectors_rotation,
    get_box,
)


def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))


def arena_world_to_text_crowd_scenario(
    arena_world: World | WorldDescription,
    scenario_size: Tuple[int, int] = (800, 800),
    wall_thickness: float = 0.5,
    *,
    entrances: List[Entrance] | None = None,
    exits: List[Exit] | None = None,
) -> Tuple[Scenario, Dict[str, str]]:
    """
    Convert an Arena World into a Text-Crowd Scenario, keep corners and walls only, Arena Zones is considered as entrances and exits.

    Args:
        arena_world : World | WorldDescription
            The world to be converted.
        entrances: List[Entrance]
            List of the entrances for agents in a Text-Crowd Scenario
        exits: List[Exit]
            List of the exits for agents in a Text-Crowd Scenario
        scenario_size: Tuple[int, int]
            Size (width, height) of a Text-Crowd Scenario (it's a 2D grid map)
        wall_thickness: float
            Walls in Arena World will be converted into Rectangle in Text-Crowd Scenario, hence the wall thickness
        entrances: List[Entrance]
            List of entrances
        exits: List[Exit]
            List of exits

    Returns:
        scenario : Scenario
        arena_entity_to_semantic_entity_map : Dict[str, str]
    """
    if isinstance(arena_world, World):
        arena_world_description = arena_world.load()
    elif isinstance(arena_world, WorldDescription):
        arena_world_description = arena_world

    arena_entity_to_semantic_entity_map = {}
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

    scenario = Scenario(ScenarioConfig(window_size=scenario_size))

    for zone in arena_world_description.zones:
        if entrances and not exits:
            raise ValueError("Missing exits")
        elif not entrances and exits:
            raise ValueError("Missing entrances")
        elif entrances and exits:
            for entrance in entrances:
                entrance.set_infor()
                entrance.set_obj_graph()
                scenario.add_object(entrance)

            for exit_ in exits:
                exit_.set_infor()
                exit_.set_obj_graph()
                scenario.add_object(exit_)

        # Convert Arena Wall into Text-Crowd Rectangle
        # Assuming the length of the wall is the corresponding rectangle height,
        # its thickness is the rectangle's width
        for wall in zone.walls:
            end = np.array(
                [
                    wall.end.x * scenario_size[0] / arena_world_size[0],
                    wall.end.y * scenario_size[1] / arena_world_size[1],
                ]
            )
            start = np.array(
                [
                    wall.start.x * scenario_size[0] / arena_world_size[0],
                    wall.start.y * scenario_size[1] / arena_world_size[1],
                ]
            )
            d = end - start
            height = np.linalg.norm(d)
            width = wall_thickness
            rotation = np.arctan(d[0] / d[1]) if d[1] != 0 else np.pi / 2
            ctr = (end + start) / 2.0
            box = vectors_rotation(
                np.array(get_box(width, height, [0.0, 0.0])).reshape(-1, 2).tolist(),
                rotation,
            )
            box = (np.array(box) + ctr).tolist()
            obj_poly = geom.Polygon([[p[0], p[1]] for p in box])
            rec = Rectangle(width=width, height=height, polygon=obj_poly, vertexes=box)
            rec.set_infor()
            rec.set_obj_graph()
            scenario.add_object(rec)

        # Convert Arena Door into Text-Crowd Passage
        for door in zone.doors:
            end = np.array(
                [
                    door.end.x * scenario_size[0] / arena_world_size[0],
                    door.end.y * scenario_size[1] / arena_world_size[1],
                ]
            )
            start = np.array(
                [
                    door.start.x * scenario_size[0] / arena_world_size[0],
                    door.start.y * scenario_size[1] / arena_world_size[1],
                ]
            )
            d = end - start
            height = np.linalg.norm(d)
            width = wall_thickness
            rotation = np.arctan(d[1] / d[0]) if d[0] != 0 else np.pi / 2
            ctr = (end + start) / 2.0
            box = vectors_rotation(
                np.array(get_box(width, height, [0.0, 0.0])).reshape(-1, 2).tolist(),
                rotation,
            )
            box = (np.array(box) + ctr).tolist()
            obj_poly = geom.Polygon([[p[0], p[1]] for p in box])
            passage = Passage(
                width=height,
                height=width,
                center=ctr,
                passage_width=height,  # Take whole length of the door as passage width
                rotation=rotation / np.pi * 180.0,
                polygon=obj_poly,
                name=door.name,
            )
            passage.set_infor()
            passage.set_obj_graph()
            scenario.add_object(passage)
            arena_entity_to_semantic_entity_map[door.name] = (
                AllSemanticObjects.PASSAGE.value
            )

        # Convert hallways into Text-Crowd Zebra Crossing
        if "hallway" in zone.name or "corridor" in zone.name:
            width, height, rot = (
                zone.floor.x_length * scenario_size[0] / arena_world_size[0]
                - wall_thickness,
                zone.floor.y_length * scenario_size[1] / arena_world_size[1]
                - wall_thickness,
                90,
            )

            if width > height:
                temp = width
                width = height
                height = temp
                rot = 0
            ctr = [
                zone.floor.pos.x * scenario_size[0] / arena_world_size[0],
                zone.floor.pos.y * scenario_size[1] / arena_world_size[1],
            ]
            zb_w = width / 15 - 1e-6
            box = vectors_rotation(
                np.array(get_box(width, height, [0.0, 0.0])).reshape(-1, 2).tolist(),
                rot,
            )
            box = (np.array(box) + np.array(ctr)).tolist()
            obj_poly = geom.Polygon([[p[0], p[1]] for p in box])
            passage = ZebraCrossing(
                width=height,
                height=width,
                center=ctr,
                zb_line_width=zb_w,
                rotation=rot,
                polygon=obj_poly,
            )
            passage.set_infor()
            passage.set_obj_graph()
            scenario.add_object(passage)
            arena_entity_to_semantic_entity_map[zone.name] = (
                AllSemanticObjects.ZEBRA_CROSSING.value
            )

    for obstacle in arena_world_description.all_static_entities:
        # TODO: Implement
        pass
        # object = ObjectAnnotation(obstacle.name, ...)
        # rec = Rectangle(polygon=object.bounding_box)
        # scenario.add_object(rec)

    # Hard coded empty space for world `hospital_1`
    empty_spaces = [[17.1, 7.0, [21.5, 25.65]], [4.05, 2.95, [23.525, 15.075]]]
    for zone in empty_spaces:
        height, width, ctr = zone
        height = height * scenario_size[1] / arena_world_size[1]
        width = width * scenario_size[0] / arena_world_size[0]
        ctr = [
            ctr[0] * scenario_size[0] / arena_world_size[0],
            ctr[1] * scenario_size[1] / arena_world_size[1],
        ]
        box = vectors_rotation(
            np.array(get_box(width, height, [0.0, 0.0])).reshape(-1, 2).tolist(),
            0,
        )
        box = (np.array(box) + ctr).tolist()
        obj_poly = geom.Polygon([[p[0], p[1]] for p in box])
        rec = Rectangle(width=width, height=height, polygon=obj_poly, vertexes=box)
        rec.set_infor()
        rec.set_obj_graph()
        scenario.add_object(rec)

    return scenario, arena_entity_to_semantic_entity_map


if __name__ == "__main__":
    from pathlib import Path
    from arena_text_crowd.crowd_generation_pipeline.utils.field import Field

    world_path = Path(
        "/home/linh/ductai_nguyen_ws/Arena_ws/install/arena_simulation_setup/share/arena_simulation_setup/worlds/hospital_1"
    )
    arena_world = World(path=world_path)
    text_crowd_scenario, arena_entity_to_semantic_entity_map = (
        arena_world_to_text_crowd_scenario(arena_world=arena_world, wall_thickness=1.0)
    )

    fld = Field(text_crowd_scenario, grid_width=20.0)
    fld.field_visualization(
        np.zeros((fld.grid.grid_size[0], fld.grid.grid_size[1], 2)), None
    )
