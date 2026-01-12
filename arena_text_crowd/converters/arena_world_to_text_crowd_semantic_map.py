from typing import List

from arena_simulation_setup.worlds.world import WorldDescription
from arena_models.impl.build.ObjectDatabaseBuilder import ObjectAnnotation

from arena_text_crowd.crowd_generation_pipeline.input_models.scenario import Scenario, ScenarioConfig
from arena_text_crowd.crowd_generation_pipeline.input_models.semantic.semantic_object import Entrance, Exit, Passage, Rectangle

def arena_world_to_text_crowd_semantic_map(
    arena_world: WorldDescription, 
    entrances: List[ObjectAnnotation], 
    exits: List[ObjectAnnotation]
) -> Scenario:
    """
    Convert an Arena world description, keeps corners and walls only and converts them to a Text-Crowd Scenario.

    Args:
        world_description : WorldDescription
            The world description to be converted.

    Returns:
        scenario : Scenario
    """
    scenario = Scenario(ScenarioConfig())
    return scenario
    for entrance in entrances:
        converted_entrance = Entrance(polygon=entrance.bounding_box)
        scenario.add_object(converted_entrance)

    for exit in exits: 
        converted_exit = Exit(polygon=exit.bounding_box)
        scenario.add_object(converted_exit)

    for zone in arena_world.zones:
        for wall in zone.walls:
            object = ObjectAnnotation(wall?)
            passage = Rectangle(polygon=object.bounding_box)
            scenario.add_object(passage)

        for door in zone.doors:
            object = ObjectAnnotation(door?)
            passage = Passage(polygon=object.bounding_box)
            scenario.add_object(passage)

    for obstacles in arena_world.all_static_entities:
        object = ObjectAnnotation(wall?)
        passage = Rectangle(polygon=object.bounding_box)
        scenario.add_object(passage)

    return scenario
