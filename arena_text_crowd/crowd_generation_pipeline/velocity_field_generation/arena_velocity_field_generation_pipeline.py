import copy
import time
from typing import List, Tuple

from arena_simulation_setup.tree.World import WorldDescription
import attrs

import numpy as np

from task_generator.tasks.obstacles.prompt.prompt_utils.response_schema import (
    EmergencyResponseSchema,
)
from arena_hunav_sim_bridge.global_planner.path_finder import (
    PathFinder,
    build_grid_from_world,
)

from arena_text_crowd.crowd_generation_pipeline.utils.field import (
    Guidance,
    Constrains,
)
from arena_text_crowd.crowd_generation_pipeline.utils.curve_tracking_field import (
    CurveTrackingField,
)
from arena_text_crowd.converters.arena_world_to_text_crowd_scenario import (
    arena_world_to_text_crowd_scenario,
    get_arena_world_size,
)


@attrs.define
class ArenaVelocityFieldGenerationPipelineConfig:
    scenario_size: Tuple[int, int] = attrs.field(default=(1024, 1024))
    grid_width: int = attrs.field(default=16)


@attrs.define
class ArenaVelocityFieldGenerationPipeline:
    generation_pipeline_config: ArenaVelocityFieldGenerationPipelineConfig
    arena_world_description: WorldDescription

    def get_groups_start_centroids(self, response_schema: EmergencyResponseSchema):
        agent_pos = {}
        for agent in response_schema.hunav_agents:
            agent_pos[agent.name] = np.array([agent.pos[0], agent.pos[1]])

        groups = {}
        for node in response_schema.single_agent_nodes:
            group_id = node.attributes.velocity_field_group_id
            agent_name = node.attributes.agent_name

            if group_id not in groups.keys():
                groups[group_id] = {"agent_names": [agent_name]}
                groups[group_id]["centroid"] = agent_pos[agent_name]
            else:
                groups[group_id]["agent_names"].append(agent_name)
                groups[group_id]["centroid"] += agent_pos[agent_name]

        centroids = []
        for group_id in sorted(groups.keys()):
            group = groups[group_id]
            group["centroid"] /= len(group["agent_names"])
            centroids.append(tuple(group["centroid"].tolist()))

        return centroids

    def generate(
        self,
        response_schema: EmergencyResponseSchema,
        show: bool = False,
    ):
        print("Building grid from world...")
        matrix, origin = build_grid_from_world(self.arena_world_description)
        arena_world_size = get_arena_world_size(self.arena_world_description)
        groups_centroids = self.get_groups_start_centroids(response_schema)

        starts = groups_centroids
        goal = response_schema.exit_pos

        scenario_size = self.generation_pipeline_config.scenario_size

        scenario, _ = arena_world_to_text_crowd_scenario(
            arena_world=self.arena_world_description,
            scenario_size=scenario_size,
            wall_thickness=1.0,
            auto_entrance_exit_mode=False,
        )

        print("Generating velocity field...")
        start_time = time.time()
        pred_group_fields = []
        for start in starts:
            matrix = copy.deepcopy(matrix)
            origin = copy.deepcopy(origin)
            path_finder = PathFinder(matrix, origin)

            waypoints = path_finder.get_waypoints(start, goal)

            lines = []
            for i in range(len(waypoints) - 1):
                waypoint, next_waypoint = waypoints[i], waypoints[i + 1]
                lines.append(
                    [
                        [
                            waypoint[0] * scenario_size[0] / arena_world_size[0],
                            waypoint[1] * scenario_size[1] / arena_world_size[1],
                        ],
                        [
                            next_waypoint[0] * scenario_size[0] / arena_world_size[0],
                            next_waypoint[1] * scenario_size[1] / arena_world_size[1],
                        ],
                    ]
                )
            guidance = Guidance(
                type="lines",
                lines=lines,
                width=150,
                decay_rate=0.9,
            )

            constrains = Constrains(
                filter_path_n_average=0,
                closed_path_flag=False,
                pt_step_len=60,
                smooth_condition=600,
            )

            crv_fld = CurveTrackingField(
                scenario=scenario,
                grid_width=self.generation_pipeline_config.grid_width,
                reverse_direction=False,
                vr=1,
                kf=0.008,
                flag_follow_obstacle=True,
                epsilon=20,
                switch_dist_0=60,
                switch_dist=40,
                lidar_N=256,
            )

            field = crv_fld.get_field(guidance, constrains)

            pred_group_fields.append(field)

        end_time = time.time()
        print(f"Generating velocity field took {end_time - start_time:.2f}s")

        pred_group_fields = np.stack(pred_group_fields)

        # Access velocity at (x,y) with field[group_id, y, x]
        pred_group_fields = np.transpose(pred_group_fields, axes=(0, 2, 1, 3))

        return pred_group_fields


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

    # Generate agents trajectories
    generation_pipeline = ArenaVelocityFieldGenerationPipeline(
        ArenaVelocityFieldGenerationPipelineConfig(), arena_world.load()
    )
    starts = [(1.0, 2.0)]
    goals = [(15.0, 17.0)]

    pred_velocity_field = generation_pipeline.generate(starts, goals)
    print(pred_velocity_field.shape)
