import copy
import random
import time
from typing import List

import attrs

import numpy as np

from scipy.interpolate import griddata

import pyglet.window.key

from arena_text_crowd.crowd_generation_pipeline.input_models.constants import (
    AllSemanticObjects,
)
from arena_text_crowd.crowd_generation_pipeline.input_models.semantic.semantic_object import (
    Exit,
)

from ..utils.utils import interp_grid_closest_4, interp_grid_fast
from ..utils.field import Field, Grid
from ..input_models.scenario import Scenario
from .agent import Agent
from .ORCA_env import ORCAEnv
from arena_text_crowd.crowd_generation_pipeline.input_models import scenario


@attrs.define
class GroupField:
    agent_ids: List[int]
    field: Field
    grid: Grid


class FieldEnv(ORCAEnv):
    def __init__(
        self,
        scenario: Scenario,
        agent_list: List[Agent],
        visual: bool = False,
        draw_scale: float = 1.0,
    ):
        super(FieldEnv, self).__init__(
            scenario=scenario,
            agent_list=agent_list,
            visual=visual,
            draw_scale=draw_scale,
        )
        self.agent_prefvs = []
        for agent_id in sorted(self.agent_dict.keys()):
            self.agent_prefvs.append(
                self.agent_dict[agent_id].pref_speed
            )  # TODO: Verify

    def reset(self, scenario: Scenario):
        super(FieldEnv, self).reset(scenario)
        self.agent_prefvs = []
        for agent_id in sorted(self.agent_dict.keys()):
            self.agent_prefvs.append(
                self.agent_dict[agent_id].pref_speed
            )  # TODO: Verify

    def perform_action_ORCAEnv(self, actions):
        super(FieldEnv, self).perform_action(actions)

    def perform_action(self, group_fields: List[GroupField]):
        agent_actions = np.zeros((self.agent_num, 2))

        start_time = time.time()
        for group_idx, _ in enumerate(group_fields):
            aids_gi = group_fields[group_idx].agent_ids
            field_gi = group_fields[group_idx].field
            grid_gi = group_fields[group_idx].grid
            grid_width = grid_gi.grid_width
            grid_size = grid_gi.grid_size
            grid_infor = grid_gi.grid_infor
            for agent_id in aids_gi:
                agent_pos = self.agent_dict[agent_id].pos
                assert (
                    0.0
                    < agent_pos[0]
                    < self.current_scenario.scenario_config.window_size[0]
                )
                assert (
                    0.0
                    < agent_pos[1]
                    < self.current_scenario.scenario_config.window_size[1]
                )
                agent_prefV = self.agent_dict[agent_id].pref_speed
                # get agent action based on bilinear interpolation
                closest_coord = [
                    int(agent_pos[0] / grid_width),
                    int(agent_pos[1] / grid_width),
                ]

                dirs = np.array(agent_pos) - np.array(
                    grid_infor[closest_coord[0]][closest_coord[1]]["center"]
                )
                for d_i, _ in enumerate(dirs):
                    if dirs[d_i] > 0:
                        dirs[d_i] = 1
                    elif dirs[d_i] < 0:
                        dirs[d_i] = -1
                    else:
                        dirs[d_i] = [-1, 1][random.randint(0, 1)]
                coords = [
                    [closest_coord[0], closest_coord[1]],
                    [closest_coord[0], int(closest_coord[1] + dirs[1])],
                    [int(closest_coord[0] + dirs[0]), int(closest_coord[1] + dirs[1])],
                    [int(closest_coord[0] + dirs[0]), closest_coord[1]],
                ]

                no_vec_cnt = 0
                for x_cd, y_cd in coords:
                    if (
                        x_cd < 0
                        or x_cd >= grid_size[0]
                        or y_cd < 0
                        or y_cd >= grid_size[1]
                        or grid_infor[x_cd][y_cd]["free"] == 1
                    ):
                        no_vec_cnt += 1
                if no_vec_cnt == len(coords):
                    labels = np.zeros((grid_size[0], grid_size[1]), dtype=int)
                    for cdi in coords:
                        labels[cdi[0]][cdi[1]] = 1
                    new_coords = []
                    for coord_i in coords:
                        for dir_x in [-1, 0, 1]:
                            for dir_y in [-1, 0, 1]:
                                new_coord_x = coord_i[0] + dir_x
                                new_coord_y = coord_i[1] + dir_y
                                if (
                                    labels[new_coord_x][new_coord_y] == 0
                                    and [new_coord_x, new_coord_y] not in new_coords
                                ):
                                    new_coords.append([new_coord_x, new_coord_y])
                                    labels[new_coord_x][new_coord_y] = 1
                    coords = copy.deepcopy(new_coords)

                points = []
                point_values = []
                fill_value = [0.0, 0.0]
                for x_cd, y_cd in coords:
                    center_ = [
                        x_cd * grid_width + grid_width / 2,
                        y_cd * grid_width + grid_width / 2,
                    ]
                    points.append(center_)
                    if (
                        x_cd < 0
                        or x_cd >= grid_size[0]
                        or y_cd < 0
                        or y_cd >= grid_size[1]
                        or grid_infor[x_cd][y_cd]["free"] == 1
                    ):
                        point_values.append(copy.deepcopy(fill_value))
                    else:
                        point_values.append(copy.deepcopy(field_gi[x_cd][y_cd]))
                points = np.array(points)
                point_values = np.array(point_values)

                mtd = "linear"
                agt_action = griddata(
                    points=points,
                    values=point_values,
                    xi=np.array([agent_pos]),
                    method=mtd,
                    fill_value=0,
                )[0]
                if abs(np.linalg.norm(agt_action) - 0.0) < 1e-6:
                    dis_list = np.linalg.norm(
                        np.array(points) - np.array(agent_pos), axis=1
                    )
                    sorted_ids = np.argsort(dis_list)
                    for id_ in sorted_ids:
                        if abs(np.linalg.norm(point_values[id_]) - 0.0) >= 1e-6:
                            agt_action = copy.deepcopy(point_values[id_])
                            break

                if np.linalg.norm(agt_action) < 1e-6:
                    agent_actions[agent_id] = np.array(agt_action)
                else:
                    agent_actions[agent_id] = (
                        np.array(agt_action) / np.linalg.norm(np.array(agt_action))
                    ) * agent_prefV
        print(f"time in computing actions: {time.time() - start_time}")

        # perform actions
        start_time = time.time()
        self.perform_action_ORCAEnv(agent_actions.tolist())
        print(f"time in performing actions: {time.time() - start_time}")

    def perform_action_fast(self, group_fields: List[GroupField]):
        agent_actions = np.zeros((self.agent_num, 2))
        all_agent_positions = np.array(self.get_current_positions())

        # get action for each agent based on the field of each group
        for group_idx, _ in enumerate(group_fields):
            aids_gi = group_fields[group_idx].agent_ids
            field_gi = group_fields[group_idx].field
            grid_gi = group_fields[group_idx].grid
            grid_width = grid_gi.grid_width
            grid_size = grid_gi.grid_size

            xp = np.arange(0, grid_size[0]) * grid_width + grid_width / 2
            yp = np.arange(0, grid_size[1]) * grid_width + grid_width / 2
            zp_0 = field_gi[:, :, 0]
            zp_1 = field_gi[:, :, 1]

            agent_ps_gi = all_agent_positions[aids_gi]
            actions_gi = np.concatenate(
                (
                    interp_grid_fast(
                        agent_ps_gi[:, 0], agent_ps_gi[:, 1], xp, yp, zp_0
                    ).reshape(-1, 1),
                    interp_grid_fast(
                        agent_ps_gi[:, 0], agent_ps_gi[:, 1], xp, yp, zp_1
                    ).reshape(-1, 1),
                ),
                axis=1,
            )

            # handle the actions that equal to zero
            zero_aids = np.argwhere(np.linalg.norm(actions_gi, axis=1) < 1e-6).reshape(
                1, -1
            )[0]
            if len(zero_aids) > 0:
                za_positions = copy.deepcopy(agent_ps_gi[zero_aids])
                za_actions = interp_grid_closest_4(
                    za_positions[:, 0], za_positions[:, 1], xp, yp, field_gi
                )
                actions_gi[zero_aids] = za_actions

            actions_gi = (
                actions_gi
                / np.linalg.norm(actions_gi, axis=1).reshape(-1, 1)
                * (np.array(self.agent_prefvs)[aids_gi]).reshape(-1, 1)
            )
            actions_gi = np.nan_to_num(actions_gi, 0.0)

            agent_actions[aids_gi] = actions_gi

        # perform actions
        self.perform_action_ORCAEnv(agent_actions.tolist())

    def set_viewer_guidance(self, guidance):
        if guidance is None:
            self.viewer.set_traj([], [])
            return
        if guidance["type"] == "lines":
            lines = guidance["params"]["lines"]
            trajs = []
            traj_colors = []
            for lidx, line_i in enumerate(lines):
                trajs.append(np.array(line_i).reshape(1, -1)[0].tolist())
                traj_colors.append([255, 0, 0])
            self.viewer.set_traj(np.array(trajs), np.array(traj_colors))

    def set_viewer_field(self, grid, field):
        if grid is None or field is None:
            self.viewer.set_arrows([], [])
            return
        arrows = []
        arrow_colors = []
        for i in range(len(field)):
            for j in range(len(field[0])):
                vec = field[i][j]
                arrow = np.array([[0, 0], vec])
                arrow = (arrow - vec / 2) * grid["grid_width"]
                arrow += np.array(grid["grid_infor"][i][j]["center"])
                arrows.append(arrow)
                arrow_colors.append([0, 0, 0])
        self.viewer.set_arrows(np.array(arrows), np.array(arrow_colors))


def sample_field_at(pos, grid: Grid, field: np.ndarray):
    gx = int(pos[0] / grid.grid_width)
    gy = int(pos[1] / grid.grid_width)
    if gx < 0 or gy < 0 or gx >= grid.grid_size[0] or gy >= grid.grid_size[1]:
        return None

    return field[gx][gy]


def reached_goal(pos, goal: Exit):
    if (
        goal.center[0] - goal.width <= pos[0] <= goal.center[0] + goal.width
        and goal.center[1] - goal.height <= pos[1] <= goal.center[1] + goal.height
    ):
        return True
    return False


if __name__ == "__main__":
    import os
    from pathlib import Path
    import pickle
    import cv2
    from arena_simulation_setup.tree.World import World
    from arena_text_crowd.converters import arena_world_to_text_crowd_scenario

    # Create Text-Crowd scenario from Arena World
    world_path = Path(
        "/home/linh/ductai_nguyen_ws/Arena_ws/install/arena_simulation_setup/share/arena_simulation_setup/worlds/hospital_1"
    )
    arena_world = World(path=world_path)
    scenario, entity_mapping = arena_world_to_text_crowd_scenario(
        arena_world=arena_world,
        scenario_size=(1024, 1024),
        wall_thickness=1.0,
        auto_entrance_exit_mode=True,
    )
    # with open(
    #     "/home/linh/ductai_nguyen_ws/Arena_ws/install/arena_simulation_setup/share/arena_simulation_setup/worlds/hospital_1/scenarios/tmp5rto5th2 (copy)/text_crowd_scenario___73nh6n.pkl",
    #     "rb",
    # ) as file:
    #     scenario: Scenario = pickle.load(file)
    window_size = scenario.scenario_config.window_size
    fld_env = FieldEnv(scenario, [], True)
    field = Field(scenario, 16)
    # velocity_fields = np.load(
    #     "/home/linh/ductai_nguyen_ws/Arena_ws/install/arena_simulation_setup/share/arena_simulation_setup/worlds/hospital_1/scenarios/tmp5rto5th2 (copy)/velocity_field_05huavvq.npy"
    # )
    # velocity_fields = np.transpose(velocity_fields, (0, 2, 1, 3))
    velocity_fields = np.load(
        "/home/linh/ductai_nguyen_ws/text_crowd_velocity_field.npy"
    )

    groups_fields = []
    for vel_field in velocity_fields:
        groups_fields.append(GroupField([], vel_field, field.grid))

    # Visualize velocity field
    N_FLOW = 64 * 64
    TRAIL_LEN = 12
    GROUP_ID = 0  # Group to be visualized
    x = np.random.uniform(0, window_size[0], N_FLOW)
    y = np.random.uniform(0, window_size[1], N_FLOW)
    flow_particles = []

    for _x, _y in zip(x, y):
        vl = fld_env.viewer.add_flow_particle(
            trail_len=TRAIL_LEN,
            color=[103, 58, 183],
        )

        flow_particles.append(
            {
                "vl": vl,
                "hist": [[_x, _y]] * TRAIL_LEN,
                "pos": np.array([_x, _y], dtype=float),
            }
        )

    keyboard = pyglet.window.key.KeyStateHandler()
    fps = 25
    width, height = window_size
    video_writer = cv2.VideoWriter(
        "/home/linh/ductai_nguyen_ws/velocity_field.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    video_duration = 50  # s
    n_frame = video_duration * fps
    current_frame = 0
    while not fld_env.viewer.closed:
        for i, p in enumerate(flow_particles):
            v = sample_field_at(
                p["pos"], groups_fields[0].grid, groups_fields[GROUP_ID].field
            )

            if (
                v is None
                # or reached_goal(
                #     p["pos"], scenario.areas_dict[AllSemanticObjects.EXIT][0]
                # )
                or np.linalg.norm(v) < 1e-6
                or p["pos"][0] < 0
                or p["pos"][1] < 0
                or p["pos"][0] > window_size[0]
                or p["pos"][1] > window_size[1]
            ):
                p["pos"] = [-1000, -1000]
                flow_particles.pop(i)
                fld_env.viewer.flow_lines.pop(i)
                continue

            v = v / np.linalg.norm(v)
            p["pos"] += v * 2.0  # step size

            p["hist"].insert(0, p["pos"].tolist())
            p["hist"] = p["hist"][:TRAIL_LEN]

            p["vl"].vertices = np.array(p["hist"]).reshape(-1).tolist()

        x = np.random.uniform(
            0,
            window_size[0],
            N_FLOW - len(flow_particles),
        )
        y = np.random.uniform(
            0,
            window_size[1],
            N_FLOW - len(flow_particles),
        )
        for _x, _y in zip(x, y):
            vl = fld_env.viewer.add_flow_particle(
                trail_len=TRAIL_LEN,
                color=[103, 58, 183],
            )

            flow_particles.append(
                {
                    "vl": vl,
                    "hist": [[_x, _y]] * TRAIL_LEN,
                    "pos": np.array([_x, _y], dtype=float),
                }
            )

        fld_env.render()

        frame = fld_env.viewer.capture_frame()
        video_writer.write(frame)

        time.sleep(0.04)

        current_frame += 1
        if current_frame > n_frame:
            break

    video_writer.release()
