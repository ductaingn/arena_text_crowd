import copy
import time
import math
from typing import Dict, List

import numpy as np

import rvo2

from ..utils.utils import make_ccw
from ..utils.visualization import Viewer
from ..input_models.scenario import Scenario
from ..input_models.constants import AllSemanticObjects
from .agent import Agent


class ORCAEnv:
    def __init__(
        self,
        scenario: Scenario,
        agent_list: List[Agent],
        visual: bool = False,
        draw_scale: float = 1.0,
    ):
        self.agent_num = len(agent_list)
        self.draw_scale = draw_scale
        self.visual = visual

        self.viewer: Viewer | None = None
        self.sensor = None
        self.current_scenario = None
        self.agent_dict: Dict[int, Agent] = {}  # {agent id in RVO2 simulator: Agent}
        self.agent_ids: List[int] = []
        self.agent_id_to_index: Dict[int, int] = {}
        self.time_step = 0

        self.sim = self.sim_prepare()
        for agent in agent_list:
            agent_id = self.add_agent_sim(*agent.pos)  # TODO: Vefiry
            agent.id = agent_id
            self.agent_dict.update({agent_id: agent})

        self.agent_ids = sorted(self.agent_dict.keys())
        self.agent_id_to_index = {
            agent_id: idx for idx, agent_id in enumerate(self.agent_ids)
        }

        self.reset(
            scenario=copy.deepcopy(scenario),
        )

    def reset(self, scenario: Scenario):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.sensor = None
        self.time_step = 0

        # preprocess the scenario and agent_settings
        scenario.extend()

        # reset env in sim
        self.current_scenario = copy.deepcopy(scenario)
        self.reset_sim_scenario(self.current_scenario)

        for id in self.agent_dict.keys():
            self.set_agent_params(id)

        # reset env in viewer
        if self.visual:
            self.viewer = Viewer(
                wind_size=(
                    int(scenario.scenario_config.window_size[0] * self.draw_scale),
                    int(scenario.scenario_config.window_size[1] * self.draw_scale),
                ),
                checker=(
                    int(scenario.scenario_config.window_size[0] * self.draw_scale),
                    int(scenario.scenario_config.window_size[1] * self.draw_scale),
                    [235, 235, 235],
                ),
            )
            self.reset_viewer(self.current_scenario)

    ######------ functions related to simulator------######
    def sim_prepare(self):
        dummy_default_agent = Agent()

        return rvo2.PyRVOSimulator(
            timeStep=1/25.0, # 25 fps
            neighborDist=dummy_default_agent.nb_Dist,
            maxNeighbors=dummy_default_agent.max_nbs,
            timeHorizon=dummy_default_agent.timeH,
            timeHorizonObst=dummy_default_agent.timeH_Obst,
            radius=dummy_default_agent.radius,
            maxSpeed=dummy_default_agent.maxSpd,
        )

    def reset_sim_scenario(self, scenario: Scenario):
        self.sim.clearObstacle()
        obs_list = scenario.get_all_obstacles()
        for obs_i in obs_list:
            self.sim.addObstacle(make_ccw([tuple(p) for p in obs_i]))
        self.sim.processObstacles()

    def sim_step(self):
        self.sim.doStep()

    def perform_action(self, actions):
        pre_ps = self.get_current_positions()
        actions = np.array(actions).reshape(-1, 2)
        if len(actions) != self.agent_num:
            raise ValueError(
                f"Expected {self.agent_num} actions, but got {len(actions)}"
            )
        # perform action
        for idx, agent_id in enumerate(self.agent_ids):
            dx = actions[idx][0]
            dy = actions[idx][1]
            len_a = math.sqrt(dx * dx + dy * dy)
            if len_a > self.agent_dict[agent_id].maxSpd:
                dx *= self.agent_dict[agent_id].maxSpd / len_a
                dy *= self.agent_dict[agent_id].maxSpd / len_a
            self.sim.setAgentPrefVelocity(agent_id, (dx, dy))
        self.sim_step()

        # update infor
        self.time_step += 1
        for idx, agent_id in enumerate(self.agent_ids):
            curr_p_i = self.sim.getAgentPosition(agent_id)
            self.agent_dict[agent_id].pos = np.array(curr_p_i).tolist()
            self.agent_dict[agent_id].add_history(
                pre_ps[idx].tolist(), actions[idx].tolist()
            )

    ######------ functions related to agent setting ------######
    def add_agent_sim(self, px, py):
        return self.sim.addAgent(pos=tuple([px, py]))

    def set_agent_params(self, agent_id: int):
        """
        Set agent parameters in simulator
        """
        agent = self.agent_dict[agent_id]
        self.sim.setAgentPosition(agent_id, tuple(agent.pos))
        self.sim.setAgentNeighborDist(agent_id, agent.nb_Dist)
        self.sim.setAgentMaxNeighbors(agent_id, agent.max_nbs)
        self.sim.setAgentTimeHorizon(agent_id, agent.timeH)
        self.sim.setAgentTimeHorizonObst(agent_id, agent.timeH_Obst)
        self.sim.setAgentRadius(agent_id, agent.radius)
        self.sim.setAgentMaxSpeed(agent_id, agent.maxSpd)
        if agent.vlcty is not None:
            self.sim.setAgentVelocity(agent_id, agent.vlcty)

    def set_agent_position(self, agent_id, pos):
        # set agent position in simulator
        self.sim.setAgentPosition(agent_id, tuple(pos))
        # update current agent infor
        self.agent_dict[agent_id].pos = copy.deepcopy(pos)

    ######------ functions related to agent information ------######
    def get_current_positions(self):
        current_positions = []
        for agent_id in self.agent_ids:
            pos = self.sim.getAgentPosition(agent_id)
            current_positions.append([pos[0], pos[1]])
        current_positions = np.array(current_positions)

        return current_positions

    def update_current_positions(self):
        for agent_id in self.agent_ids:
            pos = self.sim.getAgentPosition(agent_id)
            self.agent_dict[agent_id].pos = [pos[0], pos[1]]

    ######------ functions related to viewer ------######
    def reset_viewer(self, scenario: Scenario):
        scenario = copy.deepcopy(scenario)
        if not scenario.extended:
            scenario.extend()

        self.viewer.reset_array()

        # set scenarios
        for obstacle in (
            scenario.obstacle_dict[AllSemanticObjects.RECTANGLE]
            + scenario.obstacle_dict[AllSemanticObjects.TRIANGLE]
        ):
            self.viewer.add_obs(
                (copy.deepcopy(np.array(obstacle.vertexes)) * self.draw_scale)
                .reshape(1, -1)[0]
                .tolist()
            )
        for obstacle in scenario.obstacle_dict[AllSemanticObjects.CIRCLE]:
            self.viewer.add_obs(
                (copy.deepcopy(np.array(obstacle.edges)) * self.draw_scale)
                .reshape(1, -1)[0]
                .tolist()
            )

        for zebra_crossing in scenario.zebra_crossing_list:
            for box_i in zebra_crossing.zebra_lines_boxes:
                self.viewer.add_zebra_box(
                    (copy.deepcopy(np.array(box_i)) * self.draw_scale)
                    .reshape(1, -1)[0]
                    .tolist()
                )

        for passage in scenario.passages_list:
            for psg_obs in passage.obstacles:
                self.viewer.add_obs(
                    (copy.deepcopy(np.array(psg_obs)) * self.draw_scale)
                    .reshape(1, -1)[0]
                    .tolist()
                )

        for entrance in scenario.areas_dict[AllSemanticObjects.ENTRANCE]:
            area_color = [140, 235, 205]
            self.viewer.add_door_box(
                (copy.deepcopy(np.array(entrance.whole_box)) * self.draw_scale)
                .reshape(1, -1)[0]
                .tolist(),
                area_color,
            )
        for exit in scenario.areas_dict[AllSemanticObjects.EXIT]:
            area_color = [250, 155, 155]
            self.viewer.add_door_box(
                (copy.deepcopy(np.array(exit.whole_box)) * self.draw_scale)
                .reshape(1, -1)[0]
                .tolist(),
                area_color,
            )

        # set agents
        for agent_id in self.agent_ids:
            agent = self.agent_dict[agent_id]
            self.viewer.add_agent(
                pos=tuple(np.array(agent.pos) * self.draw_scale),
                rad=agent.radius * self.draw_scale,
                color=agent.color,
            )
            if agent.draw_goal:
                self.viewer.add_goal(
                    pos=tuple(np.array(agent.goal_pos * self.draw_scale)),
                    goal_size=agent.radius / 2 * self.draw_scale,
                    color=agent.color,
                )
            else:
                self.viewer.add_goal(pos=None)

        # set sensor
        if self.sensor is not None:
            self.viewer.sensor = self.sensor

    def render(self):
        for idx, agent_id in enumerate(self.agent_ids):
            agent = self.agent_dict[agent_id]
            self.viewer.agent_pos_array[idx] = (
                np.array(agent.pos) * self.draw_scale
            ).tolist()
            if agent.draw_goal:
                self.viewer.goal_pos_array[idx] = (
                    np.array(agent.goal_pos) * self.draw_scale
                ).tolist()
            else:
                self.viewer.goal_pos_array[idx] = None
        self.viewer.render()


if __name__ == "__main__":
    from ..input_models.scenario import ScenarioConfig

    orca_env = ORCAEnv(Scenario(ScenarioConfig()), 5)
    while True:
        orca_env.perform_action([[1, 1], [0.5, 1.4], [1, 1.4], [1.2, 1], [1.4, 1]])
        time.sleep(0.01)
