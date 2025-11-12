import copy
import time
import math
from typing import Dict

import numpy as np

import rvo2

from ..utils.utils import make_ccw
from ..input_models.scenario import Scenario
from .agent import Agent


class ORCAEnv:
    def __init__(
        self,
        scenario: Scenario,
        agent_num: int,
        draw_scale: float = 1.0,
    ):
        self.agent_num = agent_num
        self.draw_scale = draw_scale

        self.sensor = None
        self.current_scenario = None
        self.agent_dict: Dict[int, Agent] = {}  # {agent id in RVO2 simulator: Agent}
        self.agent_id_list = []  # Hash map of agents' ids in RVO2 simulator
        self.time_step = 0

        self.sim = self.sim_prepare()
        for _ in range(self.agent_num):
            agent_idx = self.add_agent_sim(0, 0)
            self.agent_id_list.append(agent_idx)
            self.agent_dict.update({agent_idx: Agent(agent_idx)})

        self.reset(
            scenario=copy.deepcopy(scenario),
        )

    def reset(self, scenario: Scenario):
        self.sensor = None
        self.time_step = 0

        # preprocess the scenario and agent_settings
        scenario.extend()

        # reset env in sim
        self.current_scenario = copy.deepcopy(scenario)
        self.reset_sim_scenario(self.current_scenario)

        for idx in self.agent_id_list:
            self.set_agent_params(idx)

    ######------ functions related to simulator------######
    def sim_prepare(self):
        dummy_default_agent = Agent()

        return rvo2.PyRVOSimulator(
            timeStep=1,
            neighborDist=dummy_default_agent.nb_Dist,
            maxNeighbors=dummy_default_agent.max_nbs,
            timeHorizon=dummy_default_agent.timeH,
            timeHorizonObst=dummy_default_agent.timeH_Obst,
            radius=dummy_default_agent.radius,
            maxSpeed=dummy_default_agent.maxSpd,
        )

    def reset_sim_scenario(self, scenario: Scenario):
        self.sim.clearObstacle()
        obs_list = scenario.get_all_obstacles(scenario_in=copy.deepcopy(scenario))
        for obs_i in obs_list:
            self.sim.addObstacle(make_ccw([tuple(p) for p in obs_i]))
        self.sim.processObstacles()

    def sim_step(self):
        self.sim.doStep()

    def perform_action(self, actions):
        pre_ps = self.get_current_positions()
        actions = np.array(actions).reshape(-1, 2)
        # perform action
        for agent_idx, agent_id in enumerate(self.agent_id_list):
            dx = actions[agent_idx][0]
            dy = actions[agent_idx][1]
            len_a = math.sqrt(dx * dx + dy * dy)
            if len_a > self.agent_dict[agent_id].maxSpd:
                dx *= self.agent_dict[agent_id].maxSpd / len_a
                dy *= self.agent_dict[agent_id].maxSpd / len_a
            self.sim.setAgentPrefVelocity(agent_id, (dx, dy))
        self.sim_step()

        # update infor
        self.time_step += 1
        for agent_idx, agent_id in enumerate(self.agent_id_list):
            curr_p_i = self.sim.getAgentPosition(agent_id)
            self.agent_dict[agent_id].pos = np.array(curr_p_i).tolist()
            self.agent_dict[agent_id].add_pos_to_trajectory(
                [pre_ps[agent_idx].tolist(), actions[agent_idx].tolist()]
            )

    ######------ functions related to agent setting ------######
    def add_agent_sim(self, px, py):
        return self.sim.addAgent(pos=tuple([px, py]))

    def set_agent_params(self, a_idx: int):
        """
        Set agent parameters in simulator
        """
        agent_id = self.agent_id_list[a_idx]
        agent = self.agent_dict[agent_id]
        self.sim.setAgentPosition(agent_id, tuple(agent.pos))
        self.sim.setAgentNeighborDist(agent_id, agent.nb_Dist)
        self.sim.setAgentMaxNeighbors(agent_id, agent.max_nbs)
        self.sim.setAgentTimeHorizon(agent_id, agent.timeH)
        self.sim.setAgentTimeHorizonObst(agent_id, agent.timeH_Obst)
        self.sim.setAgentRadius(agent_id, agent.radius)
        self.sim.setAgentMaxSpeed(agent_id, agent.maxSpd)
        self.sim.setAgentVelocity(agent_id, agent.vlcty)

    def set_agent_position(self, a_idx, pos):
        agent_id = self.agent_id_list[a_idx]
        # set agent position in simulator
        self.sim.setAgentPosition(agent_id, tuple(pos))
        # update current agent infor
        self.agent_dict[a_idx].pos = copy.deepcopy(pos)

    ######------ functions related to agent information ------######
    def get_current_positions(self):
        current_positions = []
        for agent_id in self.agent_id_list:
            pos = self.sim.getAgentPosition(agent_id)
            current_positions.append([pos[0], pos[1]])
        current_positions = np.array(current_positions)

        return current_positions

    def update_current_positions(self):
        for agent_id in self.agent_id_list:
            pos = self.sim.getAgentPosition(agent_id)
            self.agent_dict[agent_id].pos = [pos[0], pos[1]]


if __name__ == "__main__":
    from ..input_models.scenario import ScenarioConfig

    orca_env = ORCAEnv(Scenario(ScenarioConfig()), 5)
    while True:
        orca_env.perform_action([[1, 1], [0.5, 1.4], [1, 1.4], [1.2, 1], [1.4, 1]])
        time.sleep(0.01)
