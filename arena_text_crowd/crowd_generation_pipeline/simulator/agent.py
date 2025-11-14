from typing import Tuple, List

import attrs


@attrs.define
class Agent:
    id: int = None
    pos: Tuple[float, float] = [0.0, 0.0]
    goal_pos: Tuple[float, float] = [0.0, 0.0]
    nb_Dist: int = 50
    max_nbs: int = 10
    timeH: int = 5
    timeH_Obst: int = 10
    radius: float = 8.0
    pref_speed: float = 5.0
    maxSpd: float = 20.0
    vlcty: float = None
    color: Tuple[int, int, int] = (124, 79, 13)
    draw_goal: bool = False
    draw_traj: bool = False
    draw_sensor: bool = False
    traj_history: List = []
    action_history: List = []

    def add_history(self, current_position: Tuple[float, float], action: Tuple[float, float]):
        self.traj_history.append(current_position)
        self.action_history.append(action)
