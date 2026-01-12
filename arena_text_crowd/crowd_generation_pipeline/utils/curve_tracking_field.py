from copy import deepcopy

import attrs

import numpy as np

from .field import Field, Guidance, Constrains
from .utils import traj_smoothing
from .sensor import Sensor
from .distance_field import DistanceField


@attrs.define
class CurveTrackingField(Field):
    # Vector field:
    reverse_direction: bool = attrs.field(
        default=False,
        metadata={
            "description": "flag to invert the direction the curve will be followed"
        },
    )
    vr: float = attrs.field(
        default=1,
        metadata={"description": "reference forward speed for the vector field"},
    )
    kf: float = attrs.field(
        default=0.005,
        metadata={"description": "convergence gain of the vector field"},
    )
    # Collision avoidance considering the closest point:
    flag_follow_obstacle: bool = attrs.field(
        default=True,
        metadata={
            "description": "flag to enable the robot to follow an obstacle when it s blocking the vector field"
        },
    )
    epsilon: int = attrs.field(
        default=20,
        metadata={
            "description": "reference distance between the robot and the path being followed"
        },
    )
    switch_dist_0: int = attrs.field(
        default=60,
        metadata={
            "description": "distance from which the robot will start to follow the obstacle"
        },
    )
    switch_dist: int = attrs.field(
        default=40,
        metadata={
            "description": "distance from which the robot will start to follow the obstacle"
        },
    )
    lidar_N: int = attrs.field(default=128)
    vec_field_obj: DistanceField = attrs.field(init=False)

    @vec_field_obj.default
    def _vec_field_obj_factory(self):
        return DistanceField(
            self.vr,
            self.kf,
            self.reverse_direction,
            self.flag_follow_obstacle,
            self.epsilon,
            self.switch_dist_0,
            self.switch_dist,
        )

    def get_field(self, guidance: Guidance, constrains: Constrains = Constrains()):
        assert guidance.type == "lines"

        filter_path_n_average = constrains.filter_path_n_average
        closed_path_flag = constrains.closed_path_flag
        pt_step = constrains.pt_step_len
        smooth_cd = constrains.smooth_condition
        line_lengths = np.linalg.norm(
            np.array(guidance.lines)[:, 1] - np.array(guidance.lines)[:, 0],
            axis=1,
        )
        point_n = np.sum((line_lengths / pt_step).astype(int)) + 1
        if point_n < 10:
            pt_step = np.sum(line_lengths) / (10 + 2)

        path_points = []
        for line_i in guidance.lines:
            pt1 = deepcopy(line_i[0])
            pt2 = deepcopy(line_i[1])
            length_ = np.linalg.norm(np.array(pt2) - np.array(pt1))
            for step_ in range(int(length_ / pt_step)):
                pct = step_ / int(length_ / pt_step)
                pt_ = pct * np.array(pt2) + (1 - pct) * np.array(pt1)
                path_points.append((pt_[0], pt_[1], 0.0))
        path_points.append(
            (
                guidance.lines[-1][-1][0],
                guidance.lines[-1][-1][1],
                0.0,
            )
        )

        # path smoothing
        path_2d = np.array(path_points)[:, 0:2]
        path_2d_smooth = traj_smoothing(path_2d, s_=smooth_cd)
        path_points = np.array(path_points)
        path_points[:, 0:2] = np.array(path_2d_smooth)
        path_points = path_points.tolist()

        if self.flag_follow_obstacle:
            lidar_ = Sensor(
                agent_r=0,
                obs_all=self.scenario.get_all_obstacles(),
                sensorRange=max(self.switch_dist_0, self.switch_dist) * 2,
                N=self.lidar_N,
            )

        grid_map = self.grid.grid_map
        grid_infor = self.grid.grid_infor
        grid_size = self.grid.grid_size
        field = np.zeros((grid_size[0], grid_size[1], 2))
        for xid in range(grid_size[0]):
            for yid in range(grid_size[1]):
                if grid_map[xid][yid] == 1:
                    continue
                center = deepcopy(grid_infor[xid][yid]["center"])
                pos = [center[0], center[1], 0]
                self.vec_field_obj.set_pos(pos)
                self.vec_field_obj.set_path(
                    path_points, 0, filter_path_n_average, closed_path_flag
                )

                if self.flag_follow_obstacle:
                    lidar_infor = lidar_.get_sensor_reading([[pos[0], pos[1]]])[0]
                    if 0.0 in lidar_infor:
                        lidar_infor.remove(0.0)
                    min_dir_id = np.argmin(lidar_infor)
                    if (
                        abs(
                            lidar_infor[min_dir_id]
                            - max(self.switch_dist_0, self.switch_dist) * 2
                        )
                        <= 1e-3
                    ):
                        closest_p_from_obs = [1e5, 1e5, 1e5]
                    else:
                        inter_p = np.array(lidar_.dirs[min_dir_id]) * lidar_infor[
                            min_dir_id
                        ] + np.array([pos[0], pos[1]])
                        closest_p_from_obs = [inter_p[0], inter_p[1], 0]
                    self.vec_field_obj.set_closest(closest_p_from_obs)

                if self.vec_field_obj.is_ready():
                    [Vx, Vy, Vz, terminated] = self.vec_field_obj.vec_field_path()
                    Vec = np.nan_to_num(
                        np.array([Vx, Vy]) / np.linalg.norm(np.array([Vx, Vy])), 0
                    )
                    field[xid][yid] = deepcopy(Vec)

        return field


if __name__ == "__main__":
    from ..input_models.scenario import Scenario, ScenarioConfig
    from ..input_models.constants import AllSemanticObjects
    from ..input_models.semantic.semantic_object import Rectangle, Triangle, Circle

    scenario_test = Scenario(
        ScenarioConfig(window_size=[800, 800]),
        obstacle_dict={
            AllSemanticObjects.RECTANGLE: [
                Rectangle(
                    width=None,
                    height=None,
                    vertexes=[[100, 400], [200, 400], [200, 500], [100, 500]],
                )
            ],
            AllSemanticObjects.TRIANGLE: [
                Triangle(vertexes=[[150, 150], [250, 150], [150, 250]])
            ],
            AllSemanticObjects.CIRCLE: [Circle(center=[400.0, 400.0], radius=50.0)],
        },
    )

    guidance = Guidance(
        type="lines",
        lines=[
            [[50, 50], [50, 600]],
            [[50, 600], [300, 600]],
            [[300, 600], [300, 250]],
            [[300, 250], [500, 250]],
            [[500, 250], [500, 450]],
            [[500, 450], [750, 750]],
        ],
        width=150,
        decay_rate=0.9,
    )

    constrains = Constrains(
        filter_path_n_average=0,
        closed_path_flag=False,
        pt_step_len=60,
        smooth_condition=600,
    )

    grid_width = 20.0

    crv_fld = CurveTrackingField(
        scenario=scenario_test,
        grid_width=grid_width,
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
    crv_fld.field_visualization(field, guidance)
