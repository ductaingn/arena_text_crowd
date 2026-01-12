from typing import Tuple, List
import copy

import attrs

import numpy as np

import shapely.geometry as geom

import rasterio
from rasterio.features import geometry_mask

from ..input_models.scenario import Scenario
from ..utils.visualization import Viewer


@attrs.define
class Grid:
    grid_width: int
    grid_size: Tuple[int, int]
    grid_map: np.ndarray
    grid_infor: np.ndarray


# # Example guidance:
# guidance = {
#     "type": "lines",
#     "params": {
#         "lines":[[[50, 50], [50, 600]],[[50, 600], [300, 600]], [[300, 600], [300, 250]], [[300, 250], [500, 250]]],
#         # "width": 150,   # influence width
#         # "decay_rate": 0.9,   # decay rate of the guidance field along width
#     }
# }
@attrs.define
class Guidance:
    type: str
    lines: List[List[List[int]]] | np.ndarray
    width: int | None = None
    decay_rate: float | None = None


# # Example constrains:
# constrains = {
#     "filter_path_n_average": ,
#     "closed_path_flag": False,
#     "pt_step_len": 20,  # pt_step_len
#     "smooth_condition": 600,
# }
@attrs.define
class Constrains:
    filter_path_n_average: int = 0  # Number of points to use in the average filter (it is forced to be an odd number) - if 0 the path is not filtered
    closed_path_flag: bool = False  # Flag to indicate if the path is closed or not
    pt_step_len: int = 20  # The length between two neighbor points
    smooth_condition: int = 600  # The smooth condition for trajectory smoothing


@attrs.define
class Field:
    scenario: Scenario
    grid_width: int
    grid: Grid = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.scenario = copy.deepcopy(self.scenario)
        self.grid = self.space_discretization()

    def space_discretization(self) -> Grid:
        grid = {}
        wind_size = copy.deepcopy(self.scenario.scenario_config.window_size)
        assert (int(wind_size[0] / self.grid_width)) * self.grid_width == wind_size[0]
        assert (int(wind_size[1] / self.grid_width)) * self.grid_width == wind_size[1]
        grid_size = [
            int(wind_size[0] / self.grid_width),
            int(wind_size[1] / self.grid_width),
        ]
        grid["grid_size"] = copy.deepcopy(grid_size)

        grid_map = np.zeros((grid_size[0], grid_size[1]), dtype=int)
        grid_infor = []

        obs_list = self.scenario.get_all_obstacles()
        if len(obs_list) != 0:
            obs_polys = []
            for obs_i in obs_list:
                obs_polys.append(geom.Polygon(obs_i))
            transform_ = rasterio.transform.from_bounds(
                0, 0, wind_size[0], wind_size[1], grid_size[0], grid_size[1]
            )
            geom_mask = rasterio.features.geometry_mask(
                obs_polys,
                out_shape=(grid_size[1], grid_size[0]),
                transform=transform_,
                all_touched=True,
            )
            geom_mask = np.flip(geom_mask, axis=0).transpose(1, 0)
            grid_map[~geom_mask] = 1
        for i in range(grid_size[0]):
            grid_infor_rowi = []
            for j in range(grid_size[1]):
                center_ = [
                    i * self.grid_width + self.grid_width / 2,
                    j * self.grid_width + self.grid_width / 2,
                ]
                edge_len = self.grid_width
                box_ = [
                    [center_[0] - edge_len / 2, center_[1] - edge_len / 2],
                    [center_[0] + edge_len / 2, center_[1] - edge_len / 2],
                    [center_[0] + edge_len / 2, center_[1] + edge_len / 2],
                    [center_[0] - edge_len / 2, center_[1] + edge_len / 2],
                ]
                grid_infor_rowi.append(
                    {
                        "center": center_,
                        "edge_len": edge_len,
                        "box": box_,
                        "free": grid_map[i][j],
                    }
                )
            grid_infor.append(grid_infor_rowi)
        grid_infor = np.array(grid_infor)

        return Grid(
            grid_width=self.grid_width,
            grid_size=grid_size,
            grid_map=grid_map,
            grid_infor=grid_infor,
        )

    def get_field(self, guidance: Guidance, constrains: Constrains | None = None):
        raise NotImplementedError

    def field_visualization(self, field, guidance: Guidance | None = None):
        # visualization
        wind_size = copy.deepcopy(self.scenario.scenario_config.window_size)
        grid_size = copy.deepcopy(self.grid.grid_size)
        grid_width = self.grid_width

        viewer = Viewer(wind_size=tuple((int(wind_size[0]), int(wind_size[1]))))
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if self.grid.grid_infor[i][j]["free"] == 1:
                    viewer.add_obs(
                        np.array(self.grid.grid_infor[i][j]["box"])
                        .reshape(1, -1)[0]
                        .tolist()
                    )
                else:
                    viewer.add_obs(
                        np.array(self.grid.grid_infor[i][j]["box"])
                        .reshape(1, -1)[0]
                        .tolist(),
                        [200, 200, 200],
                    )
        arrows = []
        arrow_colors = []
        for i in range(len(field)):
            for j in range(len(field[0])):
                vec = field[i][j]
                arrow = np.array([[0, 0], vec])
                arrow = (arrow - vec / 2) * grid_width
                arrow += np.array(self.grid.grid_infor[i][j]["center"])
                arrows.append(arrow)
                arrow_colors.append([0, 0, 0])
        viewer.set_arrows(np.array(arrows), np.array(arrow_colors))

        if guidance is not None and guidance.type == "lines":
            lines = guidance.lines
            trajs = []
            traj_colors = []
            for lidx, line_i in enumerate(lines):
                trajs.append(np.array(line_i).reshape(1, -1)[0].tolist())
                traj_colors.append([255, 0, 0])
            viewer.set_traj(np.array(trajs), np.array(traj_colors))

        while 1:
            viewer.render()
            if viewer.closed:
                break

    def fields_blending(self, fields, weights):
        return np.average(
            np.array(fields), axis=0, weights=np.array(weights) / np.sum(weights)
        )


if __name__ == "__main__":
    from ..input_models.scenario import ScenarioConfig
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
    grid_width = 20.0

    fld = Field(scenario_test, grid_width)
    fld.field_visualization(
        np.zeros((fld.grid.grid_size[0], fld.grid.grid_size[1], 2)), None
    )
