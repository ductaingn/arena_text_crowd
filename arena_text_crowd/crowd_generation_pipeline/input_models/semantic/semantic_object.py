from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import random
import math
import copy

import attrs

import numpy as np

import shapely.geometry as geom

from ...utils.utils import vector_rotation, vectors_rotation, get_box, get_circle


@attrs.define
class SemanticObject(ABC):
    polygon: Optional[geom.Polygon] = attrs.field(default=None, kw_only=True)
    points: List = attrs.field(init=False, default=[])
    edges: List = attrs.field(init=False, default=[])
    graph_box: List = attrs.field(init=False, default=[])
    points_idx_inRM: List = attrs.field(init=False, default=[])
    edges_idx_inRM: List = attrs.field(init=False, default=[])

    @classmethod
    @abstractmethod
    def random(
        cls,
        size_range: Tuple[float, float],
        area: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> "SemanticObject":
        """
        Parameters
        ----------
        size_range: range use for both height and width of object

        area: bounding box

        Returns
        -------
        An instance of this class
        """
        raise NotImplementedError

    @abstractmethod
    def set_obj_graph(self):
        raise NotImplementedError

    @abstractmethod
    def set_infor(self):
        raise NotImplementedError


@attrs.define
class Rectangle(SemanticObject):
    width: float
    height: float
    vertexes: List
    graph_buffer: int = 20
    center: List = attrs.field(init=False, default=[])

    @classmethod
    def random(cls, size_range, area) -> "Rectangle":
        ctr = [
            random.uniform(area[0][0], area[0][1]),
            random.uniform(area[1][0], area[1][1]),
        ]
        r, ag = random.uniform(size_range[0], size_range[1]) / 2, random.uniform(
            30.0, 60.0
        )
        w, h = (
            r * math.cos(ag / 180.0 * math.pi) * 2,
            r * math.sin(ag / 180.0 * math.pi) * 2,
        )
        rot = random.randint(-180, 180)
        box = vectors_rotation(
            np.array(get_box(w, h, [0.0, 0.0])).reshape(-1, 2).tolist(),
            rot / 180.0 * math.pi,
        )
        box = (np.array(box) + np.array(ctr)).tolist()
        obj_poly = geom.Polygon([[p[0], p[1]] for p in box])

        return Rectangle(width=w, height=h, polygon=obj_poly, vertexes=box)

    def set_obj_graph(self):
        edges_list = []
        points_list = []
        box_ = geom.Polygon([[p[0], p[1]] for p in self.vertexes]).buffer(
            self.graph_buffer, cap_style=3, join_style=2
        )
        points_list = np.array(list(box_.exterior.coords)[:-1]).tolist()
        for p_id in range(len(points_list)):
            edges_list.append([p_id, (p_id + 1) % len(points_list)])
            edges_list.append([(p_id + 1) % len(points_list), p_id])
        box_inner = geom.Polygon(
            [[p[0], p[1]] for p in copy.deepcopy(self.vertexes)]
        ).buffer(self.graph_buffer - 1e-3, cap_style=3, join_style=2)

        self.points = points_list
        self.edges = edges_list
        self.graph_box = copy.deepcopy(
            np.array(list(box_inner.exterior.coords)[:-1]).tolist()
        )

    def set_infor(self):
        rec_vs = np.array(copy.deepcopy(self.vertexes))
        rec_center = np.sum(rec_vs, axis=0) / len(rec_vs)
        self.center = rec_center.tolist()


@attrs.define
class Triangle(SemanticObject):
    vertexes: List
    graph_buffer: int = 20
    center: List = attrs.field(init=False, default=[])

    @classmethod
    def random(cls, size_range, area) -> "Triangle":
        ctr = [
            random.uniform(area[0][0], area[0][1]),
            random.uniform(area[1][0], area[1][1]),
        ]
        r = random.uniform(size_range[0], size_range[1]) / 2
        tri = []
        for rg_i in [[0.0, 60.0], [120.0, 180.0], [240.0, 300.0]]:
            ag = random.uniform(rg_i[0], rg_i[1])
            tri.append(
                [r * math.cos(ag / 180.0 * math.pi), r * math.sin(ag / 180.0 * math.pi)]
            )
        tri = (np.array(tri) + np.array(ctr)).tolist()
        obj_poly = geom.Polygon([[p[0], p[1]] for p in tri])

        return Triangle(vertexes=tri, polygon=obj_poly)

    def set_obj_graph(self):
        edges_list = []
        points_list = []
        tri_ = geom.Polygon(
            [[p[0], p[1]] for p in copy.deepcopy(self.vertexes)]
        ).buffer(self.graph_buffer, cap_style=3, join_style=2)
        points_list = np.array(list(tri_.exterior.coords)[:-1]).tolist()
        for p_id in range(len(points_list)):
            edges_list.append([p_id, (p_id + 1) % len(points_list)])
            edges_list.append([(p_id + 1) % len(points_list), p_id])
        tri_inner = geom.Polygon(
            [[p[0], p[1]] for p in copy.deepcopy(self.vertexes)]
        ).buffer(self.graph_buffer - 1e-3, cap_style=3, join_style=2)

        self.points = points_list
        self.edges = edges_list
        self.graph_box = copy.deepcopy(np.array(list(tri_inner.exterior.coords)[:-1]).tolist())

    def set_infor(self):
        rec_vs = np.array(copy.deepcopy(self.vertexes))
        rec_center = np.sum(rec_vs, axis=0) / len(rec_vs)
        self.center = rec_center.tolist()


@attrs.define
class Circle(SemanticObject):
    center: Tuple[float, float]
    radius: float
    graph_buffer: int = 20

    @classmethod
    def random(cls, size_range, area) -> "Circle":
        ctr = [
            random.uniform(area[0][0], area[0][1]),
            random.uniform(area[1][0], area[1][1]),
        ]
        r = random.uniform(size_range[0], size_range[1]) / 2
        circle_ps = (
            np.array(get_circle(r, fill=False, RES=32)).reshape(-1, 2) + np.array(ctr)
        ).tolist()
        obj_poly = geom.Polygon([[p[0], p[1]] for p in circle_ps])

        return Circle(center=ctr, radius=r, polygon=obj_poly)

    def set_obj_graph(self):
        edges_list = []
        points_list = []
        bbox_ = (
            np.array(
                get_box(self.radius * 2, self.radius * 2, copy.deepcopy(self.center))
            )
            .reshape(-1, 2)
            .tolist()
        )
        box_ = geom.Polygon([[p[0], p[1]] for p in bbox_]).buffer(
            self.graph_buffer, cap_style=3, join_style=2
        )
        points_list = np.array(list(box_.exterior.coords)[:-1]).tolist()
        for p_id in range(len(points_list)):
            edges_list.append([p_id, (p_id + 1) % len(points_list)])
            edges_list.append([(p_id + 1) % len(points_list), p_id])
        box_inner = geom.Polygon([[p[0], p[1]] for p in bbox_]).buffer(
            self.graph_buffer - 1e-3, cap_style=3, join_style=2
        )

        self.points = points_list
        self.edges = edges_list
        self.graph_box = copy.deepcopy(np.array(list(box_inner.exterior.coords)[:-1]).tolist())

    def set_infor(self):
        circle_edges = np.array(
            get_circle(
                r=self.radius,
                fill=False,
                RES=int(self.radius / 10) * 16,
            )
        ).reshape(-1, 2) + np.array(self.center)

        self.edges = circle_edges.tolist()


@attrs.define
class ZebraCrossing(SemanticObject):
    width: float
    height: float
    center: Tuple[float, float]
    zb_line_width: float
    rotation: int  # degree
    graph_buffer: int = 20
    whole_box: List = attrs.field(init=False, default=[])
    in_out_lines: List = attrs.field(init=False, default=[])
    zebra_lines_boxes: List = attrs.field(init=False, default=[])

    @classmethod
    def random(cls, size_range, area) -> "ZebraCrossing":
        ctr = [
            random.uniform(area[0][0], area[0][1]),
            random.uniform(area[1][0], area[1][1]),
        ]
        r, ag = random.uniform(size_range[0], size_range[1]) / 2, random.uniform(
            20.0, 40.0
        )
        w, h = (
            r * math.cos(ag / 180.0 * math.pi) * 2,
            r * math.sin(ag / 180.0 * math.pi) * 2,
        )
        zb_w = w / 15 - 1e-6
        rot = random.randint(-180, 180)
        box_ = vectors_rotation(
            np.array(get_box(w, h, [0.0, 0.0])).reshape(-1, 2).tolist(),
            rot / 180.0 * math.pi,
        )
        box_ = (np.array(box_) + np.array(ctr)).tolist()
        obj_poly = geom.Polygon([[p[0], p[1]] for p in box_])

        return ZebraCrossing(
            width=w,
            height=h,
            center=ctr,
            zb_line_width=zb_w,
            rotation=rot,
            polygon=obj_poly,
        )

    def set_obj_graph(self):
        edges_list = []
        points_list = []
        obj_buffer = copy.deepcopy(self)
        obj_buffer.width += obj_buffer.graph_buffer * 2
        obj_buffer.height += obj_buffer.graph_buffer * 2
        obj_buffer.set_infor()
        for l_i in obj_buffer.in_out_lines:
            alpha_ = random.uniform(0.4, 0.6)
            points_list.append(
                (np.array(l_i[0]) * alpha_ + np.array(l_i[1]) * (1 - alpha_)).tolist()
            )
        edges_list.append([0, 1])
        edges_list.append([1, 0])
        side_lines = [
            [
                copy.deepcopy(self.whole_box[0]),
                copy.deepcopy(self.whole_box[1]),
            ],
            [
                copy.deepcopy(self.whole_box[3]),
                copy.deepcopy(self.whole_box[2]),
            ],
        ]
        for l_i in side_lines:
            alpha_ = random.uniform(0.4, 0.6)
            points_list.append(
                (np.array(l_i[0]) * alpha_ + np.array(l_i[1]) * (1 - alpha_)).tolist()
            )
        edges_list.append([2, 3])
        edges_list.append([3, 2])
        obj_buffer_inner = copy.deepcopy(self)
        obj_buffer_inner.width += (obj_buffer_inner.graph_buffer - 1e-3) * 2
        obj_buffer_inner.height += (obj_buffer_inner.graph_buffer - 1e-3) * 2
        obj_buffer_inner.set_infor()
        self.points = points_list
        self.edges = edges_list 
        self.graph_box = copy.deepcopy(obj_buffer_inner.whole_box)

    def set_infor(self):  # TODO: Test
        w_ = self.width
        h_ = self.height
        whole_box = np.array(
            [[-w_ / 2, -h_ / 2], [w_ / 2, -h_ / 2], [w_ / 2, h_ / 2], [-w_ / 2, h_ / 2]]
        )
        for vc_id in range(len(whole_box)):
            whole_box[vc_id] = vector_rotation(
                whole_box[vc_id], self.rotation / 180 * math.pi
            )
        whole_box += np.array(self.center)

        in_out_lines = np.array(
            [
                [copy.deepcopy(whole_box[0]), copy.deepcopy(whole_box[3])],
                [copy.deepcopy(whole_box[1]), copy.deepcopy(whole_box[2])],
            ]
        )

        boxes_ = []
        x_ = -self.width / 2
        y_ = -self.height / 2
        lw_ = self.zb_line_width
        lh_ = self.height
        while x_ + lw_ <= self.width / 2:
            boxes_.append(
                [[x_, y_], [x_ + lw_, y_], [x_ + lw_, y_ + lh_], [x_, y_ + lh_]]
            )
            x_ += lw_ * 2
        boxes_ = np.array(boxes_)
        for box_id in range(len(boxes_)):
            for vc_id in range(len(boxes_[0])):
                boxes_[box_id][vc_id] = vector_rotation(
                    boxes_[box_id][vc_id], self.rotation / 180 * math.pi
                )
        boxes_ += np.array(self.center)

        self.whole_box = whole_box.tolist()
        self.in_out_lines = in_out_lines.tolist()
        self.zebra_lines_boxes = boxes_.tolist()


@attrs.define
class Passage(SemanticObject):
    width: float
    height: float
    center: Tuple[float, float]
    passage_width: float
    rotation: int  # degree
    graph_buffer: int = 20
    whole_box: List = attrs.field(init=False, default=[])
    in_out_lines: List = attrs.field(init=False, default=[])
    free_space: List = attrs.field(init=False, default=[])
    obstacles: List = attrs.field(init=False, default=[])

    @classmethod
    def random(cls, size_range, area) -> "Passage":
        ctr = [
            random.uniform(area[0][0], area[0][1]),
            random.uniform(area[1][0], area[1][1]),
        ]
        r, ag = random.uniform(size_range[0], size_range[1]) / 2, random.uniform(
            40.0, 60.0
        )
        w, h = (
            r * math.cos(ag / 180.0 * math.pi) * 2,
            r * math.sin(ag / 180.0 * math.pi) * 2,
        )
        ps_w = w * random.uniform(0.75, 0.85)
        rot = random.randint(-180, 180)
        box = vectors_rotation(
            np.array(get_box(w, h, [0.0, 0.0])).reshape(-1, 2).tolist(),
            rot / 180.0 * math.pi,
        )
        box = (np.array(box) + np.array(ctr)).tolist()
        obj_poly = geom.Polygon([[p[0], p[1]] for p in box])

        return Passage(
            width=w,
            height=h,
            center=ctr,
            passage_width=ps_w,
            rotation=rot,
            polygon=obj_poly,
        )

    def set_obj_graph(self):
        edges_list = []
        points_list = []
        obj_buffer = copy.deepcopy(self)
        obj_buffer.width += obj_buffer.graph_buffer * 2
        obj_buffer.height += obj_buffer.graph_buffer * 2
        obj_buffer.set_infor()
        for l_i in obj_buffer.in_out_lines:
            alpha_ = 0.5
            points_list.append(
                (np.array(l_i[0]) * alpha_ + np.array(l_i[1]) * (1 - alpha_)).tolist()
            )
        edges_list.append([0, 1])
        edges_list.append([1, 0])
        obj_buffer_inner = copy.deepcopy(self)
        obj_buffer_inner.width += (obj_buffer_inner.graph_buffer - 1e-3) * 2
        obj_buffer_inner.height += (obj_buffer_inner.graph_buffer - 1e-3) * 2
        obj_buffer_inner.set_infor()

        self.points = points_list
        self.edges = edges_list
        self.graph_box = copy.deepcopy(obj_buffer_inner.whole_box)

    def set_infor(self):  # TODO: Test
        w_ = self.width
        h_ = self.height
        free_w = self.passage_width
        whole_box = np.array(
            [[-w_ / 2, -h_ / 2], [w_ / 2, -h_ / 2], [w_ / 2, h_ / 2], [-w_ / 2, h_ / 2]]
        )
        free_box = np.array(
            [
                [-free_w / 2, -h_ / 2],
                [free_w / 2, -h_ / 2],
                [free_w / 2, h_ / 2],
                [-free_w / 2, h_ / 2],
            ]
        )
        for vc_id in range(len(whole_box)):
            whole_box[vc_id] = vector_rotation(
                whole_box[vc_id], self.rotation / 180 * math.pi
            )
            free_box[vc_id] = vector_rotation(
                free_box[vc_id], self.rotation / 180 * math.pi
            )
        whole_box += np.array(self.center)
        free_box += np.array(self.center)
        obstacles = np.array(
            [
                [
                    copy.deepcopy(whole_box[0]),
                    copy.deepcopy(free_box[0]),
                    copy.deepcopy(free_box[3]),
                    copy.deepcopy(whole_box[3]),
                ],
                [
                    copy.deepcopy(free_box[1]),
                    copy.deepcopy(whole_box[1]),
                    copy.deepcopy(whole_box[2]),
                    copy.deepcopy(free_box[2]),
                ],
            ]
        )
        in_out_lines = np.array(
            [
                [copy.deepcopy(free_box[0]), copy.deepcopy(free_box[1])],
                [copy.deepcopy(free_box[3]), copy.deepcopy(free_box[2])],
            ]
        )

        self.whole_box = whole_box.tolist()
        self.obstacles = obstacles.tolist()
        self.free_space = free_box.tolist()
        self.in_out_lines = in_out_lines.tolist()


@attrs.define
class Entrance(SemanticObject):
    width: float
    height: float
    center: Tuple[float, float]
    rotation: int  # degree
    graph_buffer = None
    whole_box: List = attrs.field(init=False, default=[])

    @classmethod
    def random(cls, size_range, area) -> "Entrance":
        w = random.uniform(size_range[0], size_range[1])
        h = w * 1.0
        edge_id = random.randint(0, 3)
        if edge_id == 0:
            ctr = [random.uniform(area[0][0], area[0][1]), area[1][0] - h / 2]
            rot = 0
        elif edge_id == 1:
            ctr = [random.uniform(area[0][0], area[0][1]), area[1][1] + h / 2]
            rot = 0
        elif edge_id == 2:
            ctr = [area[0][0] - h / 2, random.uniform(area[1][0], area[1][1])]
            rot = 90
        else:
            ctr = [area[0][1] + h / 2, random.uniform(area[1][0], area[1][1])]
            rot = 90
        box_ = vectors_rotation(
            np.array(get_box(w, h, [0.0, 0.0])).reshape(-1, 2).tolist(),
            rot / 180.0 * math.pi,
        )
        box_ = (np.array(box_) + np.array(ctr)).tolist()
        obj_poly = geom.Polygon([[p[0], p[1]] for p in box_])

        return Entrance(
            width=w,
            height=h,
            center=ctr,
            rotation=rot,
            polygon=obj_poly,
        )

    def set_obj_graph(self):
        self.points = [copy.deepcopy(self.center)]
        self.edges = []
        self.graph_box = []

    def set_infor(self):
        w_ = self.width
        h_ = self.height
        whole_box = np.array(
            [[-w_ / 2, -h_ / 2], [w_ / 2, -h_ / 2], [w_ / 2, h_ / 2], [-w_ / 2, h_ / 2]]
        )
        for vc_id in range(len(whole_box)):
            whole_box[vc_id] = vector_rotation(
                whole_box[vc_id], self.rotation / 180 * math.pi
            )
        whole_box += np.array(self.center)
        self.whole_box = whole_box.tolist()


@attrs.define
class Exit(SemanticObject):
    width: float
    height: float
    center: Tuple[float, float]
    rotation: int  # degree
    graph_buffer = None
    whole_box: List = attrs.field(init=False, default=[])

    @classmethod
    def random(cls, size_range, area) -> "Exit":
        w = random.uniform(size_range[0], size_range[1])
        h = w * 1.0
        edge_id = random.randint(0, 3)
        if edge_id == 0:
            ctr = [random.uniform(area[0][0], area[0][1]), area[1][0] - h / 2]
            rot = 0
        elif edge_id == 1:
            ctr = [random.uniform(area[0][0], area[0][1]), area[1][1] + h / 2]
            rot = 0
        elif edge_id == 2:
            ctr = [area[0][0] - h / 2, random.uniform(area[1][0], area[1][1])]
            rot = 90
        else:
            ctr = [area[0][1] + h / 2, random.uniform(area[1][0], area[1][1])]
            rot = 90
        box_ = vectors_rotation(
            np.array(get_box(w, h, [0.0, 0.0])).reshape(-1, 2).tolist(),
            rot / 180.0 * math.pi,
        )
        box_ = (np.array(box_) + np.array(ctr)).tolist()
        obj_poly = geom.Polygon([[p[0], p[1]] for p in box_])

        return Exit(
            width=w,
            height=h,
            center=ctr,
            rotation=rot,
            polygon=obj_poly,
        )

    def set_obj_graph(self):
        self.points = [copy.deepcopy(self.center)]
        self.edges = []
        self.graph_box = []

    def set_infor(self):
        w_ = self.width
        h_ = self.height
        whole_box = np.array(
            [[-w_ / 2, -h_ / 2], [w_ / 2, -h_ / 2], [w_ / 2, h_ / 2], [-w_ / 2, h_ / 2]]
        )
        for vc_id in range(len(whole_box)):
            whole_box[vc_id] = vector_rotation(
                whole_box[vc_id], self.rotation / 180 * math.pi
            )
        whole_box += np.array(self.center)
        self.whole_box = whole_box.tolist()