import enum
from typing import Generic, List, TypeVar, Tuple

import attrs


class AllSemanticObjects(enum.Enum):
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    CIRCLE = "circle"
    ZEBRA_CROSSING = "zebra_crossing"
    PASSAGE_OBSTACLE = "passage_obs"
    PASSAGE_FREE = "passage_free"
    ENTRANCE = "entrance"
    EXIT = "exit"
    PASSAGE = "passage"  # Additional, TODO: Test


SUPPORTED_ACTIONS = {
    AllSemanticObjects.RECTANGLE: (
        "passes",
        "passes by",
        "moves past",
        "bypasses",
        "circles around",
    ),
    AllSemanticObjects.TRIANGLE: (
        "passes",
        "passes by",
        "moves past",
        "bypasses",
        "circles around",
    ),
    AllSemanticObjects.CIRCLE: (
        "passes",
        "passes by",
        "moves past",
        "bypasses",
        "circles around",
    ),
    AllSemanticObjects.ZEBRA_CROSSING: (
        "passes",
        "passes by",
        "moves past",
        "crosses",
        "walks across",
        "moves across",
    ),
    AllSemanticObjects.PASSAGE: (
        "passes",
        "passes by",
        "moves past",
        "moves through",
        "walks through",
        "passes through",
        "travels through",
    ),
    AllSemanticObjects.ENTRANCE: (
        "enters from",
        "gets in from",
        "moves from",
    ),
    AllSemanticObjects.EXIT: (
        "exits through",
        "leaves through",
        "quits through",
    ),
}


class ParametersMode(enum.Enum):
    SIMPLE = "simple"
    HARD = "hard"


ParametersModeT = TypeVar("ParametersModeT", bound=ParametersMode)


@attrs.define
class AdaptiveW(Generic[ParametersModeT]):
    mode: ParametersMode  # Currently unused

    v_w = 0.95
    e_w = 0.7


@attrs.define
class PathConstrains(Generic[ParametersModeT]):
    mode: ParametersMode  # Currently unused

    start_p: Tuple[float, float]
    goal_p: Tuple[float, float]
    line_angle_lim: float = attrs.field(init=False, default=89.0)
    safe_dis: float = attrs.field(init=False, default=100.0)
    path_len_range: Tuple[int, int] = attrs.field(init=False, default=(1, 9))
    adaptive_w: AdaptiveW = attrs.field(init=False)

    @adaptive_w.default
    def _adaptive_w_factory(self):
        return AdaptiveW(self.mode)


@attrs.define
class RoadmapParams(Generic[ParametersModeT]):
    mode: ParametersMode  # Currently unused
    path_constrains: PathConstrains

    roadmap_sample_n: int = attrs.field(init=False, default=0)
    roadmap_nb_dis: int = attrs.field(init=False, default=2048)
    path_relax_dis: float = attrs.field(init=False, default=30.0)


@attrs.define
class DropOutPS:
    mode: ParametersMode  # Currently unused

    action_loc: float = attrs.field(init=False, default=0.6)
    obj_loc: float = attrs.field(init=False, default=0.1)
    action_dir: float = attrs.field(init=False, default=0.4)


@attrs.define
class GroupSizeRange:
    mode: ParametersMode  # Currently unused

    tiny: Tuple[int, int] = attrs.field(init=False, default=(2, 3))
    small: Tuple[int, int] = attrs.field(init=False, default=(4, 7))
    big: Tuple[int, int] = attrs.field(init=False, default=(8, 15))
    large: Tuple[int, int] = attrs.field(init=False, default=(16, 30))


@attrs.define
class BehaviorParams(Generic[ParametersModeT]):
    mode: ParametersMode  # Currently unused

    mid_scale: float = attrs.field(init=False, default=0.25)
    inner_scale: float = attrs.field(init=False, default=0.66)
    global_adj_num: int = attrs.field(init=False, default=8)
    local_adj_num: int = attrs.field(init=False, default=8)
    dir_adj_num: int = attrs.field(init=False, default=8)
    drop_out_ps: DropOutPS = attrs.field(init=False)
    group_size_range: GroupSizeRange = attrs.field(init=False)

    @drop_out_ps.default
    def _drop_out_ps_factory(self):
        return DropOutPS(self.mode)

    @group_size_range.default
    def _group_size_range_factory(self):
        return GroupSizeRange(self.mode)


class Field(enum.Enum):
    CTF = "CurveTrackingField"
    NF = "NavigationField"  # TODO: Verify


@attrs.define
class CTFConstrains:
    filter_path_n_average: int = attrs.field(init=False, default=0)
    closed_path_flag: bool = attrs.field(
        init=False,
        default=False,
    )
    pt_step_len: int = attrs.field(
        init=False,
        default=15,
        metadata={"description": "distance between two discrete points in path"},
    )
    smooth_condition: int = attrs.field(
        init=False,
        default=600,
        metadata={
            "description": "smooth condition for trajectory smoothing, should range in [15, 30]"
        },
    )


@attrs.define
class CTFGuidance:
    type: str = attrs.field(init=False, default="lines")
    lines: List = attrs.field(init=False, default=None)
    width: int = attrs.field(init=False, default=100)  # Currently unused
    decay_rate: float = attrs.field(init=False, default=0.95)  # Currently unused


@attrs.define
class FieldParams(Generic[ParametersModeT]):
    mode: ParametersMode  # Currently unused
    field_use: Field = attrs.field(default=Field.CTF)

    reverse_direction: bool = attrs.field(init=False, default=False)
    vr: float = attrs.field(init=False, default=1.0)
    kf: float = attrs.field(
        init=False,
        default=0.05,
        metadata={
            "description": "convergence rate, should be one of (0.05, 0.008, 0.015)"
        },
    )
    flag_follow_obstacle: bool = attrs.field(init=False, default=True)
    epsilon: float = attrs.field(init=False, default=0.0)
    switch_dist_0: float = attrs.field(
        init=False,
        default=40.0,
        metadata={"description": "should be one of (40.0, 60.0)"},
    )
    switch_dist: float = attrs.field(
        init=False,
        default=40.0,
        metadata={"description": "convergence rate, should be one of (40.0, 60.0)"},
    )
    lidar_N: int = attrs.field(init=False, default=256)
    constrains_CTF: CTFConstrains = attrs.field(init=False, default=CTFConstrains())
    guidance_CTF: CTFGuidance = attrs.field(init=False, default=CTFGuidance())


@attrs.define
class EnvironmentParams:
    grid_width_map = 16
    grid_width_field = 16
