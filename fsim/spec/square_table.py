"""square_table and one_leg task specs.

Design source: FurnitureBench (Heo et al., RSS 2023; MIT License). Dimensions, masses, scene
layout, randomization levels, assembled poses and tolerances follow its square_table design;
everything here is re-expressed in this project's frames (see fsim/spec/types.py).
"""
from __future__ import annotations

import math

import numpy as np

from fsim.geometry import mat_to_quat, orthonormalize, rot_y
from fsim.spec.types import (
    IDENTITY_QUAT, AssemblyTarget, BoxObstacle, CameraSpec, LegPartSpec, Placement,
    RandomizationSpec, RandomLevel, SceneSpec, ScrewSpec, TaskSpec, ThreadSpec, TrayPartSpec,
)

TOP = "square_table_top"
LEGS = tuple(f"square_table_leg{i}" for i in range(1, 5))

HOLE_OFFSET = 0.05625          # hole axes at (+-56.25, +-56.25) mm
SEAT_HEIGHT = 0.046875         # leg origin above the top origin when screwed in
PITCH = 0.009375               # right-handed threads, 3 x 3.125 mm

# The tabletop is an open tray seen from below: 6.25 mm plate, 6.25 mm rim, and a threaded boss
# around each blind hole (the hole floor is the plate).
TOP_SPEC = TrayPartSpec(
    name=TOP,
    size=(0.1625, 0.1625, 0.03125),
    mass=0.151,
    hole_centers=tuple((sx * HOLE_OFFSET, sy * HOLE_OFFSET) for sx in (1, -1) for sy in (1, -1)),
    plate_thickness=0.00625,
    rim_width=0.00625,
    boss_outer_radius=0.01625,
    bore_radius=0.0125,
    thread=ThreadSpec(pitch=PITCH, minor_radius=0.0109, major_radius=0.0147),
)


def _leg(name: str) -> LegPartSpec:
    return LegPartSpec(
        name=name,
        width=0.030,
        body_length=0.0625,
        screw_length=0.025,
        mass=0.0231,
        peg_radius=0.0115,
        thread=ThreadSpec(pitch=PITCH, minor_radius=0.0072, major_radius=0.0128),
    )


LEG_SPECS = tuple(_leg(n) for n in LEGS)

SCREW = ScrewSpec(parent=TOP, children=LEGS, pitch=PITCH, seat_height=SEAT_HEIGHT)


def _corner_targets(child: str) -> AssemblyTarget:
    poses = tuple(((sx * HOLE_OFFSET, sy * HOLE_OFFSET, SEAT_HEIGHT), IDENTITY_QUAT)
                  for sx in (1, -1) for sy in (1, -1))
    return AssemblyTarget(parent=TOP, child=child, rel_poses=poses)


# Layout in W: tabletop holes-up near the fence, legs lying along +x with the stud toward the robot.
LEG_REST_QUAT = tuple(mat_to_quat(rot_y(math.pi / 2)))
LAYOUT = (
    Placement(TOP, (0.2415, 0.0, 0.015625), IDENTITY_QUAT),
    Placement(LEGS[0], (0.0715, -0.20, 0.015), LEG_REST_QUAT),
    Placement(LEGS[1], (0.0715, -0.12, 0.015), LEG_REST_QUAT),
    Placement(LEGS[2], (0.0715, 0.12, 0.015), LEG_REST_QUAT),
    Placement(LEGS[3], (0.0715, 0.20, 0.015), LEG_REST_QUAT),
)

# --- High-randomness presets ---------------------------------------------------------------
# FurnitureBench defines three measured layouts for "high" randomness. They are recorded in its
# AprilTag base-tag frame and part frames, and converted here into W and our part frames:
#   tag frame -> W:   p_W = (0.0015 + p_y, p_x, -p_z),  R_W = M @ R_tag
#   their part frame -> ours: our (x, y, z) axes = their (x, -z, y)  ->  R_ours = R_W @ C
_M = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

_HIGH_TAG_DATA = {
    TOP: [
        ((0.00165764, 0.23166147, -0.01571113), ((-0.22164249, -0.00542635, -0.97511292), (0.97511977, -0.00533661, -0.22161436), (-0.00400126, -0.99997103, 0.00647417))),
        ((-0.02074748, 0.20895012, -0.01565138), ((0.18759322, -0.01144737, -0.98218012), (0.98217165, -0.01018527, 0.18771034), (-0.01215252, -0.99988264, 0.00933257))),
        ((-0.07365019, 0.13859308, -0.01718625), ((-0.34981984, -0.00092953, -0.93681651), (0.93678069, -0.00914782, -0.34979743), (-0.00824469, -0.99995774, 0.00407088))),
    ],
    LEGS[0]: [
        ((0.12411077, 0.172485, -0.01282854), ((-0.99050552, 0.13746983, -0.00065276), (0.13745576, 0.9904536, 0.01037287), (0.00207249, 0.01018466, -0.99994576))),
        ((0.11414307, 0.16866885, -0.01511348), ((0.99296767, -0.11772603, 0.01248006), (0.11803517, 0.99261737, -0.02790101), (-0.00910325, 0.02917789, 0.99953276))),
        ((0.07065351, 0.16127343, -0.0162864), ((0.00333085, -0.52690697, -0.8499164), (-0.01105705, 0.84984976, -0.52690899), (0.9999333, 0.01115263, -0.00299532))),
    ],
    LEGS[1]: [
        ((0.14908047, 0.14851391, -0.02330433), ((0.98727685, 0.15897338, 0.0034444), (-0.15894894, 0.98726553, -0.00647893), (-0.00443052, 0.00584901, 0.99997306))),
        ((-0.04624121, 0.20801648, -0.01895902), ((0.9999948, 0.0032234, 0.00009178), (-0.00322341, 0.9999948, 0.00006646), (-0.00009156, -0.00006676, 1.0))),
        ((-0.07811823, 0.31678003, -0.01311489), ((-0.00119546, -0.40191379, -0.91567671), (-0.00777176, 0.91565347, -0.40189344), (0.99996912, 0.00663596, -0.0042182))),
    ],
    LEGS[2]: [
        ((-0.11946367, 0.18800808, -0.01739212), ((-0.01415714, -0.15679362, 0.98752993), (0.02789432, 0.98718256, 0.15713838), (-0.99951059, 0.0297711, -0.00960203))),
        ((0.21110627, 0.09755837, -0.01618727), ((0.00594478, -0.18838574, -0.98207718), (-0.01028048, 0.98203105, -0.18843913), (0.99992949, 0.01121645, 0.00390126))),
        ((0.1526014, 0.10060443, -0.0157905), ((-0.98098826, -0.19385707, -0.00902952), (-0.19405723, 0.98034805, 0.03548874), (0.00197233, 0.03656628, -0.99932933))),
    ],
    LEGS[3]: [
        ((-0.15307285, 0.12734982, -0.01829645), ((-0.79311496, -0.6090616, -0.0035405), (-0.60903323, 0.79298568, 0.01588211), (-0.00686562, 0.01475262, -0.99986762))),
        ((-0.14806452, 0.25308317, -0.01665781), ((-0.88222593, -0.47080287, -0.00470781), (-0.470696, 0.88170207, 0.03235343), (-0.0110812, 0.03075899, -0.99946541))),
        ((0.08496851, 0.28050846, -0.01651655), ((0.01009589, -0.99462211, -0.10307747), (-0.01416063, 0.10293016, -0.99458778), (0.99984878, 0.01150092, -0.0130454))),
    ],
}


def tag_pose_to_placement(part: str, pos_tag, rot_tag) -> Placement:
    """Convert a FurnitureBench base-tag-frame part pose into a W-frame Placement."""
    p = np.asarray(pos_tag, dtype=float)
    R = orthonormalize(_M @ np.asarray(rot_tag, dtype=float) @ _C)
    return Placement(part, (0.0015 + p[1], p[0], -p[2]), tuple(mat_to_quat(R)))


HIGH_PRESETS = tuple(
    tuple(tag_pose_to_placement(part, *_HIGH_TAG_DATA[part][i]) for part in (TOP,) + LEGS)
    for i in range(3)
)

# Reset rectangles of the design (they sit 2 cm closer to the parts than the built fence).
RESET_OBSTACLES = (
    BoxObstacle("reset_front", (0.3615, 0.0, 0.0), (0.01, 0.175, 0.015)),
    BoxObstacle("reset_left", (0.3065, 0.175, 0.0), (0.085, 0.01, 0.015)),
    BoxObstacle("reset_right", (0.3065, -0.175, 0.0), (0.085, 0.01, 0.015)),
)

RANDOMIZATION = RandomizationSpec(
    levels={"low": RandomLevel(0.015, 15.0), "med": RandomLevel(0.05, 45.0)},
    bounds_x=(0.0715, 0.3715),
    bounds_y=(-0.21, 0.21),
    footprints={TOP: (0.1625, 0.1625), **{leg: (0.0875, 0.05) for leg in LEGS}},
    clearance=0.01,
    reset_obstacles=RESET_OBSTACLES,
    high_presets=HIGH_PRESETS,
)

FENCE_FRONT = BoxObstacle("fence_front", (0.3815, 0.0, 0.015), (0.01, 0.175, 0.015))
FENCE_LEFT = BoxObstacle("fence_left", (0.3065, 0.175, 0.015), (0.085, 0.01, 0.015))
FENCE_RIGHT = BoxObstacle("fence_right", (0.3065, -0.175, 0.015), (0.085, 0.01, 0.015))
ROBOT_BASE = (-0.3, 0.0, 0.43)
OPERATOR_CAMERA = "operator"
_REAR_POS = (ROBOT_BASE[0] + 0.08, 0.0, ROBOT_BASE[2] + 0.2)
SCENE = SceneSpec(
    robot_base=ROBOT_BASE,
    obstacles=(FENCE_FRONT, FENCE_LEFT, FENCE_RIGHT),
    cameras=(
        CameraSpec("front", (0.90, 0.0, 0.65), (-1.0, 0.0, 0.30), fovy=40.0),
        CameraSpec("rear", _REAR_POS,
                   (_REAR_POS[0] + math.cos(math.radians(35)), 0.0, _REAR_POS[2] - math.sin(math.radians(35))),
                   fovy=40.0),
        # At the VR operator's mean head pose over the one_leg demos (robot's left, looking down across the table).
        CameraSpec("side", (0.27, 0.59, 1.04), (0.25, -0.05, 0.45), fovy=60.0),
        # Above the table's far corner on the robot's left, looking down across the workspace at the arm.
        CameraSpec("corner", (0.70, 0.45, 1.15), (0.05, 0.0, 0.55), fovy=60.0),
        # The VR operator's view (--record-operator): the collector moves it to the headset pose each
        # step. This default is the page's starting viewpoint, at the robot's left-rear.
        CameraSpec(OPERATOR_CAMERA, (-0.62, 0.475, 1.015), (0.2, 0.0, 0.415), fovy=90.0),
    ),
)

# Phase-2 target: the tabletop pushed snug into the +y corner of the built fence.
CORNER_XY = (FENCE_FRONT.center[0] - FENCE_FRONT.half_size[0] - TOP_SPEC.size[0] / 2,
             FENCE_LEFT.center[1] - FENCE_LEFT.half_size[1] - TOP_SPEC.size[1] / 2)
_CORNER = {"corner_xy": CORNER_XY, "corner_tol": 0.02, "corner_yaw_tol_deg": 15.0}

PHASES = ("grasp_top", "place_top_corner", "pick_leg", "insert_leg", "screw_leg")

_COMMON = dict(
    parts=(TOP_SPEC,) + LEG_SPECS,
    screws=(SCREW,),
    layout=LAYOUT,
    randomization=RANDOMIZATION,
    scene=SCENE,
)

ONE_LEG = TaskSpec(
    name="one_leg",
    targets=(_corner_targets(LEGS[3]),),
    phases=PHASES,
    extras={"target_leg": LEGS[3], **_CORNER},
    **_COMMON,
)

_SQUARE_ORDER = (LEGS[3], LEGS[2], LEGS[0], LEGS[1])
SQUARE_TABLE = TaskSpec(
    name="square_table",
    targets=tuple(_corner_targets(leg) for leg in _SQUARE_ORDER),
    phases=("grasp_top", "place_top_corner") + tuple(
        f"{step}_{leg[-4:]}" for leg in _SQUARE_ORDER for step in ("pick", "insert", "screw")),
    max_steps=4500,
    extras=dict(_CORNER),
    **_COMMON,
)
