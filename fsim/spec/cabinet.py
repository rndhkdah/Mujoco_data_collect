"""cabinet task spec: a body with two hinge poles and a threaded stud, two doors that slide onto the
poles, and a top that screws onto the stud and caps the poles.

Design source: FurnitureBench (Heo et al., RSS 2023; MIT License) cabinet. Dimensions were measured from
its meshes and re-expressed as parametric primitives in this project's frames (see fsim/spec/types.py).

Frames (origin = bounding-box centre of the collision geoms, studs excluded; z = pole / screw axis,
pointing to the body's top end): body front (doors) at -x, back wall at +x; the right door's pole at
-y, the left door's at +y. Doors and top use the same axes when assembled (identity relative rotation):
door x = panel normal (handle on -x), top bore opening on -z.

Mechanisms:
  * doors: one InsertSpec per pole. The door's C-clip (two jaws, 9.5 mm slot) is captured 6 mm
    before its leading end reaches the pole tip, then slides 123.75 mm down the pole and is locked
    (no hinge: the design scores a fixed pose, and a free door would swing open when the body is
    tipped upright).
  * top: an inverted screw (the body owns the stud; 9.375 mm right-handed thread, 1.5 turns). A
    second site rotated 180 degrees about the axis accepts the top's other placement.
The design's left-door pose is 4.75 mm off its pole (a data slip); we use the mirror of the right
door. The design also lets the top be counted after only the left door; we require both doors.
"""
from __future__ import annotations

import math

import numpy as np

from fsim.geometry import make_T, mat_to_quat, rot_z
from fsim.spec.common import LEVELS, RESET_OBSTACLES, SCENE, ring_geoms, seated_rel_pose, tag_pose_to_placement
from fsim.spec.types import (
    IDENTITY_QUAT, AssemblyTarget, Geom, InsertSpec, Phase, PrimitivePartSpec, RandomizationSpec, ScrewSpec, Site,
    Socket, Stud, TaskSpec, ThreadSpec,
)

BODY = "cabinet_body"
DOOR_L = "cabinet_door_left"
DOOR_R = "cabinet_door_right"
TOP = "cabinet_top"

PITCH = 0.009375                       # right-handed, as the square_table legs
TOP_FACE_Z = 0.0675                    # body top-plate face = pole tips = stud shoulder
POLE_X, POLE_Y, POLE_R = -0.021, 0.0494, 0.0031
STUD_POS = (0.0005, 0.0, TOP_FACE_Z)
STUD_R = 0.011                         # collision radius (thread major ~12.2 mm, visual)
BORE_R = 0.012                         # top bore (collision), 1 mm around the stud
CATCH_R = 0.014                        # catch collar: under the top's bore wall, 1.25 mm clear of the doors

# --- body (60.5 x 120 x 135 mm + 25 mm stud, 130.98 g) ------------------------------------------
BODY_SPEC = PrimitivePartSpec(
    name=BODY, mass=0.13098,
    geoms=(
        Geom("box", (0.001875, 0.06, 0.0675), (0.028375, 0.0, 0.0)),               # back wall
        Geom("box", (0.026875, 0.001875, 0.0625), (-0.000375, 0.058125, 0.005)),   # side walls
        Geom("box", (0.026875, 0.001875, 0.0625), (-0.000375, -0.058125, 0.005)),
        Geom("box", (0.028375, 0.06, 0.005), (-0.001875, 0.0, -0.0625)),           # base plate (3 mm front lip)
        Geom("box", (0.02, 0.05625, 0.001875), (0.0065, 0.0, 0.065625)),           # top plate (front edge cut back)
        Geom("box", (0.019375, 0.05625, 0.00188), (0.007125, 0.0, 0.005)),         # shelf
        Geom("cylinder", (POLE_R, 0.0625), (POLE_X, -POLE_Y, 0.005)),              # hinge poles
        Geom("cylinder", (POLE_R, 0.0625), (POLE_X, POLE_Y, 0.005)),
    ),
    studs=(Stud(STUD_POS, IDENTITY_QUAT, STUD_R, 0.025, ThreadSpec(PITCH, 0.0076, 0.0122), "stud"),),
    rgba=(0.80, 0.68, 0.52, 1.0),
)

# --- doors (30 x 53.75 x 123.75 mm, 30.2 g) ---------------------------------------------------------
CLIP = (0.00875, -0.021875, 0.061875)  # clip centre at the door's top end (right door)


def _door(name: str, s: float) -> PrimitivePartSpec:
    """s = +1: right door (clip on -y); s = -1: left door (mirror)."""
    return PrimitivePartSpec(
        name=name, mass=0.0302,
        geoms=(
            Geom("box", (0.00625, 0.021675, 0.061875), (0.00875, s * 0.0052, 0.0)),     # panel
            Geom("box", (0.00075, 0.0052, 0.061875), (0.00325, -s * 0.021675, 0.0)),    # clip jaws (1.5 mm)
            Geom("box", (0.00075, 0.0052, 0.061875), (0.01425, -s * 0.021675, 0.0)),
            Geom("box", (0.002825, 0.0138, 0.00375), (-0.012175, s * 0.003025, 0.0)),   # pull handle bar
            Geom("box", (0.005925, 0.00255, 0.00375), (-0.003425, -s * 0.008225, 0.0)), # handle posts
            Geom("box", (0.005925, 0.00255, 0.00375), (-0.003425, s * 0.014275, 0.0)),
        ),
        rgba=(0.86, 0.76, 0.60, 1.0),
    )


DOOR_R_SPEC = _door(DOOR_R, 1.0)
DOOR_L_SPEC = _door(DOOR_L, -1.0)


def _door_insert(door: str, s: float) -> InsertSpec:
    return InsertSpec(
        parent=BODY, children=(door,),
        sites=(Site((POLE_X, -s * POLE_Y, TOP_FACE_Z)),),
        travel=0.13,                     # captured 6 mm before the clip's leading end reaches the pole tip
        child_mate=((CLIP[0], s * CLIP[1], CLIP[2]), IDENTITY_QUAT),
        engage_radial_tol=0.003, engage_tilt_deg=4.0, engage_twist_deg=10.0,   # tilt: 124 mm door
    )


INSERT_R = _door_insert(DOOR_R, 1.0)
INSERT_L = _door_insert(DOOR_L, -1.0)

# --- top (57.5 x 120 x 30 mm, 60.29 g) --------------------------------------------------------------
_HOLE = 0.0165                         # half-width of the square opening the frame boxes leave around the bore ring
TOP_SPEC = PrimitivePartSpec(
    name=TOP, mass=0.06029,
    geoms=(
        Geom("box", (0.02875, 0.06, 0.002), (0.0, 0.0, 0.013)),                    # cap over the blind bore
        Geom("box", (0.02875, 0.02175, 0.013), (0.0, 0.03825, -0.002)),
        Geom("box", (0.02875, 0.02175, 0.013), (0.0, -0.03825, -0.002)),
        Geom("box", (0.006125, _HOLE, 0.013), (0.022625, 0.0, -0.002)),
        Geom("box", (0.006125, _HOLE, 0.013), (-0.022625, 0.0, -0.002)),
    ) + ring_geoms((0.0, 0.0, 0.0), IDENTITY_QUAT, BORE_R, 0.017, -0.015, 0.011),
    sockets=(Socket((0.0, 0.0, -0.015), (0.0, 1.0, 0.0, 0.0), 0.026, BORE_R,
                    thread=ThreadSpec(PITCH, 0.0100, 0.0130), name="bore"),),
    rgba=(0.74, 0.62, 0.46, 1.0),
)

SCREW = ScrewSpec(
    parent=BODY, children=(TOP,), pitch=PITCH,
    sites=(Site(STUD_POS), Site(STUD_POS, tuple(float(v) for v in mat_to_quat(rot_z(math.pi))))),
    catch_radius=CATCH_R,
    # The top is a 120 mm bar (20x the leg's spin inertia): more thread friction keeps it from springing
    # back off the seated stop after a fast twist (EST, the leg uses 0.01 N*m).
    thread_friction=0.03,
)

_BOTH_DOORS = ((BODY, DOOR_R), (BODY, DOOR_L))
TARGETS = (
    AssemblyTarget(BODY, DOOR_R, (seated_rel_pose(INSERT_R, 0, DOOR_R_SPEC),)),
    AssemblyTarget(BODY, DOOR_L, (seated_rel_pose(INSERT_L, 0, DOOR_L_SPEC),)),
    AssemblyTarget(BODY, TOP, tuple(seated_rel_pose(SCREW, i, TOP_SPEC) for i in range(2)), requires=_BOTH_DOORS),
)

# --- layout ---------------------------------------------------------------------------------------
# Our part frame in the design's part frame: axis map (columns = our x, y, z) and bbox-centre origin.
_C_BODY = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_C_DOOR = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
_C_TOP = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
_FB_T = {BODY: make_T((-0.0015, 0.005, 0.0), _C_BODY), DOOR_L: make_T((-0.003025, 0.036875, -0.00375), _C_DOOR),
         DOOR_R: make_T((0.003025, 0.036875, -0.00375), _C_DOOR), TOP: make_T((0.0, 0.0, 0.001), _C_TOP)}
_TAG_RESET = {   # the design's skill-0 reset poses (AprilTag base-tag frame)
    BODY: ((0.0, 0.24, -0.02875), ((0, 0, -1), (0, 1, 0), (1, 0, 0))),
    DOOR_L: ((0.12, 0.08, -0.01125), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
    DOOR_R: ((0.20, 0.08, -0.01125), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
    TOP: ((-0.15, 0.15, -0.015), ((0, 0, 1), (1, 0, 0), (0, 1, 0))),
}
_HIGH_TAG_DATA = {   # the design's three measured high-randomness layouts
    BODY: [
        ((0.105, 0.128, -0.034), ((0.016, 0.791, 0.612), (-0.011, -0.611, 0.791), (1.0, -0.019, -0.001))),
        ((0.076, 0.273, -0.033), ((0.018, 0.953, -0.301), (-0.004, 0.301, 0.954), (1.0, -0.016, 0.009))),
        ((-0.08, 0.265, -0.03), ((0.013, -0.156, -0.988), (0.002, 0.988, -0.156), (1.0, -0.0, 0.013))),
    ],
    DOOR_L: [
        ((0.045, 0.221, -0.01), ((0.79, 0.613, 0.009), (-0.613, 0.79, -0.009), (-0.013, 0.001, 1.0))),
        ((-0.056, 0.315, -0.003), ((-0.823, -0.569, 0.007), (0.568, -0.823, -0.018), (0.016, -0.011, 1.0))),
        ((0.076, 0.317, -0.011), ((-0.453, -0.892, 0.006), (0.891, -0.453, -0.012), (0.014, -0.0, 1.0))),
    ],
    DOOR_R: [
        ((-0.077, 0.139, -0.011), ((0.564, -0.826, 0.001), (0.826, 0.563, -0.028), (0.023, 0.017, 1.0))),
        ((0.077, 0.081, -0.012), ((0.861, -0.508, -0.003), (0.508, 0.861, -0.027), (0.016, 0.022, 1.0))),
        ((0.133, 0.199, -0.013), ((0.962, -0.271, 0.019), (0.272, 0.962, -0.013), (-0.015, 0.018, 1.0))),
    ],
    TOP: [
        ((-0.068, 0.303, -0.015), ((-0.993, 0.003, 0.117), (0.117, -0.02, 0.993), (0.006, 1.0, 0.02))),
        ((-0.061, 0.169, -0.016), ((-0.854, 0.004, -0.521), (-0.521, -0.008, 0.854), (-0.001, 1.0, 0.009))),
        ((0.065, 0.135, -0.017), ((-0.962, 0.005, 0.273), (0.273, -0.002, 0.962), (0.005, 1.0, 0.001))),
    ],
}
_ORDER = (BODY, DOOR_L, DOOR_R, TOP)
LAYOUT = tuple(tag_pose_to_placement(p, *_TAG_RESET[p], fb_T_part=_FB_T[p]) for p in _ORDER)
HIGH_PRESETS = tuple(tuple(tag_pose_to_placement(p, *_HIGH_TAG_DATA[p][i], fb_T_part=_FB_T[p]) for p in _ORDER)
                     for i in range(3))

RANDOMIZATION = RandomizationSpec(
    levels=LEVELS,
    bounds_x=(0.0715, 0.3715),
    bounds_y=(-0.21, 0.21),
    # Real extents at the layout yaw (W x, W y), centred on the part origin (the body's stud makes it 185 mm).
    footprints={BODY: (0.185, 0.12), DOOR_L: (0.12375, 0.05375), DOOR_R: (0.12375, 0.05375), TOP: (0.12, 0.0575)},
    reset_obstacles=RESET_OBSTACLES,
    high_presets=HIGH_PRESETS,
)

# --- phases ---------------------------------------------------------------------------------------
# The body, lying on its back with the stud toward the robot, goes snug into the +y fence corner (the
# design's skill-2 pose): the front fence then backs it while the doors are pushed on along +W x. After
# both doors it is tipped upright (top end up) and the top is screwed on.
BODY_CORNER = (0.3715 - 0.0675, 0.165 - 0.06, 0.03025)


def _door_phases(door: str, suffix: str) -> tuple[Phase, ...]:
    return (Phase("pick", door, label=f"pick_{suffix}"), Phase("insert", door, BODY, f"insert_{suffix}"),
            Phase("slide", door, BODY, f"slide_{suffix}"))


PHASES = (
    Phase("grasp", BODY, label="grasp_body"),
    Phase("place_pose", BODY, label="place_body_corner", pos=BODY_CORNER, pos_tol=0.02,
          axis=(0.0, 0.0, 1.0), axis_dir=(-1.0, 0.0, 0.0)),
) + _door_phases(DOOR_R, "door_right") + _door_phases(DOOR_L, "door_left") + (
    Phase("grasp", BODY, label="grasp_body_flip"),
    Phase("place_pose", BODY, label="tip_body_upright", axis=(0.0, 0.0, 1.0), axis_dir=(0.0, 0.0, 1.0)),
    Phase("pick", TOP, label="pick_top"),
    Phase("insert", TOP, BODY, "insert_top"),
    Phase("screw", TOP, BODY, "screw_top"),
)

CABINET = TaskSpec(
    name="cabinet",
    parts=(BODY_SPEC, DOOR_L_SPEC, DOOR_R_SPEC, TOP_SPEC),
    screws=(SCREW,),
    inserts=(INSERT_R, INSERT_L),
    targets=TARGETS,
    layout=LAYOUT,
    randomization=RANDOMIZATION,
    scene=SCENE,
    phases=PHASES,
    max_steps=6000,
    extras={"body_corner": BODY_CORNER},
)
