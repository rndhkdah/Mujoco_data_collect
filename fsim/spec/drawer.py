"""drawer task spec: a chest (box) and two drawers (containers) that slide into its two slots.

Design source: FurnitureBench (Heo et al., RSS 2023; MIT License) drawer. Dimensions were measured from
its meshes and re-expressed as parametric primitives in this project's frames (see fsim/spec/types.py).

Frames (part origin = bounding-box centre, z = slide axis): the box keeps the design's axes, so its
+z face is the drawer opening and its +y points DOWN when it stands upright (the chest "floor" is at
+y, an open tray sits on top at -y). The two containers share one geometry (the design's two meshes
differ only by their origin): +z is the front panel / knob, +y the floor. Seated, a container's axes equal
the box's (identity rotation).

Mechanism: one InsertSpec with a slot per drawer opening; either container fits either slot. The slot
captures a container 85 mm out (1 cm before its back reaches the opening), so it never has to slide in
by contact; the front panel stops on the side-wall front edges (the seat).
"""
from __future__ import annotations

import numpy as np

from fsim.geometry import make_T, mat_to_quat
from fsim.spec.common import LEVELS, RESET_OBSTACLES, SCENE, tag_pose_to_placement
from fsim.spec.types import (
    IDENTITY_QUAT, AssemblyTarget, Geom, InsertSpec, Phase, PrimitivePartSpec, RandomizationSpec, Site, TaskSpec,
)

BOX = "drawer_box"
TOP = "drawer_container_top"          # the design's upper drawer
BOTTOM = "drawer_container_bottom"    # the design's lower drawer (assembled first)

# --- box (112.5 x 121.5 x 97.5 mm, 151.63 g) ---------------------------------------------------------
FACE_Z = 0.04125                       # drawer opening plane (side-wall front edges)
SLOT_Y = {"upper": -0.010875, "lower": 0.034125}  # container centre y per slot (box frame, +y down)
RAIL_R = 0.003                         # rails r 3 mm (design: 3.75) for clearance (EST)


def _rails():
    out = []
    for y in (0.0045, 0.0495):         # one rail pair per slot, 15.375 mm below the container centre
        for x in (0.0525, -0.0525):
            out.append(Geom("capsule", (RAIL_R, 0.03825), (x, y, -0.00375)))
    return tuple(out)


BOX_SPEC = PrimitivePartSpec(
    name=BOX, mass=0.15163,
    geoms=(
        Geom("box", (0.001875, 0.06075, 0.045), (0.054375, 0.0, -0.00375)),        # side walls
        Geom("box", (0.001875, 0.06075, 0.045), (-0.054375, 0.0, -0.00375)),
        Geom("box", (0.001875, 0.013875, 0.00375), (0.054375, -0.046875, 0.045)),  # side walls beside the tray
        Geom("box", (0.001875, 0.013875, 0.00375), (-0.054375, -0.046875, 0.045)),
        Geom("box", (0.0525, 0.058875, 0.001875), (0.0, -0.001875, -0.046875)),    # back panel
        Geom("box", (0.05625, 0.001875, 0.04875), (0.0, 0.058875, 0.0)),           # end wall (chest floor, +y)
        Geom("box", (0.0525, 0.001875, 0.046875), (0.0, -0.034875, 0.001875)),     # divider (tray floor)
        Geom("box", (0.0525, 0.012, 0.002), (0.0, -0.04875, 0.04675)),             # tray front plate
    ) + _rails(),
    rgba=(0.80, 0.68, 0.52, 1.0),
)

# --- container (110.25 x 42.75 x 99.75 mm, 59.37 g) ------------------------------------------------
PANEL_BACK_Z = 0.025125                # back face of the front panel (the mate / stop face)


def _container(name: str, rgba) -> PrimitivePartSpec:
    return PrimitivePartSpec(
        name=name, mass=0.05937,
        geoms=(
            Geom("box", (0.00225, 0.016625, 0.0375), (0.04875, -0.00475, -0.012375)),    # side walls (above the groove)
            Geom("box", (0.00225, 0.016625, 0.0375), (-0.04875, -0.00475, -0.012375)),
            Geom("box", (0.0015, 0.002875, 0.0375), (0.046, 0.01475, -0.012375)),        # wall-to-floor fillets
            Geom("box", (0.0015, 0.002875, 0.0375), (-0.046, 0.01475, -0.012375)),
            Geom("box", (0.0475, 0.001875, 0.0375), (0.0, 0.0195, -0.012375)),           # floor (+y)
            Geom("box", (0.0465, 0.0195, 0.00225), (0.0, -0.001875, -0.047625)),         # back wall
            Geom("box", (0.05512, 0.021375, 0.00375), (0.0, 0.0, 0.028875)),             # front panel
            Geom("cylinder", (0.0055, 0.00325), (0.0, -0.00133, 0.035875)),              # knob neck
            Geom("cylinder", (0.0089, 0.005375), (0.0, -0.00133, 0.0445)),               # knob cap
        ),
        rgba=rgba,
    )


TOP_SPEC = _container(TOP, (0.86, 0.76, 0.60, 1.0))
BOTTOM_SPEC = _container(BOTTOM, (0.74, 0.62, 0.46, 1.0))

INSERT = InsertSpec(
    parent=BOX, children=(BOTTOM, TOP),
    sites=(Site((0.0, SLOT_Y["lower"], FACE_Z)), Site((0.0, SLOT_Y["upper"], FACE_Z))),
    travel=0.085,
    child_mate=((0.0, 0.0, PANEL_BACK_Z), IDENTITY_QUAT),
    seat_gap=0.0003,
    engage_radial_tol=0.004, engage_tilt_deg=8.0, engage_twist_deg=8.0,
)

_SEAT_Z = FACE_Z + INSERT.seat_gap - PANEL_BACK_Z
SLOT_POSES = tuple(((0.0, SLOT_Y[s], _SEAT_Z), IDENTITY_QUAT) for s in ("lower", "upper"))
TARGETS = tuple(AssemblyTarget(BOX, c, SLOT_POSES) for c in (BOTTOM, TOP))

# --- layout ---------------------------------------------------------------------------------------
# Our part frame in the design's part frame (axes unchanged, origin at the bbox centre).
_FB_T = {BOX: make_T((0.0, -0.0195, 0.0)), TOP: make_T((0.0, 0.004125, 0.008625)),
         BOTTOM: make_T((0.0, 0.004125, 0.004875))}
_TAG_RESET = {   # the design's skill-0 reset poses (AprilTag base-tag frame)
    BOX: ((0.08, 0.13, -0.04125), ((-1, 0, 0), (0, 0, 1), (0, 1, 0))),
    TOP: ((-0.11, 0.12, -0.0255), ((1, 0, 0), (0, 0, -1), (0, 1, 0))),
    BOTTOM: ((-0.11, 0.25, -0.0255), ((1, 0, 0), (0, 0, -1), (0, 1, 0))),
}
_HIGH_TAG_DATA = {   # the design's three measured high-randomness layouts
    BOX: [
        ((-0.078, 0.159, -0.042), ((-0.928, -0.001, 0.374), (0.374, -0.005, 0.927), (0.001, 1.0, 0.005))),
        ((-0.101, 0.168, -0.043), ((-0.633, 0.005, -0.774), (-0.774, -0.021, 0.632), (-0.013, 1.0, 0.017))),
        ((0.103, 0.118, -0.046), ((-0.991, 0.007, 0.131), (0.131, -0.015, 0.991), (0.009, 1.0, 0.014))),
    ],
    TOP: [
        ((-0.1, 0.293, -0.029), ((-1.0, 0.016, -0.016), (-0.016, 0.012, 1.0), (0.016, 1.0, -0.012))),
        ((0.069, 0.137, -0.027), ((0.871, -0.0, 0.492), (0.492, -0.025, -0.87), (0.012, 1.0, -0.021))),
        ((-0.038, 0.156, -0.028), ((-0.9, 0.006, -0.436), (-0.436, -0.023, 0.9), (-0.004, 1.0, 0.023))),
    ],
    BOTTOM: [
        ((0.086, 0.173, -0.029), ((0.221, -0.011, 0.975), (0.975, -0.002, -0.221), (0.005, 1.0, 0.01))),
        ((0.034, 0.261, -0.027), ((-0.938, 0.008, -0.346), (-0.346, -0.016, 0.938), (0.002, 1.0, 0.018))),
        ((-0.085, 0.28, -0.018), ((-0.9, 0.006, -0.436), (-0.436, -0.023, 0.9), (-0.004, 1.0, 0.023))),
    ],
}
_ORDER = (BOX, TOP, BOTTOM)
LAYOUT = tuple(tag_pose_to_placement(p, *_TAG_RESET[p], fb_T_part=_FB_T[p]) for p in _ORDER)
HIGH_PRESETS = tuple(tuple(tag_pose_to_placement(p, *_HIGH_TAG_DATA[p][i], fb_T_part=_FB_T[p]) for p in _ORDER)
                     for i in range(3))

RANDOMIZATION = RandomizationSpec(
    levels=LEVELS,
    bounds_x=(0.0715, 0.3715),
    bounds_y=(-0.21, 0.21),
    # Real extents at the layout yaw (W x, W y): box opening toward +x, containers knob toward the robot.
    footprints={BOX: (0.0975, 0.1125), TOP: (0.09975, 0.11025), BOTTOM: (0.09975, 0.11025)},
    reset_obstacles=RESET_OBSTACLES,
    high_presets=HIGH_PRESETS,
)

# --- phases ---------------------------------------------------------------------------------------
# The box goes snug into the +y fence corner with its opening facing -W y (the design's skill-2 pose);
# the left fence then backs it while the drawers are pushed in along +W y.
BOX_CORNER = (0.3715 - 0.05625, 0.165 - 0.04875, 0.06075)
_BOX_CORNER_R = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])   # columns: x -> -W x, y -> -W z, z -> -W y
BOX_CORNER_QUAT = tuple(float(v) for v in mat_to_quat(_BOX_CORNER_R))


def _drawer_phases(part: str, suffix: str) -> tuple[Phase, ...]:
    return (Phase("pick", part, label=f"pick_{suffix}"), Phase("insert", part, BOX, f"insert_{suffix}"),
            Phase("slide", part, BOX, f"push_{suffix}"))


PHASES = (
    Phase("grasp", BOX, label="grasp_box"),
    Phase("place_pose", BOX, label="place_box_corner", pos=BOX_CORNER, pos_tol=0.02,
          axis=(0.0, 0.0, 1.0), axis_dir=(0.0, -1.0, 0.0)),
    Phase("place_pose", BOX, label="box_upright", axis=(0.0, 1.0, 0.0), axis_dir=(0.0, 0.0, -1.0)),
) + _drawer_phases(BOTTOM, "container_bottom") + _drawer_phases(TOP, "container_top")

DRAWER = TaskSpec(
    name="drawer",
    parts=(BOX_SPEC, TOP_SPEC, BOTTOM_SPEC),
    screws=(),
    inserts=(INSERT,),
    targets=TARGETS,
    layout=LAYOUT,
    randomization=RANDOMIZATION,
    scene=SCENE,
    phases=PHASES,
    max_steps=4500,
    extras={"box_corner": BOX_CORNER, "box_corner_quat": BOX_CORNER_QUAT},
)
