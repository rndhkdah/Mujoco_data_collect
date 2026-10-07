"""lamp task spec: a bulb screwed into a base, then a hood set over the bulb onto the base.

Design source: FurnitureBench (Heo et al., RSS 2023; MIT License). Masses, layout, randomization,
measured layouts, assembled poses and tolerances follow its lamp design; dimensions were measured from
its part geometry and rebuilt from primitives in this project's frames (see fsim/spec/types.py).

  base  84.7 mm square foot, stepped loft, neck and a 55.2 mm square top plate (the grasp) around a
        threaded bore, 63 mm tall
  bulb  threaded stud, collar, 35 mm body and a Ø60 globe; lying, it rests on globe and collar (8.3 deg)
  hood  open conical shell (Ø88 -> Ø60 over 100 mm, 4 mm wall) as 24 tilted staves; it rests with its
        inside on the corners of the base's top plate (position-only target)
Threads: the FurnitureBench family (9.375 mm pitch, right-handed), as square_table.
"""
from __future__ import annotations

import math

import numpy as np

from fsim.geometry import make_T, mat_to_quat, rot_y, rot_z
from fsim.spec.common import _C, LEVELS, RESET_OBSTACLES, SCENE, corner_extras, ring_geoms, seated_rel_pose, tag_pose_to_placement
from fsim.spec.types import (
    IDENTITY_QUAT, AssemblyTarget, Geom, Phase, Placement, PrimitivePartSpec, RandomizationSpec, ScrewSpec, Site,
    Socket, Stud, TaskSpec, ThreadSpec,
)

BASE, BULB, HOOD = "lamp_base", "lamp_bulb", "lamp_hood"

PITCH = 0.009375
MALE = ThreadSpec(PITCH, 0.0072, 0.0128)
FEMALE = ThreadSpec(PITCH, 0.0109, 0.0147)
DOWN = (0.0, 1.0, 0.0, 0.0)
FLIP_Z = (0.0, 0.0, 0.0, 1.0)                    # yaw 180 degrees (the design's seated bulb)

# --- Base (origin at its bbox centre, 63 mm tall) ---------------------------------------------
BASE_HALF_H = 0.0315
FOOT_HALF = 0.04235
PLATE_HALF, PLATE_HALF_T = 0.0276, 0.0025
BORE_R, NECK_R = 0.0125, 0.0163
BORE_DEPTH = BASE_HALF_H                         # floor at the base origin
_PLATE_Z = BASE_HALF_H - PLATE_HALF_T
_FRAME = (PLATE_HALF - BORE_R) / 2
BASE_SPEC = PrimitivePartSpec(
    name=BASE, mass=0.05965,
    geoms=(
        Geom("box", (FOOT_HALF, FOOT_HALF, 0.0051), (0.0, 0.0, -BASE_HALF_H + 0.0051)),
        Geom("box", (0.0384, 0.0384, 0.00355), (0.0, 0.0, -0.01775)),          # square-to-round loft
        Geom("box", (0.0304, 0.0304, 0.00355), (0.0, 0.0, -0.01065)),
        Geom("cylinder", (0.0224, 0.00355), (0.0, 0.0, -0.00355)),
    )
    + ring_geoms((0, 0, 0), IDENTITY_QUAT, BORE_R, NECK_R, 0.0, 0.0165)        # neck
    + ring_geoms((0, 0, 0), IDENTITY_QUAT, BORE_R, 0.0210, 0.0165, _PLATE_Z - PLATE_HALF_T)   # flare
    + tuple(Geom("box", (_FRAME, PLATE_HALF, PLATE_HALF_T), (s * (BORE_R + _FRAME), 0.0, _PLATE_Z)) for s in (1, -1))
    + tuple(Geom("box", (BORE_R, _FRAME, PLATE_HALF_T), (0.0, s * (BORE_R + _FRAME), _PLATE_Z)) for s in (1, -1))
    + ring_geoms((0, 0, 0), IDENTITY_QUAT, BORE_R, NECK_R, _PLATE_Z - PLATE_HALF_T, BASE_HALF_H),
    sockets=(Socket((0.0, 0.0, BASE_HALF_H), FLIP_Z, BORE_DEPTH, BORE_R, thread=FEMALE, name="hole"),),
    rgba=(0.55, 0.60, 0.68, 1.0),
)

# --- Bulb (origin at the bbox centre of its geoms; stud below) -----------------------------------
GLOBE_R = 0.0295                 # 30 mm in the design; 0.5 mm smaller keeps the hood off the globe
COLLAR_R = 0.0216                # 22.2 mm in the design; with the smaller globe it keeps the 8.3 deg rest
BULB_SPEC = PrimitivePartSpec(
    name=BULB, mass=0.03847,
    geoms=(
        Geom("cylinder", (0.0160, 0.0020), (0.0, 0.0, -0.0430)),               # neck above the shoulder
        Geom("cylinder", (COLLAR_R, 0.0025), (0.0, 0.0, -0.0385)),
        Geom("cylinder", (0.0175, 0.0130), (0.0, 0.0, -0.0230)),               # body (35 mm grasp)
        Geom("cylinder", (COLLAR_R, 0.0030), (0.0, 0.0, -0.0070)),
        Geom("sphere", (GLOBE_R,), (0.0, 0.0, 0.045 - GLOBE_R)),
    ),
    studs=(Stud((0.0, 0.0, -0.045), DOWN, 0.0115, 0.0305, MALE, "stud"),),
    rgba=(0.93, 0.92, 0.85, 1.0),
)

# --- Hood (origin at its bbox centre; wide end down) -------------------------------------------
HOOD_HALF_H = 0.050
HOOD_R_IN = (0.040, 0.026)                       # inner radius at the bottom / top
HOOD_WALL = 0.004
HOOD_STAVES = 24
_SLOPE = (HOOD_R_IN[0] - HOOD_R_IN[1]) / (2 * HOOD_HALF_H)
_THETA = math.atan(_SLOPE)
_R_MID = (HOOD_R_IN[0] + HOOD_R_IN[1]) / 2 + HOOD_WALL / 2
_STAVE = (HOOD_WALL * math.cos(_THETA) / 2,
          (HOOD_R_IN[0] + HOOD_WALL) * math.tan(math.pi / HOOD_STAVES),       # no gaps at the wide bottom
          HOOD_HALF_H / math.cos(_THETA))


def _stave(k: int) -> Geom:
    a = 2 * math.pi * k / HOOD_STAVES
    R = rot_z(a) @ rot_y(-_THETA)
    return Geom("box", _STAVE, (_R_MID * math.cos(a), _R_MID * math.sin(a), 0.0), tuple(mat_to_quat(R)))


HOOD_SPEC = PrimitivePartSpec(name=HOOD, mass=0.05999, geoms=tuple(_stave(k) for k in range(HOOD_STAVES)),
                              rgba=(0.92, 0.80, 0.45, 1.0))

# --- Mechanism and targets --------------------------------------------------------------------
SCREW = ScrewSpec(parent=BASE, children=(BULB,), pitch=PITCH,
                  sites=(Site((0.0, 0.0, BASE_HALF_H), FLIP_Z, BORE_DEPTH),), child_feature="stud")


def hood_rest_height(face_toward_corner: bool) -> float:
    """Hood origin height above the base origin when its inside rests on the top-plate corners."""
    corner = PLATE_HALF * math.sqrt(2)
    # inner face apothem at hood height z: HOOD_R_IN[0] - _SLOPE * (z + HOOD_HALF_H); at a stave edge
    # the inside is 1/cos(pi/24) farther out.
    need = corner if face_toward_corner else corner * math.cos(math.pi / HOOD_STAVES)
    z_contact = (HOOD_R_IN[0] - need) / _SLOPE - HOOD_HALF_H        # hood frame height of the plate top
    return BASE_HALF_H - z_contact


# FurnitureBench's assembled hood sits 75.4 mm above the base origin; ours rests 74.6 (stave face toward
# a plate corner) to 72.2 mm (stave edge toward it) depending on its yaw, so the target is centred there.
HOOD_REL_Z = round((hood_rest_height(True) + hood_rest_height(False)) / 2, 4)
TARGETS = (
    AssemblyTarget(BASE, BULB, (seated_rel_pose(SCREW, 0, BULB_SPEC),)),
    AssemblyTarget(BASE, HOOD, (((0.0, 0.0, HOOD_REL_Z), IDENTITY_QUAT),), position_only=True,
                   requires=((BASE, BULB),)),
)

# --- Layout (FurnitureBench reset pose, skill 0) ---------------------------------------------------
BULB_TILT = math.atan2(7.6, 52.0)                # lying on globe + collar, globe end up
LAYOUT = (
    Placement(BASE, (0.1415, 0.05, BASE_HALF_H), tuple(mat_to_quat(rot_z(math.pi / 2)))),
    Placement(BULB, (0.1543, 0.18, 0.0279), tuple(mat_to_quat(rot_y(math.pi / 2 - BULB_TILT) @ rot_z(math.pi / 2)))),
    Placement(HOOD, (0.1615, -0.12, HOOD_HALF_H + 0.0003), tuple(mat_to_quat(rot_z(math.pi / 2)))),
)

# --- High-randomness presets: measured layouts in the design's AprilTag base-tag frame. Base and hood
# frames: our z = their -y; bulb: our z = their y, origin 23 mm along it.
_CB = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_FB_T = {BASE: make_T((0.0, -0.0143, 0.0), _CB), BULB: make_T((0.0, 0.023, 0.0), _C),
         HOOD: make_T((0.0, -0.00142, 0.0), _CB)}
_HIGH_TAG_DATA = {
    BASE: [
        ((-0.129, 0.205, -0.02), ((0.911, 0.005, -0.413), (-0.413, -0.028, -0.91), (-0.016, 1.0, -0.023))),
        ((-0.03, 0.26, -0.019), ((0.824, 0.005, 0.567), (0.567, -0.01, -0.824), (0.001, 1.0, -0.011))),
        ((-0.072, 0.248, -0.019), ((0.737, 0.004, 0.675), (0.675, -0.01, -0.737), (0.004, 1.0, -0.01))),
    ],
    BULB: [
        ((0.07429203, 0.29051307, -0.0336427), ((-0.0242578, -0.98711663, -0.15815291), (0.93183929, -0.07962076, 0.35402828), (-0.36205944, -0.13878517, 0.92176551))),
        ((0.077, 0.156, -0.023), ((0.967, -0.143, 0.211), (0.116, 0.984, 0.135), (-0.227, -0.106, 0.968))),
        ((0.03684583, 0.15773909, -0.0237988), ((0.60462284, -0.72320396, -0.33377719), (0.7204029, 0.67527962, -0.15816779), (0.33978051, -0.14482222, 0.92928773))),
    ],
    HOOD: [
        ((0.084, 0.141, -0.053), ((-0.995, 0.014, -0.095), (-0.095, -0.01, 0.995), (0.013, 1.0, 0.011))),
        ((-0.088, 0.12, -0.05), ((-0.971, 0.046, 0.235), (0.238, 0.063, 0.969), (0.03, 0.997, -0.072))),
        ((0.081, 0.284, -0.052), ((-0.839, 0.005, 0.544), (0.544, -0.015, 0.839), (0.012, 1.0, 0.01))),
    ],
}
HIGH_PRESETS = tuple(
    tuple(tag_pose_to_placement(p, *_HIGH_TAG_DATA[p][i], fb_T_part=_FB_T[p]) for p in (BASE, BULB, HOOD))
    for i in range(3)
)

RANDOMIZATION = RandomizationSpec(
    levels=LEVELS,
    bounds_x=(0.0715, 0.3715),
    bounds_y=(-0.21, 0.21),
    footprints={BASE: (0.0847, 0.0847), BULB: (0.13, 0.057), HOOD: (0.088, 0.088)},
    clearance=0.01,
    reset_obstacles=RESET_OBSTACLES,
    high_presets=HIGH_PRESETS,
)

PHASES = (
    Phase("grasp", BASE, label="grasp_base"), Phase("place_corner", BASE, label="place_base_corner"),
    Phase("pick", BULB, label="pick_bulb"), Phase("insert", BULB, BASE, "insert_bulb"),
    Phase("screw", BULB, BASE, "screw_bulb"),
    Phase("pick", HOOD, label="pick_hood"), Phase("place", HOOD, BASE, "place_hood"),
)

LAMP = TaskSpec(
    name="lamp",
    parts=(BASE_SPEC, BULB_SPEC, HOOD_SPEC),
    screws=(SCREW,),
    targets=TARGETS,
    layout=LAYOUT,
    randomization=RANDOMIZATION,
    scene=SCENE,
    phases=PHASES,
    max_steps=3500,
    extras=corner_extras(2 * FOOT_HALF, 2 * FOOT_HALF),
)
