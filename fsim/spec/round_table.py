"""round_table task spec.

Design source: FurnitureBench (Heo et al., RSS 2023; MIT License): its round_table (furniture/round_table.py,
config "round_table"). Dimensions measured on its meshes, masses from its sim_config, scene / layout /
randomization / presets / assembled poses re-expressed in this project's frames (see fsim/spec/types.py).

Parts (our frames, z = assembly axis, origin at the bounding-box centre of the collision geoms):
  top   Ø200 mm disc: 6.25 mm plate, 6.25 mm rim up to 18.75 mm, a Ø42.5 mm boss in the middle up to 32.5 mm
        with a threaded Ø25 mm socket, 26.25 mm deep (the table is built upside down, socket up).
  leg   65 mm body: a 4-flat collar (33.4 mm across the flats, corners on Ø42.5) then a Ø42.5 mm cylinder;
        a 26.25 mm threaded stud below and a threaded Ø25 mm socket, 27.5 mm deep, on top.
  base  "+"-shaped foot, 130 mm tip to tip, 12.5 mm thick, arms 30 mm wide at the ends narrowing to 22 mm
        at a Ø40 mm hub, with a 27.5 mm threaded stud below. It starts standing on its stud, as in the design.
The leg screws into the top, the base screws onto the leg (a chain). Threads: 9.375 mm pitch, right-handed.
"""
from __future__ import annotations

import math

from fsim.geometry import make_T, mat_to_quat, rot_x, rot_z
from fsim.spec.common import _C, LEVELS, RESET_OBSTACLES, SCENE, corner_extras, ring_geoms, seated_rel_pose, \
    tag_pose_to_placement
from fsim.spec.types import (
    IDENTITY_QUAT, AssemblyTarget, Geom, Phase, Placement, PrimitivePartSpec, RandomizationSpec, ScrewSpec, Site,
    Socket, Stud, TaskSpec, ThreadSpec, site_at,
)

TOP, LEG, BASE = "round_table_top", "round_table_leg", "round_table_base"

PITCH = 0.009375
DOWN = tuple(float(v) for v in mat_to_quat(rot_x(math.pi)))           # feature z along the part's -z
MALE = ThreadSpec(PITCH, minor_radius=0.0078, major_radius=0.0122)     # measured root / crest radii
FEMALE = ThreadSpec(PITCH, minor_radius=0.0109, major_radius=0.0147)   # as square_table's holes
BORE = 0.0125
PEG = 0.0115                                                           # stud collision radius, as square_table
THREAD_FRICTION = 0.01                                                 # N*m; see the spin test
# Seated yaws of the design's assembled poses (fixed by where each thread starts): the leg sits at
# -175 deg in the top (FurnitureBench: get_mat([0, 0, 0.044375], [pi/2, 0, pi + pi/36])), the base at
# +90 deg on the leg (get_mat([0, 0.053125, 0], [-pi/2, pi/2, 0])).
TOP_SITE_QUAT = tuple(float(v) for v in mat_to_quat(rot_z(math.radians(-175.0))))
LEG_SITE_QUAT = tuple(float(v) for v in mat_to_quat(rot_z(math.radians(90.0))))

TOP_SPEC = PrimitivePartSpec(
    name=TOP, mass=0.12169,
    geoms=(Geom("cylinder", (0.100, 0.003125), (0.0, 0.0, -0.013125)),)                  # plate, z -16.25..-10 mm
    + ring_geoms((0.0, 0.0, 0.0), IDENTITY_QUAT, 0.09375, 0.100, -0.010, 0.0025, segments=32)   # rim
    + ring_geoms((0.0, 0.0, 0.0), IDENTITY_QUAT, BORE, 0.02125, -0.010, 0.01625, segments=24),  # boss
    sockets=(Socket((0.0, 0.0, 0.01625), TOP_SITE_QUAT, 0.02625, BORE, thread=FEMALE, name="socket"),),
    rgba=(0.80, 0.68, 0.52, 1.0),
)

# Collar: two crossed boxes give the 4 flats (33.42 mm across) with corners on the Ø42.5 cylinder.
_COLLAR_Z, _COLLAR_H = -0.019375, 0.013125
LEG_SPEC = PrimitivePartSpec(
    name=LEG, mass=0.03228,
    geoms=(Geom("box", (0.01671, 0.01313, _COLLAR_H), (0.0, 0.0, _COLLAR_Z)),
           Geom("box", (0.01313, 0.01671, _COLLAR_H), (0.0, 0.0, _COLLAR_Z)),
           Geom("cylinder", (0.02125, 0.005625), (0.0, 0.0, -0.000625)))                  # solid below the socket
    + ring_geoms((0.0, 0.0, 0.0), IDENTITY_QUAT, BORE, 0.02125, 0.005, 0.0325, segments=16),
    studs=(Stud((0.0, 0.0, -0.0325), DOWN, PEG, 0.02625, MALE, "stud"),),
    sockets=(Socket((0.0, 0.0, 0.0325), LEG_SITE_QUAT, 0.0275, BORE, thread=FEMALE, name="socket"),),
)

BASE_SPEC = PrimitivePartSpec(
    name=BASE, mass=0.03396,
    geoms=(Geom("cylinder", (0.020, 0.00625)),                                            # hub
           Geom("box", (0.065, 0.011, 0.00625)), Geom("box", (0.011, 0.065, 0.00625)))    # arms (22 mm)
    + tuple(Geom("box", (0.015, 0.015, 0.00625), (sx * 0.050, sy * 0.050, 0.0))          # 30 mm wide arm ends
            for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1))),
    studs=(Stud((0.0, 0.0, -0.00625), DOWN, PEG, 0.0275, MALE, "stud"),),
)

LEG_SCREW = ScrewSpec(TOP, (LEG,), PITCH, sites=(site_at(TOP_SPEC.feature("socket")),), child_feature="stud",
                      thread_friction=THREAD_FRICTION)
# The base's arms give it ~8x the leg's inertia about the axis: when the hand lets go after driving it
# hard against the seated stop, the stop springs it back ~5 deg before thread friction catches it. The
# assembled check allows ~20 deg, so count it as seated within 8 deg (default 1 deg).
BASE_SCREW = ScrewSpec(LEG, (BASE,), PITCH, sites=(site_at(LEG_SPEC.feature("socket")),),
                       thread_friction=THREAD_FRICTION, seated_tol_deg=8.0)

# Like square_table we use 5 mm on every axis (FurnitureBench: 10 / 5 / 10 mm in its parent frames).
TARGETS = (
    AssemblyTarget(TOP, LEG, (seated_rel_pose(LEG_SCREW, 0, LEG_SPEC),)),
    AssemblyTarget(LEG, BASE, (seated_rel_pose(BASE_SCREW, 0, BASE_SPEC),)),
)

# Our part frames in FurnitureBench's: top and base keep their z axis (shifted to our origin), the leg
# remaps (x, -z, y) like square_table's legs, with its origin 19.375 mm along the leg.
FB_T_PART = {TOP: make_T((0.0, 0.0, 0.015)), LEG: make_T((0.0, 0.019375, 0.0), _C), BASE: make_T((0.0, 0.0, 0.005))}

# Layout (FurnitureBench reset pose 0, converted): top socket up near the fence, leg lying along +x with
# the stud toward the robot and a collar flat down, base standing on its stud tip (arms along W x / y).
_R2 = math.sqrt(0.5)
LAYOUT = (
    Placement(TOP, (0.2415, 0.0, 0.01625), (_R2, 0.0, 0.0, -_R2)),
    Placement(LEG, (0.120875, 0.13, 0.02125), (0.5, 0.5, 0.5, 0.5)),
    Placement(BASE, (0.1215, -0.12, 0.03375), (_R2, 0.0, 0.0, _R2)),
)

_HIGH_TAG_DATA = {
    TOP: [
        ((0.047, 0.216, -0.006), ((-0.986, 0.165, -0.004), (0.165, 0.986, 0.019), (0.007, 0.019, -1.0))),
        ((-0.077, 0.19, -0.006), ((-0.709, 0.705, -0.011), (0.705, 0.709, 0.008), (0.014, -0.002, -1.0))),
        ((0.076, 0.124, -0.003), ((0.956, 0.292, 0.002), (0.292, -0.956, 0.041), (0.013, -0.038, -0.999))),
    ],
    LEG: [
        ((-0.118, 0.263, -0.039), ((-0.79042137, 0.370952, 0.48747155), (0.3175943, 0.9286462, -0.19170311), (-0.52380127, 0.00329194, -0.85183406))),
        ((0.066, 0.301, -0.02), ((-0.402, 0.901, 0.16), (-0.915, -0.393, -0.088), (-0.016, -0.182, 0.983))),
        ((-0.09, 0.155, -0.018), ((-0.855, -0.507, 0.106), (-0.517, 0.85, -0.103), (-0.038, -0.143, -0.989))),
    ],
    BASE: [
        ((-0.135, 0.109, -0.03), ((0.911, -0.413, -0.006), (-0.412, -0.91, 0.035), (-0.02, -0.029, -0.999))),
        ((0.124, 0.16, -0.034), ((0.688, -0.726, -0.006), (-0.725, -0.688, 0.022), (-0.02, -0.011, -1.0))),
        ((-0.007, 0.293, -0.029), ((0.967, 0.256, -0.01), (0.256, -0.967, 0.015), (-0.006, -0.017, -1.0))),
    ],
}
HIGH_PRESETS = tuple(
    tuple(tag_pose_to_placement(part, *_HIGH_TAG_DATA[part][i], fb_T_part=FB_T_PART[part]) for part in (TOP, LEG, BASE))
    for i in range(3))

# Tabletop position noise [-5, +1] cm on both axes at every level (FurnitureBench's RoundTableTop
# override for from_skill 0). Footprints are FurnitureBench's reset lengths: the base's 2 x 2 cm is the
# design's (its real 13 cm cross would never fit next to the top); the leg's is centred on our origin.
_TOP_RANGE = {TOP: ((-0.05, 0.01), (-0.05, 0.01))}
RANDOMIZATION = RandomizationSpec(
    levels=dict(LEVELS),
    bounds_x=(0.0715, 0.3715),
    bounds_y=(-0.21, 0.21),
    footprints={TOP: (0.20, 0.20), LEG: (0.09125, 0.0425), BASE: (0.02, 0.02)},
    clearance=0.01,
    reset_obstacles=RESET_OBSTACLES,
    high_presets=HIGH_PRESETS,
    part_pos_range={"low": dict(_TOP_RANGE), "med": dict(_TOP_RANGE)},
)

_CORNER = corner_extras(0.20, 0.20)
CORNER_XY = _CORNER["corner_xy"]

PHASES = (
    Phase("grasp", TOP, label="grasp_top"),
    Phase("place_corner", TOP, label="place_top_corner", yaw_sym_deg=None),          # round: any yaw
    Phase("pick", LEG, label="pick_leg"), Phase("insert", LEG, TOP, "insert_leg"), Phase("screw", LEG, TOP, "screw_leg"),
    Phase("pick", BASE, label="pick_base"), Phase("insert", BASE, LEG, "insert_base"),
    Phase("screw", BASE, LEG, "screw_base"),
)

ROUND_TABLE = TaskSpec(
    name="round_table",
    parts=(TOP_SPEC, LEG_SPEC, BASE_SPEC),
    screws=(LEG_SCREW, BASE_SCREW),
    targets=TARGETS,
    layout=LAYOUT,
    randomization=RANDOMIZATION,
    scene=SCENE,
    phases=PHASES,
    max_steps=4000,
    extras=dict(_CORNER),
)
