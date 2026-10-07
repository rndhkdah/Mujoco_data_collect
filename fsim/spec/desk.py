"""desk task spec.

Design source: FurnitureBench (Heo et al., RSS 2023; MIT License): its desk (furniture/desk.py, config
"desk"). Dimensions measured on its meshes, masses from its sim_config, scene / layout / randomization /
presets / assembled poses re-expressed in this project's frames (see fsim/spec/types.py).

The desk is square_table's construction at another size: a 210 x 150 x 31.25 mm open tray (6.25 mm
plate and rim, threaded bosses around 4 blind holes at (+-80, +-50) mm) and four 30 x 30 x 100 mm legs
with a 25 mm stud. Same thread (9.375 mm pitch, right-handed), same hole and stud sizes.
"""
from __future__ import annotations

import math

from fsim.geometry import make_T, mat_to_quat, rot_y, rot_z
from fsim.spec.common import FB_AXES, LEVELS, RESET_OBSTACLES, SCENE, corner_extras, tag_pose_to_placement
from fsim.spec.types import (
    IDENTITY_QUAT, AssemblyTarget, LegPartSpec, Phase, Placement, RandomizationSpec, ScrewSpec, TaskSpec,
    ThreadSpec, TrayPartSpec,
)

TOP = "desk_top"
LEGS = tuple(f"desk_leg{i}" for i in range(1, 5))

PITCH = 0.009375
# Our part frames are FurnitureBench's (x, -z, y) turned 90 deg about the assembly axis (FB_T_PART), so
# that in the layout (and in the fence corner) the 210 mm side runs along W y at zero yaw: our x = their -z,
# our y = -their x. Legs get the same turn, which keeps the design's assembled rotation the identity.
HOLE_X, HOLE_Y = 0.050, 0.080          # FurnitureBench (x, z) = (+-80, +-50) mm
SEAT_HEIGHT = 0.065625                 # top half thickness 15.625 + leg body half 50 mm

TOP_SPEC = TrayPartSpec(
    name=TOP,
    size=(0.15, 0.21, 0.03125),
    mass=0.1694,
    hole_centers=tuple((sx * HOLE_X, sy * HOLE_Y) for sx in (1, -1) for sy in (1, -1)),
    plate_thickness=0.00625,
    rim_width=0.00625,
    boss_outer_radius=0.01625,
    bore_radius=0.0125,
    thread=ThreadSpec(pitch=PITCH, minor_radius=0.0109, major_radius=0.0147),
)


def _leg(name: str) -> LegPartSpec:
    # Mass: FurnitureBench states 56.2 g (its sim's density x mesh volume gives 29.4 g; we use the stated mass).
    # Stud: measured core 10.0 mm, crest 12.19 mm radius; collision peg 11.5 mm as square_table.
    return LegPartSpec(name=name, width=0.030, body_length=0.100, screw_length=0.025, mass=0.0562,
                       peg_radius=0.0115, thread=ThreadSpec(pitch=PITCH, minor_radius=0.0100, major_radius=0.0122))


LEG_SPECS = tuple(_leg(n) for n in LEGS)
SCREW = ScrewSpec(parent=TOP, children=LEGS, pitch=PITCH, seat_height=SEAT_HEIGHT)

HOLE_POSES = tuple(((sx * HOLE_X, sy * HOLE_Y, SEAT_HEIGHT), IDENTITY_QUAT) for sx in (1, -1) for sy in (1, -1))
# FurnitureBench's tolerance is (10, 5, 10) mm with 5 along the axis; like square_table we use 5 mm on every axis.
TARGETS = tuple(AssemblyTarget(parent=TOP, child=leg, rel_poses=HOLE_POSES) for leg in LEGS)

# Layout (FurnitureBench reset pose 0): tabletop holes up with its long side along W y, legs lying
# along +x with the stud toward the robot.
FB_T_PART = FB_AXES @ make_T(R=rot_z(math.pi / 2))     # our part frame in theirs (tabletop and legs)
LEG_REST_QUAT = tuple(float(v) for v in mat_to_quat(rot_y(math.pi / 2) @ rot_z(math.pi / 2)))
LAYOUT = (Placement(TOP, (0.2415, 0.0, 0.015625), IDENTITY_QUAT),) + tuple(
    Placement(leg, (0.0715, y, 0.015), LEG_REST_QUAT) for leg, y in zip(LEGS, (-0.15, -0.08, 0.08, 0.15)))

# FurnitureBench's three measured layouts in its base-tag frame (part frames as square_table's).
_HIGH_TAG_DATA = {
    TOP: [
        ((-0.002, 0.104, -0.017), ((-0.943, -0.008, 0.332), (-0.332, 0.046, -0.942), (-0.008, -0.999, -0.046))),
        ((-0.05966621, 0.25662723, -0.01701638), ((-0.99573445, -0.00066294, -0.09226216), (0.09208053, 0.05598222, -0.99417663), (0.00582412, -0.99843156, -0.05568231))),
        ((0.02099706, 0.16332534, -0.01756242), ((-0.98298514, -0.0125294, -0.18325743), (0.18326062, 0.0009035, -0.983064), (0.01248278, -0.99992114, 0.00140799))),
    ],
    LEGS[0]: [
        ((-0.118, 0.227, -0.019), ((-0.482, -0.876, -0.0), (-0.876, 0.482, 0.021), (-0.018, 0.01, -1.0))),
        ((-0.14033373, 0.1019271, -0.01784891), ((0.01059847, -0.53727186, -0.84334254), (-0.01687089, 0.8431738, -0.53737634), (0.99980152, 0.01992333, -0.00012791))),
        ((-0.11994868, 0.255285, -0.01709043), ((0.99956113, 0.02944234, -0.00326414), (0.02945254, -0.99956137, 0.0031231), (-0.00317076, -0.00321786, -0.99998987))),
    ],
    LEGS[1]: [
        ((-0.035, 0.279, -0.015), ((-0.888, 0.459, 0.014), (-0.459, -0.888, -0.01), (0.008, -0.015, 1.0))),
        ((0.05981686, 0.10712636, -0.0224212), ((0.18335986, 0.98291534, 0.01602139), (-0.98289353, 0.18359402, -0.01461495), (-0.01730669, -0.01306752, 0.99976486))),
        ((0.18557446, 0.1461009, -0.02011154), ((-0.01125821, 0.0096676, 0.99988991), (0.00722122, 0.999928, -0.00958666), (-0.99991053, 0.0071125, -0.01132721))),
    ],
    LEGS[2]: [
        ((0.068, 0.323, -0.011), ((-0.343, -0.934, 0.097), (-0.909, 0.356, 0.217), (-0.237, -0.013, -0.971))),
        ((0.1628194, 0.1732633, -0.02252014), ((-0.80881345, -0.58802843, -0.00660876), (-0.58803391, 0.80883527, -0.00128764), (0.00610256, 0.00284471, -0.99997747))),
        ((-0.17351371, 0.12436365, -0.01877135), ((-0.8064419, -0.5913133, -0.00050146), (-0.59120387, 0.80627567, 0.01993935), (-0.01138609, 0.01637639, -0.99980122))),
    ],
    LEGS[3]: [
        ((0.166, 0.147, -0.02), ((-0.011, -0.199, 0.98), (0.019, 0.98, 0.2), (-1.0, 0.021, -0.007))),
        ((0.12731045, 0.3074856, -0.01941558), ((0.97035968, -0.24163274, 0.00395575), (0.24165221, 0.97034711, -0.00554586), (-0.00249839, 0.00633739, 0.99997681))),
        ((0.0530014, 0.3194076, -0.02061281), ((-0.00796196, -0.99176419, 0.12782967), (0.00122496, 0.12782392, 0.99179614), (-0.99996758, 0.00805318, 0.00019711))),
    ],
}
HIGH_PRESETS = tuple(
    tuple(tag_pose_to_placement(part, *_HIGH_TAG_DATA[part][i], fb_T_part=FB_T_PART)
          for part in (TOP,) + LEGS) for i in range(3))

# The tabletop's position noise is [-5, +2] cm on both axes at every level (FurnitureBench's
# DeskTableTop.randomize_init_pose override, applied for from_skill 0); yaw noise follows the level.
_TOP_RANGE = {TOP: ((-0.05, 0.02), (-0.05, 0.02))}
RANDOMIZATION = RandomizationSpec(
    levels=dict(LEVELS),
    bounds_x=(0.0715, 0.3715),
    bounds_y=(-0.21, 0.21),
    footprints={TOP: (0.15, 0.21), **{leg: (0.125, 0.03) for leg in LEGS}},   # FurnitureBench reset lengths
    clearance=0.01,
    reset_obstacles=RESET_OBSTACLES,
    high_presets=HIGH_PRESETS,
    part_pos_range={"low": dict(_TOP_RANGE), "med": dict(_TOP_RANGE)},
)

# Phase-2 target: the tabletop snug in the +y fence corner, long side along W y (yaw 0 mod 180).
_CORNER = corner_extras(TOP_SPEC.size[0], TOP_SPEC.size[1])
CORNER_XY = _CORNER["corner_xy"]


def _leg_phases(leg: str) -> tuple[Phase, ...]:
    s = leg[len("desk_"):]
    return (Phase("pick", leg, label=f"pick_{s}"), Phase("insert", leg, TOP, f"insert_{s}"),
            Phase("screw", leg, TOP, f"screw_{s}"))


PHASES = (Phase("grasp", TOP, label="grasp_top"),
          Phase("place_corner", TOP, label="place_top_corner", yaw_sym_deg=180.0)) + tuple(
    ph for leg in LEGS for ph in _leg_phases(leg))          # FurnitureBench order: leg1 .. leg4

DESK = TaskSpec(
    name="desk",
    parts=(TOP_SPEC,) + LEG_SPECS,
    screws=(SCREW,),
    targets=TARGETS,
    layout=LAYOUT,
    randomization=RANDOMIZATION,
    scene=SCENE,
    phases=PHASES,
    max_steps=4500,
    extras=dict(_CORNER),
)
