"""stool task spec: a round seat and three splayed legs screwed into it.

Design source: FurnitureBench (Heo et al., RSS 2023; MIT License). Masses, layout, randomization,
measured layouts, assembled poses and tolerances follow its stool design; dimensions were measured
from its part geometry and rebuilt from primitives in this project's frames (see fsim/spec/types.py).

  seat  Ø100 x 30 mm open tray (holes up): 2.5 mm plate, 2 mm rim, three threaded bosses at r 32.5 mm
  legs  35 x 35 x 60 mm body sheared 15 degrees (12 stepped slabs), vertical threaded stud under the
        bottom-face centre; screwed in, the body leans radially outward
Threads: the FurnitureBench family (9.375 mm pitch, right-handed), as square_table.
"""
from __future__ import annotations

import math

from fsim.geometry import make_T, mat_to_quat, rot_z
from fsim.spec.common import _C, LEVELS, RESET_OBSTACLES, SCENE, corner_extras, ring_geoms, seated_rel_pose, tag_pose_to_placement
from fsim.spec.types import (
    IDENTITY_QUAT, AssemblyTarget, Geom, Phase, Placement, PrimitivePartSpec, RandomizationSpec, ScrewSpec, Site,
    Socket, Stud, TaskSpec, ThreadSpec,
)

SEAT = "stool_seat"
LEGS = tuple(f"stool_leg{i}" for i in range(1, 4))

PITCH = 0.009375
MALE = ThreadSpec(PITCH, 0.0072, 0.0128)
FEMALE = ThreadSpec(PITCH, 0.0109, 0.0147)
DOWN = (0.0, 1.0, 0.0, 0.0)                      # feature z along the part's -z

SEAT_R, SEAT_HALF_H = 0.050, 0.015
PLATE_TOP = -0.0125                              # hole floor (2.5 mm plate)
HOLE_R = 0.0325                                  # socket axes on this radius (measured 32.50 mm)
BORE_R, BOSS_R = 0.0125, 0.0165
SOCKET_DEPTH = SEAT_HALF_H - PLATE_TOP           # 27.5 mm
SOCKET_ANGLES = (-90.0, 30.0, 150.0)             # degrees in the seat frame

SPLAY = math.radians(15.0)
LEG_HALF_W, LEG_HALF_H = 0.0175, 0.030           # 35 x 35 body, 60 mm tall
STUD_X = LEG_HALF_H * math.tan(SPLAY)            # stud under the bottom-face centre (8.04 mm)
STUD_LEN = 0.027                                 # 27.5 mm in the design, 0.5 mm short of the floor


def _site(angle_deg: float) -> Site:
    a = math.radians(angle_deg)
    # The leg's -x (its lean) must point radially outward: seated yaw = angle + 180 degrees.
    return Site((HOLE_R * math.cos(a), HOLE_R * math.sin(a), SEAT_HALF_H),
                tuple(mat_to_quat(rot_z(a + math.pi))), SOCKET_DEPTH)


SITES = tuple(_site(a) for a in SOCKET_ANGLES)

SEAT_SPEC = PrimitivePartSpec(
    name=SEAT, mass=0.05734,
    geoms=(Geom("cylinder", (SEAT_R, (PLATE_TOP + SEAT_HALF_H) / 2), (0.0, 0.0, (PLATE_TOP - SEAT_HALF_H) / 2)),)
    + ring_geoms((0, 0, 0), IDENTITY_QUAT, SEAT_R - 0.002, SEAT_R, PLATE_TOP, SEAT_HALF_H, segments=24)
    + tuple(g for s in SITES for g in ring_geoms((s.pos[0], s.pos[1], 0.0), IDENTITY_QUAT, BORE_R, BOSS_R,
                                                 PLATE_TOP, SEAT_HALF_H))
    + ring_geoms((0, 0, 0), IDENTITY_QUAT, 0.0165, 0.0195, PLATE_TOP, SEAT_HALF_H),     # central hub
    sockets=tuple(Socket(s.pos, s.quat, SOCKET_DEPTH, BORE_R, thread=FEMALE, name=f"hole{i + 1}")
                  for i, s in enumerate(SITES)),
    rgba=(0.80, 0.68, 0.52, 1.0),
)

_SLABS = 12                                       # stepped shear: 1.3 mm steps
_SLAB_Z = tuple(-LEG_HALF_H + LEG_HALF_H / _SLABS * (2 * k + 1) for k in range(_SLABS))


def _leg(name: str) -> PrimitivePartSpec:
    return PrimitivePartSpec(
        name=name, mass=0.02707,
        geoms=tuple(Geom("box", (LEG_HALF_W, LEG_HALF_W, LEG_HALF_H / _SLABS), (-z * math.tan(SPLAY), 0.0, z)) for z in _SLAB_Z),
        studs=(Stud((STUD_X, 0.0, -LEG_HALF_H), DOWN, 0.0115, STUD_LEN, MALE, "stud"),),
    )


LEG_SPECS = tuple(_leg(n) for n in LEGS)

SCREW = ScrewSpec(parent=SEAT, children=LEGS, pitch=PITCH, sites=SITES, child_feature="stud")

REL_POSES = tuple(seated_rel_pose(SCREW, i, LEG_SPECS[0]) for i in range(len(SITES)))
TARGETS = tuple(AssemblyTarget(SEAT, leg, REL_POSES) for leg in LEGS)

# Layout (FurnitureBench reset pose, skill 0): seat holes-up with socket hole1 toward the robot; legs lying
# on a flat side along +x, stud toward the robot.
LEG_REST_QUAT = (0.5, 0.5, 0.5, 0.5)
LAYOUT = (
    Placement(SEAT, (0.2915, 0.0, SEAT_HALF_H), tuple(mat_to_quat(rot_z(-math.pi / 2)))),
    Placement(LEGS[0], (0.1815, -0.10, LEG_HALF_W), LEG_REST_QUAT),
    Placement(LEGS[1], (0.1815, 0.00, LEG_HALF_W), LEG_REST_QUAT),
    Placement(LEGS[2], (0.1815, 0.10, LEG_HALF_W), LEG_REST_QUAT),
)

# --- High-randomness presets: FurnitureBench's measured layouts in its AprilTag base-tag frame.
# Our leg origin is the body centre, 0.38 mm / -0.27 mm off theirs.
_LEG_FB = make_T((0.00038, 0.0, -0.00027), _C)
_HIGH_TAG_DATA = {
    SEAT: [
        ((-0.06, 0.227, -0.016), ((-0.876, 0.247, -0.414), (0.474, 0.292, -0.83), (-0.084, -0.924, -0.373))),
        ((0.027, 0.249, -0.016), ((-0.174, 0.003, -0.985), (0.984, 0.028, -0.174), (0.027, -1.0, -0.008))),
        ((0.13124122, 0.09935549, -0.02212791), ((-0.754682, -0.01281741, -0.65596557), (0.65574956, 0.01750702, -0.75477552), (0.02115828, -0.99976456, -0.00480723))),
    ],
    LEGS[0]: [
        ((0.076, 0.287, -0.021), ((0.353, -0.936, 0.01), (0.936, 0.353, -0.005), (0.001, 0.011, 1.0))),
        ((0.11, 0.151, -0.02), ((0.001, -0.491, -0.871), (-0.022, 0.871, -0.491), (1.0, 0.02, -0.01))),
        ((-0.11369069, 0.19002846, -0.01333283), ((-0.01497109, -0.00862187, 0.99985069), (-0.24469948, 0.96958756, 0.00469694), (-0.96948332, -0.24459264, -0.01662555))),
    ],
    LEGS[1]: [
        ((-0.134, 0.186, -0.02), ((0.083, -0.705, -0.705), (-0.104, 0.697, -0.709), (0.991, 0.132, -0.015))),
        ((-0.045, 0.188, -0.018), ((-0.027, -0.122, -0.992), (0.239, 0.963, -0.125), (0.971, -0.24, 0.003))),
        ((0.06150678, 0.28508818, -0.02003413), ((-0.03810083, 0.09744918, 0.99451101), (-0.25105661, 0.96237803, -0.10391883), (-0.96722233, -0.25363794, -0.01220215))),
    ],
    LEGS[2]: [
        ((0.126, 0.171, -0.02), ((0.008, -0.101, -0.995), (-0.024, 0.995, -0.101), (1.0, 0.024, 0.005))),
        ((-0.108, 0.253, -0.02), ((0.033, 0.242, 0.97), (-0.144, -0.959, 0.244), (0.989, -0.148, 0.004))),
        ((0.02115019, 0.18929027, -0.0199405), ((0.87092847, -0.4912734, 0.0115781), (0.49131191, 0.87098348, -0.00056219), (-0.00980814, 0.00617809, 0.99993283))),
    ],
}
HIGH_PRESETS = tuple(
    tuple(tag_pose_to_placement(p, *_HIGH_TAG_DATA[p][i], fb_T_part=None if p == SEAT else _LEG_FB)
          for p in (SEAT,) + LEGS)
    for i in range(3)
)

RANDOMIZATION = RandomizationSpec(
    levels=LEVELS,
    bounds_x=(0.0715, 0.3715),
    bounds_y=(-0.21, 0.21),
    footprints={SEAT: (0.0875, 0.0875), **{leg: (0.0875, 0.051077) for leg in LEGS}},
    clearance=0.01,
    reset_obstacles=RESET_OBSTACLES,
    high_presets=HIGH_PRESETS,
)

ORDER = (LEGS[2], LEGS[1], LEGS[0])              # leg3 first, as the design's skill sequence
PHASES = (Phase("grasp", SEAT, label="grasp_seat"),
          Phase("place_corner", SEAT, label="place_seat_corner", yaw_sym_deg=None)) + tuple(
    ph for leg in ORDER for ph in (Phase("pick", leg, label=f"pick_{leg[-4:]}"),
                                   Phase("insert", leg, SEAT, f"insert_{leg[-4:]}"),
                                   Phase("screw", leg, SEAT, f"screw_{leg[-4:]}")))

STOOL = TaskSpec(
    name="stool",
    parts=(SEAT_SPEC,) + LEG_SPECS,
    screws=(SCREW,),
    targets=TARGETS,
    layout=LAYOUT,
    randomization=RANDOMIZATION,
    scene=SCENE,
    phases=PHASES,
    max_steps=4000,
    extras=corner_extras(2 * SEAT_R, 2 * SEAT_R, yaw_tol_deg=180.0),
)
