"""chair task spec.

Design source: FurnitureBench (Heo et al., RSS 2023; MIT License): its chair's dimensions, masses
(density x mesh volume), thread pitches, assembled poses, scene layout, randomization and measured
layouts, re-expressed in this project's frames (see fsim/spec/types.py). The geometry is
parametric primitives measured from the design's meshes; no FurnitureBench code or assets are used.

The chair: a seat tray (open face = the chair's underside), two front legs screwed into blind holes
in that face, a back plate (its posts are the rear legs) pressed onto two horizontal studs on the
seat's rear edge, and two cube nuts screwed onto those studs. A tongue between the studs keys into
the slot between the back's posts.

Assembly order: legs with the seat flat (tray up) in the fence corner, then the seat is stood up on
its rear face with the studs up (the back cannot go on while the seat lies flat: it would go
through the table), the back is dropped onto the studs and the nuts are screwed down.

Part frames (origin = bounding-box centre of the collision geoms, z = assembly axis):
  seat  z = leg-hole axis (tray open face), studs along -y, tongue toward -y
  leg   z = long axis, stud on -z (as the square_table leg)
  back  z = plate normal (insertion axis), holes and post ends toward +y, top rail toward -y
  nut   z = stud axis outward, bore open on -z
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from fsim.geometry import make_T, mat_to_quat, quat_to_mat, rot_x, rot_z
from fsim.spec.common import (
    LEVELS, RESET_OBSTACLES, SCENE, corner_extras, ring_geoms, seated_rel_pose, tag_pose_to_placement,
)
from fsim.spec.types import (
    IDENTITY_QUAT, AssemblyTarget, Geom, InsertSpec, LegPartSpec, Phase, Placement, PrimitivePartSpec,
    RandomizationSpec, ScrewSpec, Site, Socket, Stud, TaskSpec, ThreadSpec, site_at,
)

SEAT, BACK = "chair_seat", "chair_back"
LEGS = ("chair_leg1", "chair_leg2")
NUTS = ("chair_nut1", "chair_nut2")

LEG_PITCH = 0.009375           # measured 9.29 mm; the design's square_table pitch
NUT_PITCH = 0.006              # measured 6.00 mm on the studs (the nut mesh says 5.62)
SEAT_GAP = 0.0003

AXIS_OUT = tuple(float(v) for v in mat_to_quat(rot_x(math.pi / 2)))      # feature z along the part's -y (seat studs)
DOWN = tuple(float(v) for v in mat_to_quat(rot_x(math.pi)))              # feature z along the part's -z

# ---------------------------------------------------------------------------------------------
# Seat, built in its main-block frame (x +-50, y +-35.25, z +-15 mm), then recentred on the
# bounding box of its geoms (the tongue makes it 14.75 mm off along y).
HX, HY, HZ = 0.050, 0.03525, 0.015
FLOOR = 0.005                  # tray floor (z -15..-10 mm)
RIM = 0.00375
TONGUE_HX = 0.0175             # measured 19.4; narrowed so it never fights the stud holes (EST)
TONGUE_Y = -0.06475            # tongue end (measured)
HOLE_XY = ((0.03375, 0.01875), (-0.03375, 0.01875))
BORE_R, BOSS_R, BORE_DEPTH = 0.0125, 0.0155, 0.025
STUD_X = (0.035, -0.035)
SHANK_R, SHANK_LEN = 0.0078, 0.02975       # stud face to the thread start (y -35.25 .. -65 mm)
STUD_R, STUD_LEN = 0.0066, 0.01725         # threaded part, collision radius (EST; thread 4.7 / 7.3 mm)
BACK_T = 0.030                 # back plate thickness
LEG_FEMALE = ThreadSpec(LEG_PITCH, 0.0109, 0.0147)                  # EST (square_table values)
STUD_THREAD = ThreadSpec(NUT_PITCH, 0.0047, 0.0073)                 # measured
NUT_THREAD = ThreadSpec(NUT_PITCH, 0.0066, 0.0092)                  # measured


def _boxes_aabb(geoms) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box of box geoms (fsim.models.parts.parts_aabb; importing it here would be circular)."""
    signs = np.array(list(np.ndindex(2, 2, 2))) * 2 - 1
    pts = np.concatenate([np.asarray(g.pos) + (signs * g.size) @ quat_to_mat(g.quat).T for g in geoms])
    return pts.min(axis=0), pts.max(axis=0)


def _wall(x0, x1, y0, y1, z0=-HZ + FLOOR, z1=HZ) -> Geom:
    return Geom("box", ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


_ty = -HY                      # tongue root
_seat_geoms = (
    _wall(-HX, HX, -HY, HY, -HZ, -HZ + FLOOR),                       # floor, main block
    _wall(-TONGUE_HX, TONGUE_HX, TONGUE_Y, _ty, -HZ, -HZ + FLOOR),   # floor, tongue
    _wall(-HX, HX, HY - RIM, HY),                                    # rear wall
    _wall(-HX, -HX + RIM, -HY, HY), _wall(HX - RIM, HX, -HY, HY),    # side walls
    _wall(-HX, -TONGUE_HX, -HY, -HY + RIM), _wall(TONGUE_HX, HX, -HY, -HY + RIM),   # front walls
    _wall(-TONGUE_HX, -TONGUE_HX + RIM, TONGUE_Y, _ty), _wall(TONGUE_HX - RIM, TONGUE_HX, TONGUE_Y, _ty),
    _wall(-TONGUE_HX, TONGUE_HX, TONGUE_Y, TONGUE_Y + RIM),          # tongue end wall
)
for _hx, _hy in HOLE_XY:                                             # threaded bosses around the leg holes
    _seat_geoms += ring_geoms((_hx, _hy, HZ), IDENTITY_QUAT, BORE_R, BOSS_R, -BORE_DEPTH, 0.0)

_seat_raw = PrimitivePartSpec(
    name=SEAT, mass=0.06187,
    geoms=_seat_geoms,
    studs=tuple(Stud((x, -HY, 0.0), AXIS_OUT, SHANK_R, SHANK_LEN, None, f"shank{i}") for i, x in enumerate(STUD_X, 1))
    + tuple(Stud((x, -HY - SHANK_LEN, 0.0), AXIS_OUT, STUD_R, STUD_LEN, STUD_THREAD, f"stud{i}")
            for i, x in enumerate(STUD_X, 1)),
    sockets=tuple(Socket((x, y, HZ), IDENTITY_QUAT, BORE_DEPTH, BORE_R, thread=LEG_FEMALE, name=f"hole{i}")
                  for i, (x, y) in enumerate(HOLE_XY, 1)),
    rgba=(0.80, 0.68, 0.52, 1.0),
)
_lo, _hi = _boxes_aabb(_seat_raw.geoms)
SEAT_OFFSET = tuple(float(v) for v in (_lo + _hi) / 2)               # main-block frame -> seat frame


def _shift(pos) -> tuple[float, float, float]:
    """Main-block frame point -> seat frame."""
    return tuple(float(p - o) for p, o in zip(pos, SEAT_OFFSET))


SEAT_SPEC = replace(
    _seat_raw,
    geoms=tuple(replace(g, pos=_shift(g.pos)) for g in _seat_raw.geoms),
    studs=tuple(replace(s, pos=_shift(s.pos)) for s in _seat_raw.studs),
    sockets=tuple(replace(s, pos=_shift(s.pos)) for s in _seat_raw.sockets),
)

# ---------------------------------------------------------------------------------------------
LEG_SPECS = tuple(LegPartSpec(name=n, width=0.030, body_length=0.060, screw_length=0.025, mass=0.02244,
                              peg_radius=0.0105, thread=ThreadSpec(LEG_PITCH, 0.0078, 0.0122)) for n in LEGS)

# Back: two 30 x 30 mm posts with a through hole each, a 12.5 mm top rail and centre panel. The slot
# between the posts above the panel stays open for the seat's tongue.
BACK_HX, BACK_HY, BACK_HZ = 0.0625, 0.105, 0.015
POST_X0, POST_X1, POST_Y0 = 0.020, 0.050, -0.070
HOLE_Y = 0.03003               # hole axes (+-35, 30.03) mm from the plate centre (measured)
BACK_HOLE = 0.0095             # square hole half-width (EST; measured r 8.45 mm, shank r 7.8 mm)
RAIL_Y, THIN = -0.070, 0.00625


def _box(x0, x1, y0, y1, z0, z1) -> Geom:
    return Geom("box", ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


_back_geoms = (
    _box(-BACK_HX, BACK_HX, -BACK_HY, RAIL_Y, -THIN, THIN),                  # top rail
    _box(-POST_X0, POST_X0, RAIL_Y, 0.0125, -THIN, THIN),                    # centre panel
)
for _s in (1, -1):
    _hx = _s * 0.035
    _x0, _x1 = sorted((_s * POST_X0, _s * POST_X1))
    _back_geoms += (
        _box(_x0, _x1, POST_Y0, HOLE_Y - BACK_HOLE, -BACK_HZ, BACK_HZ),
        _box(_x0, _x1, HOLE_Y + BACK_HOLE, BACK_HY, -BACK_HZ, BACK_HZ),
        _box(_x0, _hx - BACK_HOLE, HOLE_Y - BACK_HOLE, HOLE_Y + BACK_HOLE, -BACK_HZ, BACK_HZ),
        _box(_hx + BACK_HOLE, _x1, HOLE_Y - BACK_HOLE, HOLE_Y + BACK_HOLE, -BACK_HZ, BACK_HZ),
    )
BACK_SPEC = PrimitivePartSpec(
    name=BACK, mass=0.12316, geoms=_back_geoms,
    sockets=tuple(Socket((x, HOLE_Y, BACK_HZ), IDENTITY_QUAT, 2 * BACK_HZ, BACK_HOLE, name=f"hole{i}")
                  for i, x in enumerate(STUD_X, 1)),
    rgba=(0.85, 0.74, 0.58, 1.0),
)

# Nut: 30 mm cube with a blind square bore open on -z, 18 mm deep (the design's 15.2 mm would hit
# the stud tip before the nut reaches the back).
NUT_H, NUT_BORE, NUT_DEPTH = 0.015, 0.0072, 0.018
_nz0, _nz1 = -NUT_H, -NUT_H + NUT_DEPTH


def _nut(name: str) -> PrimitivePartSpec:
    g = (
        _box(NUT_BORE, NUT_H, -NUT_H, NUT_H, _nz0, _nz1), _box(-NUT_H, -NUT_BORE, -NUT_H, NUT_H, _nz0, _nz1),
        _box(-NUT_BORE, NUT_BORE, NUT_BORE, NUT_H, _nz0, _nz1), _box(-NUT_BORE, NUT_BORE, -NUT_H, -NUT_BORE, _nz0, _nz1),
        _box(-NUT_H, NUT_H, -NUT_H, NUT_H, _nz1, NUT_H),                     # cap
    )
    return PrimitivePartSpec(name=name, mass=0.01025, geoms=g,
                             sockets=(Socket((0.0, 0.0, -NUT_H), DOWN, NUT_DEPTH, NUT_BORE, thread=NUT_THREAD),),
                             rgba=(0.70, 0.60, 0.46, 1.0))


NUT_SPECS = tuple(_nut(n) for n in NUTS)

# ---------------------------------------------------------------------------------------------
# Mechanisms (seat frame).
LEG_SCREW = ScrewSpec(SEAT, LEGS, LEG_PITCH, sites=tuple(site_at(SEAT_SPEC.feature(f"hole{i}")) for i in (1, 2)),
                      catch_radius=0.0095)

# Back: its mate frame is the hole-axis midpoint on its -z face; the site is that point when the
# back sits on the seat's stud face. Engages once the studs are 7 mm into the holes (travel =
# 47 - 7 mm), seats on the face. Second site: the back turned 180 deg about the stud axis (posts
# up), which the primitive geometry also accepts.
STUD_FACE = _shift((0.0, -HY, 0.0))
_site0 = make_T(STUD_FACE, rot_x(math.pi / 2))
BACK_INSERT = InsertSpec(
    SEAT, (BACK,),
    sites=tuple(Site(tuple(float(v) for v in T[:3, 3]), tuple(float(v) for v in mat_to_quat(T[:3, :3])))
                for T in (_site0, _site0 @ make_T(R=rot_z(math.pi)))),
    travel=SHANK_LEN + STUD_LEN - 0.007,
    child_mate=((0.0, HOLE_Y, -BACK_HZ), IDENTITY_QUAT),
    seat_gap=SEAT_GAP, engage_radial_tol=0.002, engage_tilt_deg=5.0, engage_twist_deg=5.0,
)

# Nuts (inverted: the seat owns the studs). Site = the nut's mouth when seated, on the back's top
# face; the catch collar (r between the nut bore and its outside, and inside the back's holes)
# holds the nut 1.5 turns out until the thread catches. Low thread friction: the standing chair is
# not braced by the fence (see tests/test_task_chair.py::test_standing_chair_does_not_spin).
NUT_SCREW = ScrewSpec(
    SEAT, NUTS, NUT_PITCH,
    sites=tuple(Site(_shift((x, -HY - SEAT_GAP - BACK_T, 0.0)), AXIS_OUT) for x in STUD_X),
    thread_friction=0.004, catch_radius=0.0082,
)

TARGETS = (
    tuple(AssemblyTarget(SEAT, leg, tuple(seated_rel_pose(LEG_SCREW, i, LEG_SPECS[0]) for i in range(2)))
          for leg in LEGS)
    + (AssemblyTarget(SEAT, BACK, tuple(seated_rel_pose(BACK_INSERT, i, BACK_SPEC) for i in range(2))),)
    + tuple(AssemblyTarget(SEAT, nut, tuple(seated_rel_pose(NUT_SCREW, i, NUT_SPECS[0]) for i in range(2)),
                           requires=((SEAT, BACK),)) for nut in NUTS)
)

# ---------------------------------------------------------------------------------------------
# FurnitureBench part frames -> ours (4x4: our frame in theirs).
_C_STD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])    # our axes = their (x, -z, y)
_C_NUT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])    # our axes = their (x, z, -y)
FB_T = {
    # their seat origin is 0.5 mm off our main-block centre (along their y)
    SEAT: make_T(_C_STD @ np.asarray(SEAT_OFFSET) + (0.0, -0.0005, 0.0), _C_STD),
    BACK: make_T((0.0, 0.00247, 0.0)),
    **{leg: make_T(R=_C_STD) for leg in LEGS},
    **{nut: make_T((0.0, -0.001, 0.0), _C_NUT) for nut in NUTS},
}
PART_ORDER = (SEAT, BACK) + LEGS + NUTS
REST_Z = {SEAT: HZ, BACK: BACK_HZ, **{n: 0.015 for n in LEGS + NUTS}}


def fb_placement(part: str, pos_tag, rot_tag, rest: bool = False) -> Placement:
    pl = tag_pose_to_placement(part, pos_tag, rot_tag, fb_T_part=FB_T[part])
    return replace(pl, pos=(pl.pos[0], pl.pos[1], REST_Z[part])) if rest else pl


def _rpy(x, y, z):
    """The design's rot_mat: Rz @ Ry @ Rx."""
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# The design's skill-0 reset poses (tag frame): seat tray up with the studs toward the robot, back
# flat, legs lying with the stud toward the robot, nuts bore down.
_RESET_TAG = {
    SEAT: ((0.03, 0.27, -0.015), _rpy(-math.pi / 2, 0, math.pi)),
    BACK: ((-0.10, 0.20, -0.015), _rpy(0, math.pi, 0)),
    LEGS[0]: ((0.01, 0.12, -0.015), np.eye(3)),
    LEGS[1]: ((0.07, 0.12, -0.015), np.eye(3)),
    NUTS[0]: ((0.13, 0.12, -0.015), _rpy(math.pi / 2, 0, 0)),
    NUTS[1]: ((0.19, 0.12, -0.015), _rpy(math.pi / 2, 0, 0)),
}
LAYOUT = tuple(fb_placement(p, *_RESET_TAG[p], rest=True) for p in PART_ORDER)

# The design's init-assembled seat pose: standing on its rear face, studs up, tray toward the robot.
SEAT_UPRIGHT = fb_placement(SEAT, (0.0, 0.16, -0.035), _rpy(math.pi, 0, 0))

_HIGH_TAG_DATA = {
    SEAT: [
        ((0.05986764, 0.19544938, -0.01780899), ((-0.92918456, -0.01316611, -0.36938149), (0.3693251, 0.00657321, -0.929277), (0.014663, -0.9998917, -0.00124513))),
        ((-0.17163497, 0.15262844, -0.01651937), ((-0.83291751, -0.006127, 0.55336332), (-0.55339634, 0.00741897, -0.83288503), (0.00099766, -0.99995375, -0.00957008))),
        ((0.06676043, 0.24115437, -0.0212097), ((-0.02653122, -0.01253995, 0.99956936), (-0.99961007, -0.0083738, -0.02663738), (0.00870425, -0.99988633, -0.01231295))),
    ],
    BACK: [
        ((-0.08739488, 0.19979191, -0.01221029), ((-0.76807487, 0.64029282, -0.00926405), (0.64025527, 0.76813066, 0.00695575), (0.01156972, -0.00058881, -0.99993277))),
        ((0.0152305, 0.17737612, -0.01473497), ((-0.15836088, -0.98732817, -0.0102406), (-0.98735344, 0.15827014, 0.00914752), (-0.00741083, 0.0115597, -0.99990565))),
        ((-0.10485372, 0.15906905, -0.01489258), ((-0.46160662, 0.8870849, -0.00000017), (0.8870849, 0.46160638, 0.00000086), (0.00000084, 0.00000025, -1.0000002))),
    ],
    LEGS[0]: [
        ((0.12987243, 0.1504644, -0.0237539), ((-0.01107043, 0.12764195, 0.99175853), (0.00335702, 0.99181849, -0.12761219), (-0.99993306, 0.00191663, -0.01140836))),
        ((-0.07327363, 0.28158466, -0.01686125), ((-0.00929469, -0.8878895, 0.45996296), (0.00425327, 0.4599435, 0.88793802), (-0.99994779, 0.01020944, -0.00049862))),
        ((0.09447257, 0.10305008, -0.02093053), ((-0.9924311, -0.12201616, -0.01388437), (-0.12223718, 0.9923659, 0.01637016), (0.01178095, 0.01794344, -0.99976969))),
    ],
    LEGS[1]: [
        ((0.08744325, 0.32398197, -0.02274978), ((-0.00169327, 0.30636734, -0.95191187), (-0.00564334, 0.95189512, 0.30637199), (0.99998266, 0.00589073, 0.00011711))),
        ((0.02714946, 0.31022108, -0.01758601), ((0.99577844, -0.08591485, 0.03231045), (0.08579749, 0.99630004, 0.00500382), (-0.03262081, -0.00221054, 0.99946535))),
        ((0.16358277, 0.11960384, -0.01879693), ((0.97426367, -0.22378716, 0.0270115), (0.22386295, 0.97462058, 0.00022389), (-0.02637606, 0.00582875, 0.9996351))),
    ],
    NUTS[0]: [
        ((0.01052985, 0.30740196, -0.02801788), ((0.05493468, 0.02182969, 0.99825132), (0.99848986, -0.00077346, -0.05493096), (-0.0004271, 0.99976146, -0.02183929))),
        ((0.0829217, 0.27938807, -0.03095993), ((0.16628975, 0.02535662, 0.98575091), (0.98604178, -0.01272193, -0.16601163), (0.00833112, 0.99959761, -0.02711824))),
        ((-0.05123203, 0.29473808, -0.02397889), ((0.95705044, 0.00778796, 0.2898168), (0.28989899, -0.01327884, -0.95696509), (-0.00360438, 0.99988151, -0.01496624))),
    ],
    NUTS[1]: [
        ((-0.12835887, 0.289353, -0.02523068), ((0.18104237, 0.00664857, 0.98345286), (0.98347282, -0.00349244, -0.18102252), (0.00223106, 0.99997187, -0.00717101))),
        ((-0.13410832, 0.26472944, -0.02297941), ((0.50522768, 0.01058421, 0.86292118), (0.86298418, -0.00828896, -0.50516284), (0.00180602, 0.99990964, -0.0133218))),
        ((-0.13044557, 0.29677352, -0.02132733), ((-0.2258541, 0.01629639, 0.97402483), (0.97405064, -0.01128122, 0.22604883), (0.01467198, 0.99980354, -0.0133256))),
    ],
}
HIGH_PRESETS = tuple(tuple(fb_placement(p, *_HIGH_TAG_DATA[p][i]) for p in PART_ORDER) for i in range(3))

RANDOMIZATION = RandomizationSpec(
    levels=LEVELS,
    bounds_x=(0.0715, 0.3715),
    bounds_y=(-0.21, 0.21),
    # (W x, W y) = the design's (reset_y_len, reset_x_len)
    footprints={SEAT: (0.11875, 0.10), BACK: (0.21, 0.10), **{n: (0.085, 0.03) for n in LEGS},
                **{n: (0.03, 0.03) for n in NUTS}},
    clearance=0.01,
    reset_obstacles=RESET_OBSTACLES,
    high_presets=HIGH_PRESETS,
    # The design narrows the seat's range ("too big"); it does so at every level, applied at med only here.
    part_pos_range={"med": {SEAT: ((-0.05, 0.035), (-0.05, 0.035))}},
    max_tries=3_000_000,       # med accepts about 1 in 100k layouts
)

# Phase 2: the seat (tray up, studs toward the robot) snug in the +y fence corner, rear face on the
# front fence. The design's own corner pose is 2.5 cm off the front fence, inside the tolerance.
_SEAT_LO, _SEAT_HI = _boxes_aabb(SEAT_SPEC.geoms)
CORNER = corner_extras(float(_SEAT_HI[1] - _SEAT_LO[1]), float(_SEAT_HI[0] - _SEAT_LO[0]), tol=0.025)
SEAT_YAW_DEG = -90.0


def _screw_phases(part: str, suffix: str) -> tuple[Phase, ...]:
    return (Phase("pick", part, label=f"pick_{suffix}"), Phase("insert", part, SEAT, f"insert_{suffix}"),
            Phase("screw", part, SEAT, f"screw_{suffix}"))


PHASES = (
    (Phase("grasp", SEAT, label="grasp_seat"),
     Phase("place_corner", SEAT, label="place_seat_corner", yaw_sym_deg=360.0, yaw_deg=SEAT_YAW_DEG))
    + _screw_phases(LEGS[0], "leg1") + _screw_phases(LEGS[1], "leg2")
    + (Phase("place_pose", SEAT, label="flip_seat", axis=(0.0, -1.0, 0.0), axis_dir=(0.0, 0.0, 1.0), axis_tol_deg=20.0),
       Phase("pick", BACK, label="pick_back"), Phase("slide", BACK, SEAT, "insert_back"))
    + _screw_phases(NUTS[0], "nut1") + _screw_phases(NUTS[1], "nut2")
)

CHAIR = TaskSpec(
    name="chair",
    parts=(SEAT_SPEC, BACK_SPEC) + LEG_SPECS + NUT_SPECS,
    screws=(LEG_SCREW, NUT_SCREW),
    inserts=(BACK_INSERT,),
    targets=TARGETS,
    layout=LAYOUT,
    randomization=RANDOMIZATION,
    scene=SCENE,
    phases=PHASES,
    max_steps=6000,
    extras=dict(CORNER),
)
