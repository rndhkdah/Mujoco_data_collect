"""A test-only task (not registered) exercising every generic mechanism:

  base   PrimitivePartSpec plate with a 25-degree tilted threaded socket, a threaded stud and a pole
  peg    screws into the tilted socket (child stud, 2 features -> child_feature)
  knob   screws into a socket on top of the peg (chained: peg is a screw child and a screw parent)
  nut    screws onto the base stud (inverted: parent stud, child socket, catch collar)
  sleeve slides down the pole (InsertSpec)
  cap    position-only placement on the base
Threads follow FurnitureBench: 9.375 mm pitch, right-handed.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from fsim.geometry import mat_to_quat, rot_x, rot_y
from fsim.models.parts import parts_aabb
from fsim.spec.common import LEVELS, RESET_OBSTACLES, SCENE, corner_extras, ring_geoms, seated_rel_pose
from fsim.spec.types import (
    IDENTITY_QUAT, AssemblyTarget, Geom, InsertSpec, Phase, Placement, PrimitivePartSpec, RandomizationSpec,
    ScrewSpec, Socket, Stud, TaskSpec, ThreadSpec, site_at,
)

PITCH = 0.009375
DOWN = tuple(mat_to_quat(rot_x(math.pi)))           # feature z along the part's -z
TILT = tuple(mat_to_quat(rot_x(math.radians(25))))
LYING = tuple(mat_to_quat(rot_y(math.pi / 2)))
MALE = ThreadSpec(PITCH, 0.0042, 0.0058)
FEMALE = ThreadSpec(PITCH, 0.0060, 0.0072)


def _recentered(spec: PrimitivePartSpec) -> PrimitivePartSpec:
    """Shift geoms and features so the geoms' bounding box is centred on the origin."""
    lo, hi = parts_aabb(spec.geoms)
    c = (lo + hi) / 2

    def sh(x):
        return replace(x, pos=tuple(float(v) for v in np.asarray(x.pos) - c))

    return replace(spec, geoms=tuple(map(sh, spec.geoms)), studs=tuple(map(sh, spec.studs)),
                   sockets=tuple(map(sh, spec.sockets)))


PLATE_HALF_Z = 0.006
BASE = _recentered(PrimitivePartSpec(
    name="base", mass=0.2,
    geoms=(Geom("box", (0.07, 0.07, PLATE_HALF_Z)),)
    + ring_geoms((-0.035, 0.0, 0.035), TILT, 0.0075, 0.013, -0.025, 0.0),
    studs=(Stud((0.035, 0.035, PLATE_HALF_Z), IDENTITY_QUAT, 0.005, 0.03, MALE, "nut_stud"),
           Stud((0.035, -0.035, PLATE_HALF_Z), IDENTITY_QUAT, 0.005, 0.04, None, "pole")),
    sockets=(Socket((-0.035, 0.0, 0.035), TILT, 0.025, 0.0075, thread=FEMALE, name="tilted"),),
    rgba=(0.6, 0.65, 0.75, 1.0),
))
PEG = PrimitivePartSpec(
    name="peg", mass=0.03,
    geoms=(Geom("box", (0.012, 0.012, 0.0125), (0.0, 0.0, -0.0125)),)
    + ring_geoms((0.0, 0.0, 0.025), IDENTITY_QUAT, 0.0075, 0.012, -0.025, 0.0),
    studs=(Stud((0.0, 0.0, -0.025), DOWN, 0.0055, 0.02, MALE, "stud"),),
    sockets=(Socket((0.0, 0.0, 0.025), IDENTITY_QUAT, 0.025, 0.0075, thread=FEMALE, name="top"),),
)
KNOB = PrimitivePartSpec(name="knob", mass=0.01, geoms=(Geom("cylinder", (0.012, 0.01)),),
                         studs=(Stud((0.0, 0.0, -0.01), DOWN, 0.0055, 0.02, MALE, "stud"),))
NUT = PrimitivePartSpec(name="nut", mass=0.01, geoms=ring_geoms((0, 0, 0), IDENTITY_QUAT, 0.0066, 0.012, -0.006, 0.006),
                        sockets=(Socket((0.0, 0.0, -0.006), DOWN, 0.012, 0.0066, thread=FEMALE),), rgba=(0.5, 0.5, 0.5, 1))
SLEEVE = PrimitivePartSpec(name="sleeve", mass=0.01,
                           geoms=ring_geoms((0, 0, 0), IDENTITY_QUAT, 0.0065, 0.011, -0.015, 0.015),
                           sockets=(Socket((0.0, 0.0, -0.015), DOWN, 0.03, 0.0065),), visual="primitive")
CAP = PrimitivePartSpec(name="cap", mass=0.01, geoms=(Geom("capsule", (0.006, 0.01), quat=LYING),))

PEG_SCREW = ScrewSpec("base", ("peg",), PITCH, sites=(site_at(BASE.feature("tilted")),), child_feature="stud",
                      catch_radius=0.006)
KNOB_SCREW = ScrewSpec("peg", ("knob",), PITCH, sites=(site_at(PEG.feature("top")),), catch_radius=0.006)
NUT_SCREW = ScrewSpec("base", ("nut",), PITCH, sites=(site_at(BASE.feature("nut_stud")),), catch_radius=0.009)
SLEEVE_INSERT = InsertSpec("base", ("sleeve",), sites=(site_at(BASE.feature("pole")),), travel=0.035)

_PLATE_TOP = BASE.geoms[0].pos[2] + PLATE_HALF_Z
TARGETS = (
    AssemblyTarget("base", "peg", (seated_rel_pose(PEG_SCREW, 0, PEG),)),
    AssemblyTarget("peg", "knob", (seated_rel_pose(KNOB_SCREW, 0, KNOB),), requires=(("base", "peg"),)),
    AssemblyTarget("base", "nut", (seated_rel_pose(NUT_SCREW, 0, NUT),)),
    AssemblyTarget("base", "sleeve", (seated_rel_pose(SLEEVE_INSERT, 0, SLEEVE),)),
    AssemblyTarget("base", "cap", (((-0.02, 0.05, _PLATE_TOP + 0.006), IDENTITY_QUAT),), pos_tol=(0.01, 0.01, 0.01),
                   position_only=True),
)

_BASE_Z = -float(parts_aabb(BASE.geoms)[0][2])
LAYOUT = (
    Placement("base", (0.22, 0.0, _BASE_Z), IDENTITY_QUAT),
    Placement("peg", (0.0, -0.17, 0.012), LYING),
    Placement("nut", (0.0, -0.08, 0.006), IDENTITY_QUAT),
    Placement("sleeve", (0.0, 0.0, 0.015), IDENTITY_QUAT),
    Placement("knob", (0.0, 0.08, 0.012), LYING),
    Placement("cap", (0.0, 0.17, 0.006), IDENTITY_QUAT),
)


def _screw_phases(part, target):
    return (Phase("pick", part), Phase("insert", part, target), Phase("screw", part, target))


SYNTHETIC = TaskSpec(
    name="synthetic",
    parts=(BASE, PEG, KNOB, NUT, SLEEVE, CAP),
    screws=(PEG_SCREW, KNOB_SCREW, NUT_SCREW),
    inserts=(SLEEVE_INSERT,),
    targets=TARGETS,
    layout=LAYOUT,
    randomization=RandomizationSpec(
        levels=LEVELS, bounds_x=(-0.06, 0.3715), bounds_y=(-0.21, 0.21),
        footprints={"base": (0.14, 0.14), "peg": (0.07, 0.025), "nut": (0.025, 0.025), "sleeve": (0.025, 0.025),
                    "knob": (0.045, 0.025), "cap": (0.035, 0.015)},
        reset_obstacles=RESET_OBSTACLES, part_pos_range={"med": {"base": ((-0.05, 0.02), (-0.03, 0.03))}}),
    scene=SCENE,
    phases=(Phase("grasp", "base"), Phase("place_corner", "base", yaw_sym_deg=90.0))
    + _screw_phases("peg", "base") + _screw_phases("knob", "peg") + _screw_phases("nut", "base")
    + (Phase("pick", "sleeve"), Phase("insert", "sleeve", "base"), Phase("slide", "sleeve", "base"),
       Phase("pick", "cap"), Phase("place", "cap", "base"),
       Phase("place_pose", "peg", pos=None, axis=(0.0, 0.0, 1.0), axis_dir=(0.0, -0.42, 0.91), label="peg_tilted")),
    extras=corner_extras(0.14, 0.14),
)
