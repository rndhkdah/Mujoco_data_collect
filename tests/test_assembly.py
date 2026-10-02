import math

import numpy as np

from fsim.geometry import make_T, rot_x, rot_z
from fsim.sim.assembly import AssemblyChecker, matches
from fsim.spec import get_task
from fsim.spec.square_table import LEGS, TOP

TASK = get_task("one_leg")
TARGET = TASK.targets[0]
SEAT = 0.046875


def rel(dx=0.0, dy=0.0, dz=0.0, R=np.eye(3), corner=(1, 1)):
    return make_T((corner[0] * 0.05625 + dx, corner[1] * 0.05625 + dy, SEAT + dz), R)


def test_all_corners_match():
    for c in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        assert matches(rel(corner=c), TARGET)


def test_position_tolerance_per_axis():
    assert matches(rel(dx=0.0045), TARGET)
    assert not matches(rel(dx=0.006), TARGET)
    assert not matches(rel(dz=-0.006), TARGET)
    assert matches(rel(dy=-0.0049, dz=0.0049), TARGET)


def test_orientation_tolerance_includes_screw_angle():
    assert matches(rel(R=rot_z(math.radians(15))), TARGET)
    assert not matches(rel(R=rot_z(math.radians(25))), TARGET)
    assert not matches(rel(R=rot_z(math.radians(90))), TARGET)      # square leg, but angle must match
    assert not matches(rel(R=rot_x(math.radians(25))), TARGET)


def test_checker_latches_and_counts_new_pairs():
    chk = AssemblyChecker(TASK)
    top = make_T((0.3, 0.0, 0.43))
    poses = {TOP: top, LEGS[3]: top @ rel()}
    assert chk.update(poses) == 1
    assert chk.all_assembled
    poses[LEGS[3]] = make_T((0.0, 0.0, 1.0))          # pulled away: stays latched, no new reward
    assert chk.update(poses) == 0 and chk.all_assembled
    chk.reset()
    assert not chk.assembled


def test_one_leg_ignores_other_legs():
    chk = AssemblyChecker(TASK)
    top = make_T((0.3, 0.0, 0.43))
    poses = {TOP: top, LEGS[0]: top @ rel(), LEGS[3]: make_T((0.0, 0.3, 0.43))}
    assert chk.update(poses) == 0
