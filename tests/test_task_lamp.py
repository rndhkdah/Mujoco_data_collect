"""lamp: task-specific geometry and assembly checks (the generic ones are in test_tasks_generic.py)."""
import math

import mujoco
import numpy as np
import pytest

from fsim.geometry import inv_T, make_T, pose_to_T, rot_z
from fsim.sim.assembly import AssemblyChecker
from fsim.sim.screw import SEATED
from fsim.sim.screw_trials import Rig
from fsim.spec.lamp import (
    BASE, BASE_SPEC, BULB, BULB_SPEC, BULB_TILT, GLOBE_R, HOOD, HOOD_REL_Z, HOOD_SPEC, LAMP, PLATE_HALF, hood_rest_height,
)
from fsim.models.parts import parts_aabb

TASK = LAMP


def _assembled(rig) -> set:
    checker = AssemblyChecker(TASK)
    for _ in TASK.targets:
        checker.update(rig.state.poses())
    return checker.assembled


def test_dimensions():
    lo, hi = parts_aabb(BASE_SPEC.geoms)
    assert np.allclose(hi - lo, (0.0847, 0.0847, 0.063), atol=1e-4)
    lo, hi = parts_aabb(HOOD_SPEC.geoms)
    assert np.allclose(hi[:2] - lo[:2], 0.088, atol=0.002) and abs(hi[2] - lo[2] - 0.100) < 0.002
    assert 2 * PLATE_HALF < 0.085                      # top plate fits the Robotiq 2F-85
    # bulb: 121 mm from stud tip to globe top
    lo, hi = parts_aabb(BULB_SPEC.geoms)
    assert abs(hi[2] - (lo[2] - BULB_SPEC.studs[0].length) - 0.1205) < 1e-3


def test_hood_target_holds_at_every_yaw():
    lo, hi = hood_rest_height(False), hood_rest_height(True)
    assert lo < hi and abs(HOOD_REL_Z - lo) < 0.004 and abs(HOOD_REL_Z - hi) < 0.004


def test_bulb_rests_tilted():
    rig = Rig(TASK)
    rig.settle(1.5)
    z_axis = rig.state.pose_W(BULB)[:3, 2]
    tilt = math.degrees(math.asin(z_axis[2]))
    assert abs(tilt - math.degrees(BULB_TILT)) < 0.75, tilt
    assert np.linalg.norm(z_axis[:2]) > 0.95            # lying, globe end up


def _seat_bulb(rig):
    rig.screw.force_seat(BULB, BASE, 0)
    mujoco.mj_forward(rig.model, rig.data)


def _drop_hood(rig, yaw_deg: float, dz: float = 0.008, dxy=(0.0, 0.0)):
    T = rig.state.pose(BASE) @ make_T((dxy[0], dxy[1], HOOD_REL_Z + dz), rot_z(math.radians(yaw_deg)))
    rig.state.set_pose(HOOD, T)
    mujoco.mj_forward(rig.model, rig.data)
    rig.settle(1.5)


@pytest.mark.parametrize("yaw", [0.0, 3.75, 7.5, 33.0])
def test_hood_dropped_on_seated_bulb_is_assembled(yaw):
    rig = Rig(TASK)
    rig.settle(0.3)
    _seat_bulb(rig)
    rig.settle(0.3)
    assert rig.screw.child_state(BULB) == SEATED
    _drop_hood(rig, yaw)
    rel = inv_T(rig.state.pose(BASE)) @ rig.state.pose(HOOD)
    assert abs(rel[2, 3] - HOOD_REL_Z) < 0.004, rel[2, 3]
    assert rel[2, 2] > 0.995                             # upright, not jammed on the globe
    assert _assembled(rig) == {(BASE, BULB), (BASE, HOOD)}


def test_hood_clears_the_globe():
    """Seated bulb: the globe stays inside the hood's inner cone with clearance at the assembled pose."""
    rig = Rig(TASK)
    _seat_bulb(rig)
    T_bulb = inv_T(rig.state.pose(BASE)) @ rig.state.pose(BULB)
    c = T_bulb[:3, 3] + T_bulb[:3, 2] * (0.045 - GLOBE_R)      # globe centre, base frame
    for z_hood in (hood_rest_height(False), hood_rest_height(True)):
        zl = c[2] - z_hood                                       # globe centre height in the hood frame
        r_in = 0.040 - 0.14 * (zl + 0.050)
        assert r_in * math.cos(math.radians(8)) - GLOBE_R > 0.0003


def test_hood_without_bulb_is_not_assembled():
    rig = Rig(TASK)
    rig.settle(0.3)
    _drop_hood(rig, 0.0)
    rel = inv_T(rig.state.pose(BASE)) @ rig.state.pose(HOOD)
    assert abs(rel[2, 3] - HOOD_REL_Z) < 0.005                # it rests on the plate all the same
    assert _assembled(rig) == set()


def test_bulb_seated_pose_matches_design():
    rel = pose_to_T(*TASK.targets[0].rel_poses[0])
    # The design: bulb 67.8 mm below the base's mesh origin, turned 180 degrees; here 76.8 mm above our
    # base origin (the shoulder on the top plate) with the same yaw.
    assert abs(rel[2, 3] - 0.0768) < 1e-4 and np.allclose(rel[:3, 2], (0, 0, 1)) and rel[0, 0] < -0.999
