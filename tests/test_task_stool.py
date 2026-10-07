"""stool: task-specific geometry and assembly checks (the generic ones are in test_tasks_generic.py)."""
import math

import numpy as np
import pytest

from fsim.geometry import pose_to_T
from fsim.models.parts import parts_aabb
from fsim.sim.screw_trials import Rig
from fsim.spec.stool import LEG_SPECS, LEGS, REL_POSES, SCREW, SEAT, SEAT_SPEC, STOOL
from task_checks import check_hand_drive

TASK = STOOL


def test_dimensions():
    lo, hi = parts_aabb(SEAT_SPEC.geoms)
    assert abs(hi[2] - lo[2] - 0.030) < 1e-4 and abs(hi[0] - lo[0] - 0.100) < 0.002
    leg = LEG_SPECS[0]
    lo, hi = parts_aabb(leg.geoms)
    assert abs(hi[2] - lo[2] - 0.060) < 1e-6 and abs(hi[1] - lo[1] - 0.035) < 1e-6
    # body + stud: 87.5 mm along the stud axis (stud 0.5 mm short of the design), 51 mm across the shear
    assert abs((hi[2] - lo[2]) + leg.studs[0].length - 0.0875) < 0.0006
    assert abs(hi[0] - lo[0] - 0.051) < 0.003


def test_sockets_at_120_degrees():
    xy = np.array([s.pos[:2] for s in SCREW.sites])
    assert np.allclose(np.linalg.norm(xy, axis=1), 0.0325)
    ang = sorted(math.degrees(math.atan2(y, x)) % 360 for x, y in xy)
    assert np.allclose(np.diff(ang), 120.0)


def test_rel_poses_match_design():
    """FurnitureBench: legs 40.54 mm out from the seat axis, 45 mm above it, leaning outward."""
    for (pos, quat), site in zip(REL_POSES, SCREW.sites):
        r = math.hypot(pos[0], pos[1])
        assert abs(r - 0.0405385) < 1e-6 and abs(pos[2] - 0.045 - SCREW.seat_gap) < 1e-9
        T = pose_to_T(pos, quat)
        top = (T @ [-0.030 * math.tan(math.radians(15)), 0, 0.030, 1])[:3]      # top-face centre
        assert math.hypot(*top[:2]) > r + 0.007                                  # leans outward


@pytest.mark.parametrize("site", [0, 1, 2])
def test_each_leg_screws_into_each_socket(site):
    check_hand_drive(TASK, SCREW, site=site, child=LEGS[site])


def test_legs_rest_flat():
    rig = Rig(TASK)
    rig.settle(1.0)
    for leg in LEGS:
        T = rig.state.pose_W(leg)
        assert T[2, 1] > 0.999 and abs(T[2, 3] - 0.0175) < 0.001          # lying on a flat side
    assert rig.state.pose_W(SEAT)[2, 2] > 0.999                             # seat holes up
