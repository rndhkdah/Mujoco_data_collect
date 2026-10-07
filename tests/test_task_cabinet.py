"""cabinet: task-specific checks (the generic suite in test_tasks_generic.py covers the rest)."""
import math

import mujoco
import numpy as np
import pytest

from fsim.geometry import inv_T, make_T, pose_to_T, quat_to_mat, rot_y
from fsim.sim.assembly import AssemblyChecker
from fsim.sim.screw import SEATED
from fsim.sim.screw_trials import Rig, drive_site
from fsim.spec.cabinet import BODY, CABINET, DOOR_L, DOOR_R, INSERT_L, INSERT_R, SCREW, TARGETS, TOP
from task_checks import check_hand_drive

TASK = CABINET
W = np.asarray(TASK.scene.W_origin)
LYING = quat_to_mat(TASK.placement(BODY).quat)            # on its back, top end toward the robot


@pytest.mark.parametrize("spec", [INSERT_R, INSERT_L], ids=["right", "left"])
def test_each_door_slides_onto_its_pole(spec):
    """Hand-driven from 10 mm beyond the capture point (the clip 16 mm short of the pole tip)."""
    check_hand_drive(TASK, spec)


@pytest.mark.parametrize("spec", [INSERT_R, INSERT_L], ids=["right", "left"])
def test_door_slides_on_with_the_body_lying(spec):
    """The task's actual configuration: body on its back, pole axis horizontal (+W x)."""
    rig = Rig(TASK)
    r = drive_site(rig, spec, parent_T=make_T(W + (0.1, 0.0, 0.3), LYING))
    assert r.engaged and r.seated and r.assembled


@pytest.mark.parametrize("site", [0, 1], ids=["0deg", "180deg"])
def test_top_screws_on_in_both_placements(site):
    check_hand_drive(TASK, SCREW, site)


def test_seated_poses_match_the_design():
    """Our seats vs the design's (FurnitureBench) rel poses converted to our frames: right door within 1 mm,
    top within the 0.3 mm seat gap. The design's left door is 4.75 mm off its pole; ours mirrors the right."""
    rel = {t.child: t.rel_poses[0][0] for t in TARGETS}
    assert np.abs(np.subtract(rel[DOOR_R], (-0.02975, -0.0280, 0.005625))).max() < 0.001
    assert np.allclose(rel[DOOR_L], np.multiply(rel[DOOR_R], (1, -1, 1)))
    assert abs(rel[TOP][2] - 0.0825) < 0.0005 and np.abs(rel[TOP][:2] - np.array([0.0005, 0.0])).max() < 1e-6


def _checker_after(rig, n=4):
    checker = AssemblyChecker(TASK)
    for _ in range(n):
        checker.update(rig.state.poses())
    return checker.assembled


def test_top_needs_both_doors():
    rig = Rig(TASK)
    rig.hold(BODY, make_T(W + (0.1, 0.0, 0.3)))                # upright
    rig.screw.force_seat(TOP, BODY, 0)
    mujoco.mj_forward(rig.model, rig.data)
    assert rig.screw.child_state(TOP) == SEATED
    assert (BODY, TOP) not in _checker_after(rig)
    rig.inserts.force_seat(DOOR_R, BODY, 0)                    # one door only (the design's slip let this count)
    mujoco.mj_forward(rig.model, rig.data)
    assert (BODY, TOP) not in _checker_after(rig)
    rig.inserts.force_seat(DOOR_L, BODY, 1)
    mujoco.mj_forward(rig.model, rig.data)
    assert {(BODY, DOOR_R), (BODY, DOOR_L), (BODY, TOP)} <= _checker_after(rig)


def test_doors_stay_locked_when_the_body_is_tipped_upright():
    rig = Rig(TASK)
    p = W + (0.1, 0.0, 0.3)
    rig.hold(BODY, make_T(p, LYING))
    rig.inserts.force_seat(DOOR_R, BODY, 0)
    rig.inserts.force_seat(DOOR_L, BODY, 1)
    mujoco.mj_forward(rig.model, rig.data)
    rig.settle(0.3)
    # Tip the top end up about W y (90 degrees in 2 s), as when lifting it and pivoting on the bottom end.
    n = 40
    for k in range(1, n + 1):
        rig.hold(BODY, make_T(p, rot_y(math.pi / 2 * k / n) @ LYING))
        rig.hand.step()
    rig.settle(0.5)
    R = rig.state.pose(BODY)[:3, :3]
    assert R[2, 2] > math.cos(math.radians(2)), "body not upright"
    for door in (DOOR_R, DOOR_L):
        assert rig.inserts.child_state(door) == SEATED
    target = {t.child: t for t in TARGETS}
    for door in (DOOR_R, DOOR_L):
        rel = inv_T(rig.state.pose(BODY)) @ rig.state.pose(door)
        ref = pose_to_T(*target[door].rel_poses[0])
        assert np.abs(rel[:3, 3] - ref[:3, 3]).max() < 5e-4
        assert np.diag(rel[:3, :3] @ ref[:3, :3].T).min() > math.cos(math.radians(1))
    assert {(BODY, DOOR_R), (BODY, DOOR_L)} <= _checker_after(rig)


def test_upright_phase_fires_after_the_flip():
    from fsim.sim.phases import PhaseTracker
    ph = next(p for p in TASK.phases if p.name == "tip_body_upright")
    tr = PhaseTracker(TASK)
    T = make_T(W + (0.34, 0.1, 0.0675), np.eye(3))
    assert tr._check(ph, {BODY: T}, None, set(), None)
    assert not tr._check(ph, {BODY: make_T(W, LYING)}, None, set(), None)
