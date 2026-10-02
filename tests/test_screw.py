"""Helical screw manager: gating logic (deterministic) and full scripted trials (physics)."""
import math

import mujoco
import numpy as np
import pytest

from fsim.geometry import make_T, rot_x, rot_z
from fsim.models import names
from fsim.sim.assembly import AssemblyChecker
from fsim.sim.screw import ENGAGED, FREE, SEATED, HelicalScrewManager
from fsim.sim.screw_trials import Rig, TrialParams, run_trial
from fsim.spec import get_task
from fsim.spec.square_table import LEGS, TOP

TASK = get_task("one_leg")
LEG = LEGS[3]


def place_at_catch(rig, hole_idx=0, twist_deg=180.0, radial=0.0, tilt_deg=0.0, axial_extra=0.0):
    hole = rig.screw.holes[hole_idx]
    seat = rig.state.pose(TOP) @ hole.T_local
    s = TASK.screws[0]
    T = seat @ make_T((radial, 0.0, s.travel + axial_extra), rot_z(math.radians(twist_deg)) @ rot_x(math.radians(tilt_deg)))
    rig.state.set_pose(LEG, T)
    mujoco.mj_forward(rig.model, rig.data)
    return hole


def sweep_twist(rig, degs, **kw):
    """Feed a sequence of twists (deg, relative to seated) to the FSM without stepping physics."""
    for a in degs:
        place_at_catch(rig, twist_deg=a, **kw)
        rig.screw.update()
    return [e for e in rig.screw.events if e["event"] == "engage"]


@pytest.fixture
def rig():
    r = Rig(TASK)
    mujoco.mj_forward(r.model, r.data)
    return r


def test_engages_when_tightening_through_thread_start(rig):
    ev = sweep_twist(rig, [190, 186, 182, 179])
    assert len(ev) == 1 and ev[0]["child"] == LEG
    hole = rig.screw.holes[0]
    assert rig.screw.hole_state(hole) == ENGAGED
    assert math.isclose(rig.data.qpos[hole.hinge_q], 3 * math.pi + math.radians(-1), abs_tol=1e-9)
    assert rig.model.geom_contype[hole.catch_geom] == 0 and rig.data.eq_active[hole.weld[LEG]] == 1


def test_needs_arming_and_crossing(rig):
    assert sweep_twist(rig, [179, 178, 175]) == []           # never seen before the thread start
    assert sweep_twist(rig, [170, 175, 182, 189]) == []      # untightening direction


@pytest.mark.parametrize("kw", [dict(radial=0.003), dict(tilt_deg=10.0), dict(axial_extra=0.004), dict(axial_extra=-0.004)])
def test_no_engagement_outside_catch_zone(rig, kw):
    assert sweep_twist(rig, [190, 186, 182, 179], **kw) == []


def test_resting_hole_reports_aligned_leg_on_catch(rig):
    place_at_catch(rig, hole_idx=2, twist_deg=40.0)
    assert rig.screw.resting_hole(LEG).index == 2
    place_at_catch(rig, hole_idx=2, twist_deg=40.0, radial=0.004)
    assert rig.screw.resting_hole(LEG) is None


def test_force_seat_is_assembled_and_stable(rig):
    rig.screw.force_seat(LEG, TOP, 3)
    mujoco.mj_forward(rig.model, rig.data)
    rig.settle(1.0)
    assert rig.screw.child_state(LEG) == SEATED
    assert AssemblyChecker(TASK).update(rig.state.poses()) == 1


def test_force_seat_preconditions(rig):
    rig.screw.force_seat(LEG, TOP, 0)
    with pytest.raises(ValueError):
        rig.screw.force_seat(LEGS[0], TOP, 0)        # hole occupied
    with pytest.raises(ValueError):
        rig.screw.force_seat(LEG, TOP, 1)            # leg already engaged


def test_force_seat_after_data_reset_without_forward(rig):
    """force_seat must not use stale/zeroed kinematics (reset flow: resetData -> set qpos -> seat)."""
    mujoco.mj_resetData(rig.model, rig.data)
    rig.screw.reset()
    T = rig.state.pose  # noqa: F841  (kinematics are zeroed here on purpose)
    W = np.asarray(TASK.scene.W_origin)
    rig.state.set_pose(TOP, make_T(W + np.array([0.25, 0.05, 0.015625]), rot_z(0.3)))
    rig.screw.force_seat(LEG, TOP, 0)
    mujoco.mj_forward(rig.model, rig.data)
    assert AssemblyChecker(TASK).update(rig.state.poses()) == 1
    rig.settle(0.5)
    assert AssemblyChecker(TASK).update(rig.state.poses()) == 1


def test_catches_reenabled_by_new_manager_after_data_reset(rig):
    rig.screw.force_seat(LEG, TOP, 0)
    hole = rig.screw.holes[0]
    assert rig.model.geom_contype[hole.catch_geom] == 0
    mujoco.mj_resetData(rig.model, rig.data)
    fresh = HelicalScrewManager(rig.model, rig.data, TASK)
    fresh.reset()
    assert rig.model.geom_contype[hole.catch_geom] == 1 and rig.model.geom_conaffinity[hole.catch_geom] == 1


def test_stale_manager_follows_eq_active_after_data_reset(rig):
    rig.screw.force_seat(LEG, TOP, 0)
    mujoco.mj_resetData(rig.model, rig.data)            # clears eq_active; manager not reset
    mujoco.mj_forward(rig.model, rig.data)
    assert rig.screw.child_state(LEG) == FREE
    assert rig.model.geom_contype[rig.screw.holes[0].catch_geom] == 1


def test_snapshot_restore_roundtrip(rig):
    rig.screw.force_seat(LEG, TOP, 1)
    snap = rig.screw.snapshot()
    rig.screw.reset()
    assert all(v is None for v in rig.screw.snapshot()["children"].values())
    rig.screw.restore(snap)
    hole = rig.screw.holes[1]
    assert hole.child == LEG and rig.data.eq_active[hole.weld[LEG]] == 1
    assert rig.model.geom_contype[hole.catch_geom] == 0


def test_snapshot_keeps_arming_state():
    a, b = Rig(TASK), Rig(TASK)
    for r in (a, b):
        mujoco.mj_forward(r.model, r.data)
    straight = sweep_twist(a, [190, 186, 182, 179, 176])
    sweep_twist(b, [190, 186, 182])
    snap = b.screw.snapshot()
    b.screw.restore(snap)
    resumed = sweep_twist(b, [179, 176])
    assert len(straight) == 1 and len(resumed) == 1


def test_reset_releases_everything(rig):
    rig.screw.force_seat(LEG, TOP, 0)
    rig.screw.reset()
    hole = rig.screw.holes[0]
    assert rig.screw.hole_state(hole) == FREE
    assert rig.data.eq_active[hole.weld[LEG]] == 0 and rig.model.geom_contype[hole.catch_geom] == 1


def _seat_at(rig, hinge):
    rig.screw.force_seat(LEG, TOP, 0)
    h, s = rig.screw.holes[0], TASK.screws[0]
    rig.data.qpos[h.hinge_q] = hinge
    rig.data.qpos[h.slide_q] = hinge / s.rad_per_m
    mujoco.mj_forward(rig.model, rig.data)
    car = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, names.carriage(TOP, 0))
    rig.state.set_pose(LEG, make_T(rig.data.xpos[car], rig.data.xmat[car].reshape(3, 3)))
    mujoco.mj_forward(rig.model, rig.data)
    return h


def _pull(rig, force, seconds=3.0):
    rig.data.xfrc_applied[rig.state.body_id[LEG]] = [0, 0, force, 0, 0, 0]
    rig.data.xfrc_applied[rig.state.body_id[TOP]] = [0, 0, -force, 0, 0, 0]
    for _ in range(int(seconds * TASK.control_hz)):
        rig.hand.step()
        if any(e["event"] == "disengage" for e in rig.screw.events):
            return True
    return False


def test_thread_holds_moderate_pull_and_releases_only_at_thread_start(rig):
    h = _seat_at(rig, 1.5 * math.pi)
    assert not _pull(rig, 5.0)                                        # mid-thread: held
    assert abs(rig.screw.hinge(h) - 1.5 * math.pi) < math.radians(10)


def test_pull_at_thread_start_releases():
    hold, pop = Rig(TASK), Rig(TASK)
    _seat_at(hold, 3 * math.pi - math.radians(1))
    assert not _pull(hold, 2.0)
    _seat_at(pop, 3 * math.pi - math.radians(1))
    assert _pull(pop, 6.0)


def test_seated_leg_hangs_upside_down_without_unscrewing(rig):
    rig.screw.force_seat(LEG, TOP, 0)
    mujoco.mj_forward(rig.model, rig.data)
    T = rig.state.pose(TOP)
    rig.hand.grasp(TOP, T)
    up = make_T(T[:3, 3] + np.array([0, 0, 0.25]), T[:3, :3])
    rig.hand.move_to(up, 1.0)
    rig.hand.move_to(make_T(up[:3, 3], rot_x(math.pi) @ T[:3, :3]), 1.0)
    h0 = rig.screw.hinge(rig.screw.holes[0])
    rig.hand.step(int(10 * TASK.control_hz))
    assert rig.screw.hinge(rig.screw.holes[0]) - h0 < math.radians(3)
    assert AssemblyChecker(TASK).update(rig.state.poses()) == 1


@pytest.mark.parametrize("p", [
    TrialParams(),
    TrialParams(hole=2, twist0_deg=-120, lateral_mm=1.0, tilt_deg=2.0, rate_deg_s=90),
    TrialParams(hole=3, twist0_deg=170, lateral_mm=1.5, lateral_dir_deg=200, tilt_deg=3.0, rate_deg_s=180, press_mm=2.0),
])
def test_scripted_screw_in_and_out(p):
    r = run_trial(TASK, p)
    assert r.engaged and r.seated and r.assembled
    assert r.seat_pos_err_mm < 1.0 and r.seat_cos_min > 0.999
    assert r.released_at_start and r.lifted_out
    assert r.rtf > 1.0


@pytest.mark.parametrize("p", [
    TrialParams(lateral_mm=20.0, unscrew=False, screw_timeout_s=5),
    TrialParams(tilt_deg=15.0, unscrew=False, screw_timeout_s=5),
    TrialParams(tighten=False, unscrew=False, screw_timeout_s=6),
])
def test_scripted_negative_cases_never_engage(p):
    assert not run_trial(TASK, p).engaged
