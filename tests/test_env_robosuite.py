"""robosuite layer (Linux only; skipped where robosuite cannot be imported, e.g. the Windows dev laptop)."""
import math
import os

import numpy as np
import pytest

pytest.importorskip("robosuite")
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from fsim.env.furniture_env import make_env  # noqa: E402
from fsim.env.ik import gripper_down, solve_ik  # noqa: E402
from fsim.env.scripted import ScriptedArm  # noqa: E402
from fsim.geometry import inv_T, make_T, rot_z  # noqa: E402
from fsim.sim.reset import max_penetration  # noqa: E402
from fsim.spec.square_table import LEGS, TOP  # noqa: E402

LEG = LEGS[3]


@pytest.fixture(scope="module")
def env():
    e = make_env(task="one_leg", randomness="none", seed=0, has_offscreen_renderer=False, use_camera_obs=False)
    yield e
    e.close()


@pytest.mark.parametrize("level", ["none", "low", "med", "high"])
def test_reset_places_parts_and_parks_arm(env, level):
    env.randomness = level
    obs = env.reset()
    m, d = env.sim.model._model, env.sim.data._data
    assert max_penetration(m, d, env.part_state.body_id.values()) < 0.0015
    for p in env.task.part_names:
        assert np.linalg.norm(d.qvel[env.part_state.dof_adr[p]:env.part_state.dof_adr[p] + 3]) < 0.05
    W = np.asarray(env.task.scene.W_origin)
    assert np.linalg.norm(obs["robot0_eef_pos"] - (W + np.array([0.15, 0.0, 0.17]))) < 0.01
    _, reward, _, info = env.step(np.zeros(7))
    assert reward == 0.0 and {"skill", "phase", "screw_events", "success"} <= set(info)
    env.randomness = "none"


def test_gripper_opening_is_capped(env):
    env.reset()
    for _ in range(10):
        env.step(np.r_[np.zeros(6), -1.0])
    m, d = env.sim.model._model, env.sim.data._data
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"gripper0_{s}_pad1") for s in ("left", "right")]
    gap = np.linalg.norm(d.geom_xpos[ids[0]] - d.geom_xpos[ids[1]]) - 2 * m.geom_size[ids[0]][1]
    assert 0.06 < gap < 0.067


def test_pick_and_lift_leg(env):
    env.reset()
    arm = ScriptedArm(env)
    leg = env.part_state.pose(LEG)
    assert arm.move_to(leg[:3, 3] + np.array([0, 0, 0.08]), gripper_down(), max_steps=150)
    assert arm.move_to(leg[:3, 3], gripper_down(), tol=0.002, max_steps=150)
    arm.grip(True, 20)
    assert env.grasped(LEG)
    arm.move_to(leg[:3, 3] + np.array([0, 0, 0.10]), gripper_down(), max_steps=150)
    assert env.part_state.pose(LEG)[2, 3] - leg[2, 3] > 0.07
    assert env.grasped(LEG)


def test_robot_screws_leg_in(env):
    """Leg upright in the gripper over hole 0: lower onto the catch, turn the wrist, get the reward."""
    env.reset()
    arm = ScriptedArm(env)
    s, r = env.task.screws[0], env.robots[0]
    hole = env.screw.holes[0]
    seat = env.part_state.pose(TOP) @ hole.T_local
    leg_T = seat @ make_T((0, 0, s.travel + 0.005), rot_z(math.radians(200)))
    eef_target = leg_T[:3, 3] + np.array([0, 0, 0.02])
    R0 = rot_z(math.radians(200)) @ gripper_down()
    ik = solve_ik(env.sim.model._model, env.sim.data._data.qpos, r.gripper.important_sites["grip_site"],
                  r._ref_joint_pos_indexes, r._ref_joint_vel_indexes, eef_target, R0,
                  q0=np.asarray(env.home_qpos), q_rest=np.asarray(env.home_qpos), iters=1500)
    assert ik.ok
    d = env.sim.data._data
    d.qpos[r._ref_joint_pos_indexes] = ik.q
    d.qvel[r._ref_joint_vel_indexes] = 0.0
    env.part_state.set_pose(LEG, leg_T)
    mujoco.mj_forward(env.sim.model._model, d)
    arm.grip(True, 20)
    assert env.grasped(LEG)
    axis_xy = seat[:2, 3]
    arm.move_to(np.array([*axis_xy, eef_target[2] - 0.007]), arm.eef_R, tol=0.0015, max_steps=60)
    # absolute target turning at 90 deg/s (like teleop: the target does not re-anchor to the arm)
    total, events, yaw = 0.0, [], 0.0
    for _ in range(300):
        yaw -= math.radians(90) / env.control_freq
        target = np.array([*axis_xy, arm.eef_pos[2] - 0.001])
        _, reward, _, info = arm.step(arm.action_toward(target, rot_z(yaw) @ R0))
        total += reward
        events += info["screw_events"]
        if env.screw.child_state(LEG) == "seated":
            break
    assert any(e["event"] == "engage" for e in events)
    assert env.screw.child_state(LEG) == "seated"
    arm.grip(False, 10)
    arm.hold(10)
    assert total == 1.0 and env._check_success()
    rel = inv_T(env.part_state.pose(TOP)) @ env.part_state.pose(LEG)
    assert np.allclose(rel[:3, 3], (0.05625, 0.05625, s.seated_z), atol=0.001)


def test_reset_after_assembly_restores_everything(env):
    env.reset()
    env.screw.force_seat(LEG, TOP, 2)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    env.step(np.zeros(7))
    assert env._check_success()
    env.reset()
    assert not env._check_success() and env.screw.child_state(LEG) == "free"
    m = env.sim.model._model
    assert all(m.geom_contype[h.catch_geom] == 1 for h in env.screw.holes)


def test_edit_model_xml_relocates_assets(env, tmp_path):
    xml = env.sim.model.get_xml()
    import re
    moved = re.sub(r'file="[^"]*?/(one_leg-[0-9a-f]{10})/', r'file="/elsewhere/gen/\1/', xml)
    moved = re.sub(r'file="[^"]*?/robosuite/', 'file="/old/site-packages/robosuite/', moved)
    moved = re.sub(r'file="[^"]*?/grippers/', 'file="/old/assets/grippers/', moved)
    fixed = env.edit_model_xml(moved)
    for path in re.findall(r'file="([^"]+)"', fixed):
        assert os.path.isfile(path), path


def test_part_xmls_can_be_written_by_several_envs_at_once():
    """Envs starting together (server + replay, parallel jobs) rewrite the same generated part XMLs."""
    import threading

    from fsim.env.objects import build_part_objects
    from fsim.spec import get_task

    task, errors = get_task("one_leg"), []

    def build():
        try:
            for _ in range(5):
                build_part_objects(task)
        except Exception as e:      # noqa: BLE001  (collected for the assert below)
            errors.append(e)

    threads = [threading.Thread(target=build) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
