"""End-to-end recording through the teleop mapper with a scripted virtual operator (Linux only)."""
import math
import os

import numpy as np
import pytest

pytest.importorskip("robosuite")
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import mujoco  # noqa: E402

from fsim.collect import CollectConfig, Collector  # noqa: E402
from fsim.data.merge import merge  # noqa: E402
from fsim.data.replay import replay  # noqa: E402
from fsim.env.furniture_env import make_env  # noqa: E402
from fsim.env.ik import gripper_down, solve_ik  # noqa: E402
from fsim.geometry import make_T, rot_z  # noqa: E402
from fsim.spec.square_table import LEGS, TOP  # noqa: E402

LEG = LEGS[3]


def stage_leg_in_gripper(env):
    """Test setup: arm holding position over hole 0 with leg4 upright between the open fingers."""
    s, r = env.task.screws[0], env.robots[0]
    seat = env.part_state.pose(TOP) @ env.screw.holes[0].T_local
    leg_T = seat @ make_T((0, 0, s.travel + 0.005), rot_z(math.radians(200)))
    ik = solve_ik(env.sim.model._model, env.sim.data._data.qpos, r.gripper.important_sites["grip_site"],
                  r._ref_joint_pos_indexes, r._ref_joint_vel_indexes, leg_T[:3, 3] + [0, 0, 0.02],
                  rot_z(math.radians(200)) @ gripper_down(), q0=np.asarray(env.home_qpos),
                  q_rest=np.asarray(env.home_qpos), iters=1500)
    d = env.sim.data._data
    d.qpos[r._ref_joint_pos_indexes] = ik.q
    d.qvel[r._ref_joint_vel_indexes] = 0.0
    env.part_state.set_pose(LEG, leg_T)
    mujoco.mj_forward(env.sim.model._model, d)


class VirtualOperator:
    """Produces controller messages: clutch + close, push down, twist until seated, release."""

    def __init__(self, env):
        self.env, self.calls, self.staged, self.step = env, 0, False, 0

    def msg(self, grip=0.0, trigger=0.0, stick=(0.0, 0.0), b=False):
        return {"type": "controller", "frame": "robot",
                "right": {"pos": [0, 0, 0], "quat": [0, 0, 0, 1], "tracked": True, "grip": grip, "trigger": trigger,
                          "a": False, "b": b, "stick": list(stick)}, "left": {}}

    def __call__(self):
        self.calls += 1
        if not self.staged:                       # first call after reset: stage the scene, start recording
            stage_leg_in_gripper(self.env)
            self.staged = True
            return self.msg(grip=1.0), 0.0
        self.step += 1
        seated = self.env.screw.child_state(LEG) == "seated"
        if self.step < 15:
            return self.msg(grip=1.0, trigger=1.0), 0.0              # hold still, close the gripper
        if self.step < 35:
            return self.msg(trigger=1.0, stick=(0.0, -1.0)), 0.0      # push down along the tool axis
        if not seated and self.step < 600:
            return self.msg(trigger=1.0, stick=(1.0, -0.3)), 0.0      # twist (tighten) while pressing lightly
        return self.msg(trigger=0.0), 0.0                             # release


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    out = tmp_path_factory.mktemp("demos")
    env = make_env(task="one_leg", randomness="none", seed=0, camera_size=64)
    op = VirtualOperator(env)
    col = Collector(env, CollectConfig(out_dir=str(out), operator="test", realtime=False, max_steps=800), op)
    obs = col.wait_for_start()
    outcome, path = col.record_episode(obs)
    env.close()
    return outcome, path, out


def test_virtual_operator_episode_is_saved(recorded):
    outcome, path, _ = recorded
    assert outcome == "save" and path and os.path.isfile(path)
    with h5py.File(path, "r") as f:
        g = f["data/demo_0"]
        T = g.attrs["num_samples"]
        assert g["actions"].shape == (T, 7) and g["actions_fb"].shape == (T, 8)
        assert g["obs/agentview_rgb"].shape == (T, 64, 64, 3) and g["obs/eye_in_hand_rgb"].shape == (T, 64, 64, 3)
        assert g["obs/parts_poses"].shape == (T, 35) and g["controller"].shape == (T, 18)
        assert float(g["rewards"][()].sum()) == 1.0 and g["dones"][-1] == 1
        assert bool(g.attrs["success"]) and g.attrs["task"] == "one_leg"
        assert "engage" in g.attrs["screw_events"]
        assert g["states_full"].shape[0] == T


def test_replay_reproduces_success(recorded):
    _, path, _ = recorded
    res = replay(path, mode="states")
    assert res["success"] and res["replay_reward"] == 1.0
    res = replay(path, mode="actions")
    assert res["success"]


def test_merge_builds_robomimic_file(recorded, tmp_path):
    _, path, out = recorded
    merged = str(tmp_path / "merged.hdf5")
    assert merge([path, path], merged) == 2
    with h5py.File(merged, "r") as f:
        assert sorted(f["data"].keys()) == ["demo_0", "demo_1"]
        assert [x.decode() for x in f["mask/none"][()]] == ["demo_0", "demo_1"]


def test_scripted_episode_outcomes_without_saving(tmp_path):
    """record_scripted(save=False) is the server's autopilot: it runs and publishes, and writes nothing."""
    env = make_env(task="one_leg", randomness="none", seed=0, has_offscreen_renderer=False, use_camera_obs=False)
    try:
        seen = []
        col = Collector(env, CollectConfig(out_dir=str(tmp_path), realtime=False, max_steps=5), lambda: (None, 1e9),
                        publish=lambda state, obs, info: seen.append((state, info.get("steps"))))

        def hold(n):
            def script(step):
                pos, R = col._ee()
                for _ in range(n):
                    step(np.zeros(7), pos, R)
            return script

        assert col.record_scripted(hold(3), save=False) == ("failed", None)      # ran, but the task is not done
        assert [s for s in seen if s[0] == "RECORD"][-1] == ("RECORD", 3)
        assert col.record_scripted(hold(9), save=False) == ("timeout", None)     # step limit 5
        col.running = False
        assert col.record_scripted(hold(3), save=False) == ("stopped", None)
        assert not os.listdir(tmp_path) and col.saved == 0
    finally:
        env.close()
