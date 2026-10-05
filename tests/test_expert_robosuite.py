"""Scripted one_leg expert (Linux + robosuite): completes the task from a randomized layout, and
an episode recorded from it replays and merges like a VR demo."""
import os

import pytest

pytest.importorskip("robosuite")
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402

from fsim.collect import CollectConfig, Collector  # noqa: E402
from fsim.data.merge import merge  # noqa: E402
from fsim.data.replay import replay  # noqa: E402
from fsim.env.expert import OneLegExpert  # noqa: E402
from fsim.env.furniture_env import make_env  # noqa: E402


def test_expert_completes_one_leg_at_medium_randomization():
    env = make_env(task="one_leg", randomness="med", seed=3, has_offscreen_renderer=False, use_camera_obs=False)
    try:
        env.reset()
        res = OneLegExpert(env).run()
        phases, limit = len(env.task.phases), env.task.max_steps
    finally:
        env.close()
    assert res.success and res.failed is None, res
    assert res.phase == phases and res.steps < limit


@pytest.mark.slow
def test_expert_recording_replays_and_merges(tmp_path):
    env = make_env(task="one_leg", randomness="low", seed=5, camera_size=48)
    try:
        col = Collector(env, CollectConfig(out_dir=str(tmp_path), operator="scripted_expert", realtime=False),
                        get_controller=lambda: (None, 1e9))
        out = {}
        outcome, path = col.record_scripted(lambda step: out.update(res=OneLegExpert(env, step_fn=step).run()))
    finally:
        env.close()
    assert out["res"].success and outcome == "save" and os.path.isfile(path)
    with h5py.File(path, "r") as f:
        g = f["data/demo_0"]
        T = g.attrs["num_samples"]
        assert T == out["res"].steps + col.cfg.settle_capture
        assert g["actions"].shape == (T, 7) and g["obs/eye_in_hand_rgb"].shape == (T, 48, 48, 3)
        assert g.attrs["source"] == "scripted" and g.attrs["operator"] == "scripted_expert"
        assert float(g["rewards"][()].sum()) == 1.0 and g["dones"][-1] == 1
        assert list(g["phase"][()][[0, -1]]) == [0, 5] and int(g["skills"][()].sum()) == 5   # one flag per phase
    for mode in ("states", "actions"):           # the open-loop replay covers ~1100 contact-rich steps
        res = replay(path, mode=mode)
        assert res["success"] and res["replay_reward"] == 1.0, (mode, res)
    merged = str(tmp_path / "merged.hdf5")
    assert merge([path], merged, success_only=True) == 1
