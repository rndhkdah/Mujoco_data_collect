#!/usr/bin/env python3
"""Replay a recorded demo (Linux, robosuite).

    MUJOCO_GL=egl python -m fsim.data.replay <file.hdf5> [--demo demo_0] [--mode states|actions]

states:  set every recorded MuJoCo state (states_full) and re-check assembly -> reproduces exactly
actions: start from the first state and re-run the recorded actions open loop -> checks determinism
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


def make_env_from_file(path: str, demo: str = "demo_0", cameras: bool = False):
    import robosuite

    import fsim.env.furniture_env  # noqa: F401  (registers KinovaFurnitureEnv)

    with h5py.File(path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        g = f["data"][demo]
        model_xml = g.attrs["model_file"]
        start = json.loads(g.attrs["start_snapshot"])
    kwargs = dict(env_args["env_kwargs"])
    if not cameras:
        kwargs.update(has_offscreen_renderer=False, use_camera_obs=False)
    env = robosuite.make(env_args["env_name"], **kwargs)
    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    return env, start


def _set_state(env, state):
    m, d = env.sim.model._model, env.sim.data._data
    mujoco.mj_setState(m, d, np.asarray(state, float), int(mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_forward(m, d)


def replay(path: str, demo: str = "demo_0", mode: str = "states") -> dict:
    env, start = make_env_from_file(path, demo)
    with h5py.File(path, "r") as f:
        g = f["data"][demo]
        states = g["states_full"][()]
        actions = g["actions"][()]
        rewards = g["rewards"][()]
    _set_state(env, states[0])
    env.restore_snapshot(start)
    total = 0.0
    if mode == "states":
        for s in states[1:]:
            _set_state(env, s)
            env.screw._sync_from_eq_active()
            total += env.checker.update(env.part_state.poses())
    else:
        for a in actions:
            _, r, _, _ = env.step(a)
            total += r
    out = {"mode": mode, "steps": len(actions), "recorded_reward": float(rewards.sum()), "replay_reward": float(total),
           "success": bool(env.checker.all_assembled)}
    env.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--mode", choices=["states", "actions"], default="states")
    args = ap.parse_args(argv)
    print(json.dumps(replay(args.path, args.demo, args.mode)), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
