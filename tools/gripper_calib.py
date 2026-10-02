#!/usr/bin/env python3
"""Measure the Robotiq 2F-85 opening (gap between pad surfaces) as a function of the gripper action.

    MUJOCO_GL=egl python tools/gripper_calib.py

Prints the action that opens the gripper to the design's 65 mm, for FsimRobotiq2f85.OPEN_LIMIT.
"""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsim.env.furniture_env import make_env  # noqa: E402

TARGET_GAP = 0.065


def pad_gap(env) -> float:
    m, d = env.sim.model._model, env.sim.data._data
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"gripper0_{s}_pad1") for s in ("left", "right")]
    sid = env.robots[0].eef_site_id
    R = d.site_xmat[sid].reshape(3, 3)
    x = [float((R.T @ (d.geom_xpos[g] - d.site_xpos[sid]))[0]) for g in ids]
    half_thickness = float(m.geom_size[ids[0]][1])          # pad box half-size across the gap
    return abs(x[0] - x[1]) - 2 * half_thickness


def main():
    env = make_env(task="one_leg", randomness="none", has_offscreen_renderer=False, use_camera_obs=False)
    rows = []
    for a in np.linspace(-1.0, 1.0, 21):
        env.reset()
        grip = env.robots[0].gripper
        grip.current_action = np.array([a])           # hold the command at exactly `a`
        for _ in range(30):
            act = np.zeros(7)
            env.step(act)                             # zero action keeps current_action unchanged
        rows.append((a, pad_gap(env)))
        print(f"action {a:+.2f}: gap {1000 * rows[-1][1]:6.1f} mm", flush=True)
    a_vals, gaps = np.array(rows).T
    order = np.argsort(gaps)
    a_cap = float(np.interp(TARGET_GAP, gaps[order], a_vals[order]))
    print(f"OPEN_LIMIT for a {1000 * TARGET_GAP:.0f} mm opening: {a_cap:+.3f}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
