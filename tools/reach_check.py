#!/usr/bin/env python3
"""M0 reach check: home pose plus every grasp/insertion pose one_leg needs, solved by IK.

    MUJOCO_GL=egl python tools/reach_check.py            # prints a table; exit 1 if any pose fails
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsim.env.furniture_env import make_env  # noqa: E402
from fsim.env.ik import gripper_down, solve_ik  # noqa: E402
from fsim.geometry import rot_y  # noqa: E402

HOME_TARGET_W = (0.15, 0.0, 0.17)      # grip site 17 cm above the table, between legs and tabletop
REST_QPOS = np.array([0.0, 0.65, 0.0, 1.89, 0.0, 0.6, -math.pi / 2])   # robosuite Kinova3 default posture


def targets(env):
    """(label, W-frame position, rotation) of the poses the task needs."""
    t = env.task
    top = t.part(t.screws[0].parent)
    leg = t.part(t.extras.get("target_leg", t.screws[0].children[0]))
    s = t.screws[0]
    lv = t.randomization.levels["med"].pos_range
    out = [("home", HOME_TARGET_W, gripper_down())]
    for pl in t.layout:
        if pl.part == top.name:
            continue
        for dx in (0.0, lv):
            for dy in (-lv, lv):
                out.append((f"grasp {pl.part[-4:]} ({dx:+.2f},{dy:+.2f})",
                            (pl.pos[0] + dx, pl.pos[1] + dy, pl.pos[2]), gripper_down()))
    rim_z = top.top_z / 2 + top.size[2] / 2 - 0.006
    base = t.placement(top.name).pos
    for dx, dy in ((-lv, -lv), (lv, lv), (0.0, 0.0)):
        out.append((f"grasp top rim ({dx:+.2f},{dy:+.2f})",
                    (base[0] + dx - top.size[0] / 2 + top.rim_width / 2, base[1] + dy, rim_z), gripper_down(math.pi / 2)))
    cx, cy = t.extras["corner_xy"]
    out.append(("top rim at corner", (cx - top.size[0] / 2 + top.rim_width / 2, cy, rim_z), gripper_down(math.pi / 2)))
    grab_z = top.size[2] / 2 + s.catch_z + 0.02          # leg held 2 cm above its origin at thread catch
    for hx, hy in top.hole_centers:
        for yaw in (0.0, math.pi):
            out.append((f"leg over hole ({hx:+.3f},{hy:+.3f}) yaw {math.degrees(yaw):.0f}",
                        (cx + hx, cy + hy, grab_z), gripper_down(yaw)))
        # leg held from the side (approach +x), e.g. right after turning it upright
        out.append((f"(optional) side grip over hole ({hx:+.3f},{hy:+.3f})", (cx + hx, cy + hy, grab_z),
                    rot_y(math.pi / 2) @ np.diag([1.0, 1.0, 1.0])))
    return out


def main():
    env = make_env(task="one_leg", randomness="none", has_offscreen_renderer=False, use_camera_obs=False)
    env.reset()
    m = env.sim.model._model
    r = env.robots[0]
    site = r.gripper.important_sites["grip_site"]
    W = np.asarray(env.task.scene.W_origin)
    q_home = None
    fails = 0
    print(f"{'pose':44s} {'pos err mm':>10s} {'rot err deg':>11s}  ok   joints")
    for label, pos_W, R in targets(env):
        res = solve_ik(m, env.sim.data._data.qpos, site, r._ref_joint_pos_indexes, r._ref_joint_vel_indexes,
                       W + np.asarray(pos_W), R, q0=q_home if q_home is not None else REST_QPOS,
                       q_rest=REST_QPOS, iters=1500)
        if label == "home":
            q_home = res.q
        fails += not res.ok and not label.startswith("(optional)")
        print(f"{label:44s} {1000 * res.pos_err:10.2f} {math.degrees(res.rot_err):11.2f}  {'yes' if res.ok else 'NO '}  "
              f"{np.round(res.q, 3).tolist()}")
    print("HOME_QPOS =", np.round(q_home, 4).tolist())
    print(f"{fails} pose(s) unreachable" if fails else "all poses reachable", flush=True)
    os._exit(1 if fails else 0)       # skip robosuite's noisy EGL teardown


if __name__ == "__main__":
    main()
