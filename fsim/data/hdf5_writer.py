"""Episode recording and atomic per-episode HDF5 files.

Layout (one demo per file; fsim/data/merge.py combines them):

  data.attrs: env_args (JSON, rebuilds the env), total, schema
  data/demo_0/
    actions (T,7)        robosuite OSC_POSE action, normalized (the teammate's convention)
    actions_fb (T,8)     FurnitureBench convention: dpos m (base frame), dquat xyzw (EE frame), gripper
    target_pose (T,7)    commanded grip-site target: pos, quat xyzw
    rewards (T,) dones (T,) skills (T,) skills_manual (T,) phase (T,)
    states (T,1+nq+nv)   [time, qpos, qvel] (teammate-compatible)
    states_full (T,k)    mj_getState(mjSTATE_INTEGRATION): adds ctrl, act, eq_active, warmstart, ...
    controller (T,18)    raw VR controller stream (fsim.teleop.mapping.controller_vector)
    obs/agentview_rgb    front camera (raw robosuite frames, OpenGL convention, uint8, gzip)
    obs/eye_in_hand_rgb  wrist camera;  obs/rear_rgb
    obs/joint_states joint_vel gripper_states gripper_width ee_pos ee_quat(xyzw) ee_pos_vel ee_ori_vel
    obs/parts_poses (T,7P)  parts in the table frame W: pos + quat xyzw, task.part_names order
    obs/operator_rgb     with --record-operator: the scene from the operator's headset pose (fovy 90)
    obs/operator_head_pose (T,7)  that pose in the robot world frame: pos, quat xyzw (camera looks along -z)
  demo attrs: num_samples, model_file, success, task, randomness, seed, layout, operator, ...
    auto_screw: [[first step, end step, seated], ...] for the steps the scripted expert drove (--auto-screw)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import h5py
import numpy as np

SCHEMA = "fsim/1"
IMAGE_KEYS = {"agentview_rgb", "eye_in_hand_rgb", "rear_rgb", "operator_rgb"}


class EpisodeRecorder:
    """Accumulates per-step records (obs is recorded before the action, robomimic style)."""

    def __init__(self):
        self.top: dict[str, list] = defaultdict(list)
        self.obs: dict[str, list] = defaultdict(list)

    def add(self, obs: dict, **step):
        for k, v in obs.items():
            self.obs[k].append(np.asarray(v))
        for k, v in step.items():
            self.top[k].append(np.asarray(v))

    def __len__(self) -> int:
        return len(self.top.get("actions", []))

    def mark_last(self, **values):
        """Set fields of the latest step (filled after env.step), or overwrite them (e.g. done=1 at the end)."""
        n = len(self)
        for k, v in values.items():
            lst = self.top[k]
            if len(lst) < n:
                lst.append(np.asarray(v))
            else:
                lst[-1] = np.asarray(v)


def _jsonable(v):
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return v


def write_episode(path: str, rec: EpisodeRecorder, env_args: dict, attrs: dict) -> str:
    """Write one demo atomically (temp file + rename), so a crash never leaves a broken file."""
    T = len(rec)
    if T == 0:
        raise ValueError("empty episode")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["total"] = T
        data.attrs["schema"] = SCHEMA
        g = data.create_group("demo_0")
        dtypes = {"dones": np.int64, "skills": np.int64, "skills_manual": np.int64, "phase": np.int64,
                  "states": np.float64, "states_full": np.float64}
        for k, vals in rec.top.items():
            g.create_dataset(k, data=np.asarray(vals, dtype=dtypes.get(k, np.float32)))
        og = g.create_group("obs")
        for k, vals in rec.obs.items():
            arr = np.stack(vals)
            if k in IMAGE_KEYS:
                og.create_dataset(k, data=arr.astype(np.uint8), compression="gzip", compression_opts=4,
                                  chunks=(1,) + arr.shape[1:])
            else:
                og.create_dataset(k, data=arr.astype(np.float32))
        g.attrs["num_samples"] = T
        for k, v in attrs.items():
            g.attrs[k] = _jsonable(v)
    os.replace(tmp, path)
    return path
