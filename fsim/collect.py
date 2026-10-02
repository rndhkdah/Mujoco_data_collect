"""Episode loop for VR collection, independent of the web server (testable headless).

Per episode: RESET (sample layout) -> WAIT (arm holds; B re-rolls the layout; the first clutch
press starts recording) -> RECORD (20 Hz, paced to real time) -> on success with the gripper
released: SETTLE (a few hold steps) -> SAVE; B or the step limit -> DISCARD.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from fsim.data.hdf5_writer import EpisodeRecorder, write_episode
from fsim.geometry import wxyz_to_xyzw, mat_to_quat
from fsim.teleop.mapping import TeleopConfig, TeleopMapper

CAMERA_KEYS = {"front_image": "agentview_rgb", "robot0_eye_in_hand_image": "eye_in_hand_rgb", "rear_image": "rear_rgb"}


@dataclass
class CollectConfig:
    out_dir: str
    operator: str = ""
    max_steps: int = 3000
    settle_capture: int = 12
    realtime: bool = True
    require_release: bool = True      # success counts once the gripper is commanded open
    teleop: TeleopConfig = field(default_factory=TeleopConfig)


def _git_commit() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def env_args_of(env) -> dict:
    """robomimic-style env metadata with the real constructor kwargs."""
    ctrl = env.controller_configs_arg
    return {"env_name": type(env).__name__, "type": 1, "env_version": "fsim",
            "env_kwargs": {"robots": [r.name for r in env.robots], "task": env.task.name,
                           "randomness": env.randomness, "gripper_types": "FsimRobotiq2f85",
                           "controller_configs": ctrl, "control_freq": env.control_freq, "horizon": env.horizon,
                           "ignore_done": env.ignore_done, "hard_reset": env.hard_reset,
                           "camera_names": list(env.camera_names), "camera_heights": list(env.camera_heights),
                           "camera_widths": list(env.camera_widths), "use_camera_obs": env.use_camera_obs,
                           "has_offscreen_renderer": env.has_offscreen_renderer, "has_renderer": False}}


class Collector:
    def __init__(self, env, cfg: CollectConfig, get_controller, publish=None):
        """get_controller() -> (message dict or None, age in seconds); publish(state, obs, info) per step."""
        self.env, self.cfg = env, cfg
        self.get_controller = get_controller
        self.publish = publish or (lambda *a, **k: None)
        self.mapper = TeleopMapper(cfg.teleop, env.control_freq)
        self.running = True
        self.saved = 0
        m = env.sim.model._model
        self._state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self._state_buf = np.zeros(mujoco.mj_stateSize(m, self._state_spec))

    # ---------------------------------------------------------------- helpers
    def _ee(self):
        e, sid = self.env, self.env.robots[0].eef_site_id
        return np.array(e.sim.data.site_xpos[sid]), np.array(e.sim.data.site_xmat[sid]).reshape(3, 3)

    def _obs_record(self, obs) -> dict:
        e = self.env
        rec = {CAMERA_KEYS[k]: obs[k] for k in CAMERA_KEYS if k in obs}
        lin, ang = e.eef_velocity()
        r, d = e.robots[0], e.sim.data._data
        rec.update(joint_states=d.qpos[r._ref_joint_pos_indexes].copy(), joint_vel=d.qvel[r._ref_joint_vel_indexes].copy(),
                   gripper_states=obs["robot0_gripper_qpos"], gripper_width=[e.gripper_width()],
                   ee_pos=obs["robot0_eef_pos"], ee_quat=obs["robot0_eef_quat"], ee_pos_vel=lin, ee_ori_vel=ang,
                   parts_poses=e.parts_pose_vector())
        return rec

    def _states(self):
        d, m = self.env.sim.data._data, self.env.sim.model._model
        mujoco.mj_getState(m, d, self._state_buf, self._state_spec)
        return np.concatenate([[d.time], d.qpos, d.qvel]), self._state_buf.copy()

    def _pace(self, t0):
        if self.cfg.realtime:
            rest = 1.0 / self.env.control_freq - (time.monotonic() - t0)
            if rest > 0:
                time.sleep(rest)

    @staticmethod
    def _hold_action(gripper=-1.0):
        """Zero OSC delta (holds the current pose) with the gripper command."""
        a = np.zeros(7)
        a[6] = gripper
        return a

    # ---------------------------------------------------------------- episode
    def wait_for_start(self):
        """Hold the arm until the operator presses the clutch. Returns the obs to start from, or None."""
        obs = self.env.reset()
        self.mapper.reset(*self._ee())
        self.publish("READY", obs, {})
        while self.running:
            t0 = time.monotonic()
            msg, age = self.get_controller()
            out = self.mapper.update(msg, age, *self._ee())
            if out.discard:                                   # B while waiting: new layout
                obs = self.env.reset()
                self.mapper.reset(*self._ee())
                self.publish("READY", obs, {"flash": "NEW LAYOUT"})
                continue
            if out.clutched:
                return obs
            obs, _, _, _ = self.env.step(self._hold_action())
            self.publish("READY", obs, {})
            self._pace(t0)
        return None

    def record_episode(self, obs) -> tuple[str, str | None]:
        e, cfg = self.env, self.cfg
        rec = EpisodeRecorder()
        events, layout = [], [dict(part=p.part, pos=list(p.pos), quat=list(p.quat)) for p in e.layout]
        start_snapshot, wall0 = e.fsim_snapshot(), time.time()
        outcome = None
        while self.running and outcome is None:
            t0 = time.monotonic()
            msg, age = self.get_controller()
            ee_pos, ee_R = self._ee()
            out = self.mapper.update(msg, age, ee_pos, ee_R)
            if out.discard:
                outcome = "discard"
                break
            states, states_full = self._states()
            rec.add(self._obs_record(obs), actions=out.action, actions_fb=out.action_fb,
                    target_pose=np.r_[out.target_pos, wxyz_to_xyzw(mat_to_quat(out.target_R))],
                    controller=out.raw, states=states, states_full=states_full, skills_manual=int(out.phase_mark))
            obs, reward, _, info = e.step(out.action)
            rec.mark_last(rewards=reward, dones=0, skills=int(info["skill"]), phase=info["phase"])
            events += info["screw_events"]
            self.publish("RECORD", obs, dict(info, steps=len(rec), clutched=out.clutched, precision=out.precision,
                                             stale=out.stale))
            if info["success"] and (out.gripper < 0 or not cfg.require_release):
                for _ in range(cfg.settle_capture):
                    states, states_full = self._states()
                    a = self._hold_action()
                    rec.add(self._obs_record(obs), actions=a, actions_fb=np.r_[np.zeros(3), 0, 0, 0, 1, -1.0],
                            target_pose=np.r_[self._ee()[0], wxyz_to_xyzw(mat_to_quat(self._ee()[1]))],
                            controller=out.raw, states=states, states_full=states_full, skills_manual=0)
                    obs, reward, _, info = e.step(a)
                    rec.mark_last(rewards=reward, dones=0, skills=int(info["skill"]), phase=info["phase"])
                    events += info["screw_events"]
                rec.mark_last(dones=1)
                outcome = "save"
            elif len(rec) >= cfg.max_steps:
                outcome = "timeout"
            self._pace(t0)
        if outcome != "save":
            self.publish("DISCARD", obs, {"flash": "DISCARD" if outcome == "discard" else "TIMEOUT"})
            return outcome or "stopped", None
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(cfg.out_dir, e.task.name, e.randomness, f"{stamp}_{self.saved:04d}.hdf5")
        import robosuite
        attrs = dict(success=True, task=e.task.name, randomness=e.randomness, layout=layout, operator=cfg.operator,
                     duration_s=round(time.time() - wall0, 2), control_hz=e.control_freq,
                     model_file=e.sim.model.get_xml(), screw_events=events, start_snapshot=start_snapshot,
                     end_snapshot=e.fsim_snapshot(), date=stamp, git_commit=_git_commit(),
                     versions=dict(mujoco=mujoco.__version__, robosuite=robosuite.__version__, numpy=np.__version__))
        write_episode(path, rec, env_args_of(e), attrs)
        self.saved += 1
        self.publish("SAVED", obs, {"flash": "SAVED", "path": path})
        return "save", path

    def run(self, num_demos: int):
        while self.running and self.saved < num_demos:
            obs = self.wait_for_start()
            if obs is None:
                break
            outcome, path = self.record_episode(obs)
            print(f"[collect] {outcome}" + (f": {path}" if path else ""), flush=True)
