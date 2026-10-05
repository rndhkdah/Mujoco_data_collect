"""Episode loop for VR collection, independent of the web server (testable headless).

Per episode: RESET (sample layout) -> WAIT (arm holds; B re-rolls the layout; the first clutch
press starts recording) -> RECORD (20 Hz, paced to real time) -> on success with the gripper
released: SETTLE (a few hold steps) -> SAVE; B or the step limit -> DISCARD.

`record_scripted` records an episode driven by a script (e.g. fsim.env.expert) into the same files.
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
from fsim.teleop.mapping import CONTROLLER_DIM, TeleopConfig, TeleopMapper

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


class EpisodeAbort(Exception):
    """A scripted episode ends early: "timeout" (step limit) or "stopped" (collector shut down)."""


@dataclass
class _Episode:
    """What one episode accumulates until it is saved or dropped."""
    rec: EpisodeRecorder
    events: list
    layout: list
    start_snapshot: dict
    wall0: float


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
    def _teleop_info(out) -> dict:
        """Mapper state for the operator view: commanded grip-site target (pos + quat wxyz) and flags."""
        return dict(target=np.r_[out.target_pos, mat_to_quat(out.target_R)], gripper=out.gripper,
                    clutched=out.clutched, precision=out.precision, stale=out.stale)

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
            self.publish("READY", obs, self._teleop_info(out))
            self._pace(t0)
        return None

    def _begin(self) -> "_Episode":
        e = self.env
        return _Episode(rec=EpisodeRecorder(), events=[], start_snapshot=e.fsim_snapshot(), wall0=time.time(),
                        layout=[dict(part=p.part, pos=list(p.pos), quat=list(p.quat)) for p in e.layout])

    def _record_step(self, ep: "_Episode", obs, action, action_fb, target_pose, controller, phase_mark=False):
        """Store the pre-step observation and state with the action, step the env, label the step."""
        states, states_full = self._states()
        ep.rec.add(self._obs_record(obs), actions=action, actions_fb=action_fb, target_pose=target_pose,
                   controller=controller, states=states, states_full=states_full, skills_manual=int(phase_mark))
        obs, reward, _, info = self.env.step(action)
        ep.rec.mark_last(rewards=reward, dones=0, skills=int(info["skill"]), phase=info["phase"])
        ep.events += info["screw_events"]
        return obs, reward, info

    def _settle(self, ep: "_Episode", obs, controller):
        """A few hold steps after success, so the episode ends at rest; the last one is marked done."""
        for _ in range(self.cfg.settle_capture):
            pos, R = self._ee()
            obs, _, _ = self._record_step(ep, obs, self._hold_action(), np.r_[np.zeros(3), 0, 0, 0, 1, -1.0],
                                          np.r_[pos, wxyz_to_xyzw(mat_to_quat(R))], controller)
        ep.rec.mark_last(dones=1)
        return obs

    def _save(self, ep: "_Episode", obs, source: str) -> str:
        e, cfg = self.env, self.cfg
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(cfg.out_dir, e.task.name, e.randomness, f"{stamp}_{self.saved:04d}.hdf5")
        import robosuite
        attrs = dict(success=True, task=e.task.name, randomness=e.randomness, layout=ep.layout, operator=cfg.operator,
                     source=source, duration_s=round(time.time() - ep.wall0, 2), control_hz=e.control_freq,
                     model_file=e.sim.model.get_xml(), screw_events=ep.events, start_snapshot=ep.start_snapshot,
                     end_snapshot=e.fsim_snapshot(), date=stamp, git_commit=_git_commit(),
                     versions=dict(mujoco=mujoco.__version__, robosuite=robosuite.__version__, numpy=np.__version__))
        write_episode(path, ep.rec, env_args_of(e), attrs)
        self.saved += 1
        self.publish("SAVED", obs, {"flash": "SAVED", "path": path})
        return path

    def record_episode(self, obs) -> tuple[str, str | None]:
        cfg = self.cfg
        ep = self._begin()
        outcome = None
        while self.running and outcome is None:
            t0 = time.monotonic()
            msg, age = self.get_controller()
            ee_pos, ee_R = self._ee()
            out = self.mapper.update(msg, age, ee_pos, ee_R)
            if out.discard:
                outcome = "discard"
                break
            obs, _, info = self._record_step(ep, obs, out.action, out.action_fb,
                                             np.r_[out.target_pos, wxyz_to_xyzw(mat_to_quat(out.target_R))],
                                             out.raw, out.phase_mark)
            self.publish("RECORD", obs, dict(info, steps=len(ep.rec), **self._teleop_info(out)))
            if info["success"] and (out.gripper < 0 or not cfg.require_release):
                obs = self._settle(ep, obs, out.raw)
                outcome = "save"
            elif len(ep.rec) >= cfg.max_steps:
                outcome = "timeout"
            self._pace(t0)
        if outcome != "save":
            self.publish("DISCARD", obs, {"flash": "DISCARD" if outcome == "discard" else "TIMEOUT"})
            return outcome or "stopped", None
        return "save", self._save(ep, obs, "teleop")

    def record_scripted(self, script, save: bool = True) -> tuple[str, str | None]:
        """Run one episode driven by a script instead of the VR operator, and record it like a VR
        demo (synthetic demos; checks the data pipeline on full episodes without a headset).

        script(step) runs the episode on the freshly reset env, calling
        step(action, target_pos, target_R) -> (obs, reward, done, info) once per control step.
        The episode is saved only if the env reports success when the script returns.
        save=False only runs and publishes it (the server's autopilot): outcome "success", no file.
        """
        cfg = self.cfg
        state = {"obs": self.env.reset(), "info": {}, "steps": 0}
        ep = self._begin() if save else None
        no_controller = np.zeros(CONTROLLER_DIM, np.float32)

        def step(action, target_pos, target_R):
            if not self.running:
                raise EpisodeAbort("stopped")
            if state["steps"] >= cfg.max_steps:
                raise EpisodeAbort("timeout")
            t0 = time.monotonic()
            action, target_pos = np.asarray(action, float), np.asarray(target_pos, float)
            if save:
                ee_pos, ee_R = self._ee()
                fb = np.r_[target_pos - ee_pos, wxyz_to_xyzw(mat_to_quat(ee_R.T @ target_R)), action[6]]
                obs, reward, info = self._record_step(ep, state["obs"], action, fb,
                                                      np.r_[target_pos, wxyz_to_xyzw(mat_to_quat(target_R))], no_controller)
            else:
                obs, reward, _, info = self.env.step(action)
            state.update(obs=obs, info=info, steps=state["steps"] + 1)
            self.publish("RECORD", obs, dict(info, steps=state["steps"], target=np.r_[target_pos, mat_to_quat(target_R)],
                                             gripper=float(action[6]), clutched=True, precision=False, stale=False))
            self._pace(t0)
            return obs, reward, False, info

        try:
            script(step)
            outcome = "save" if state["info"].get("success") else "failed"
        except EpisodeAbort as e:
            outcome = str(e)
        if outcome != "save":
            self.publish("DISCARD", state["obs"], {"flash": "TIMEOUT" if outcome == "timeout" else "DISCARD"})
            return outcome, None
        if not save:
            self.publish("SAVED", state["obs"], {"flash": "SUCCESS"})
            return "success", None
        obs = self._settle(ep, state["obs"], no_controller)
        return "save", self._save(ep, obs, "scripted")

    def run(self, num_demos: int):
        while self.running and self.saved < num_demos:
            obs = self.wait_for_start()
            if obs is None:
                break
            outcome, path = self.record_episode(obs)
            print(f"[collect] {outcome}" + (f": {path}" if path else ""), flush=True)
