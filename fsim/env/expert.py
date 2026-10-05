"""Scripted one_leg expert for the Kinova Gen3 + Robotiq 2F-85 (M2 exit check; synthetic demos).

The FurnitureBench phases, adapted to a parallel gripper that screws with the Kinova's continuous
wrist joint:
  grasp_top         clamp the tabletop's rim wall that faces the robot
  place_top_corner  turn the tabletop square to the fence, slide it snug into the corner, release
  pick_leg          grasp the lying leg across two faces, with the gripper tilted 30 deg, and lift it
  insert_leg        carry it over a hole, then pitch the gripper 90 deg about the finger axis so the
                    leg hangs stud-down beside the fingers; lower the stud into the bore until it
                    rests on the thread catch; release (the leg stands) and back away
  screw_leg         regrasp from above and turn the wrist while pressing down until the thread
                    catches and the leg seats; release

It reads part poses from the simulation state (a privileged expert, not a vision policy), and it
handles the none/low/med layouts. The "high" presets can need what it does not plan for: clearing
a leg out of the corner first, or grasping a leg that lies out of reach of the tilted grasp.

    MUJOCO_GL=egl python -m fsim.env.expert --randomness low --episodes 20                 # success rate
    MUJOCO_GL=egl python -m fsim.env.expert --randomness low --episodes 5 --record DIR     # save demos (HDF5)
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from fsim.env.ik import _rot_error, gripper_down
from fsim.env.scripted import ScriptedArm, axis_angle
from fsim.geometry import rot_z, yaw_of

SAFE_Z = 0.12                    # transit height of the grip site above the table (m)
RIM_PAD_DEPTH = 0.008            # grip site below the rim top when clamping the tabletop wall
CORNER_PUSH = 0.006              # aim this far past the corner so the tabletop seats against both bars
LEG_LIFT_Z = 0.20                # grip-site height after picking the leg, and for the regrasp transit
# The lying leg is grasped with the gripper tilted this far from vertical, wrist toward the leg's
# far end. With the leg upright the gripper then points this far below horizontal, which keeps the
# wrist links (they hang 5-9 cm below a horizontal gripper) above the lying legs and the fence.
GRASP_TILT = math.radians(30.0)
# Pointing the gripper straight away from the base would put the wrist center too close to the
# shoulder (joint 6 runs into its limit), so it approaches the hole this far off the base-to-hole line.
APPROACH_OFFSET = math.radians(45.0)
INSERT_ABOVE = 0.015             # start lowering with the stud this far above the thread catch
INSERT_SPEED = 0.015             # m/s
REGRASP_ABOVE = 0.02             # regrasp the standing leg this far above its origin
TWIST_RATE = math.radians(90.0)  # wrist turn rate while screwing (rad/s)
MAX_TWIST_LEAD = math.radians(25.0)   # the twist target stops running ahead of the wrist beyond this
_DOWN = gripper_down(0.0)


class ExpertFailure(RuntimeError):
    pass


@dataclass
class ExpertResult:
    success: bool
    phase: int                   # env phase index reached (len(task.phases) when all done)
    steps: int
    failed: str | None = None
    notes: dict = field(default_factory=dict)


def _wrap(d: float, period: float) -> float:
    return d - period * round(d / period)


class OneLegExpert:
    def __init__(self, env, log=None, step_fn=None):
        """step_fn: see ScriptedArm (records the episode when it comes from Collector.record_scripted)."""
        self.env = env
        self.arm = ScriptedArm(env, step_fn=step_fn)
        t = env.task
        self.screw_spec = t.screws[0]
        self.top, self.leg = self.screw_spec.parent, t.extras["target_leg"]
        self.top_spec, self.leg_spec = t.part(self.top), t.part(self.leg)
        self.W = np.asarray(t.scene.W_origin, float)
        self.log = log or (lambda msg: None)
        self.notes: dict = {}
        self.hole = None

    # ------------------------------------------------------------------ helpers
    def _yaw_near(self, yaw: float, period: float = math.pi) -> float:
        """The gripper yaw equivalent to `yaw` (fingers are symmetric) closest to the current one."""
        cur = yaw_of(self.arm.eef_R @ _DOWN.T)
        return cur + _wrap(yaw - cur, period)

    @staticmethod
    def _check(ok: bool, what: str):
        if not ok:
            raise ExpertFailure(what)

    def _transit(self, xy, R, z=SAFE_Z):
        """Rise to the transit height, then travel above `xy`."""
        arm = self.arm
        if arm.eef_pos[2] < self.W[2] + z - 0.01:
            arm.move_linear(np.r_[arm.eef_pos[:2], self.W[2] + z], arm.eef_R, speed=0.12)
        self._check(arm.move_linear(np.r_[xy, self.W[2] + z], R), "reach the transit pose")

    def _leg_axis_and_tip(self):
        T = self.env.part_state.pose(self.leg)
        return T[:3, 2], T[:3, 3] + T[:3, 2] * self.leg_spec.tip_z

    # ------------------------------------------------------------------ phases
    def grasp_top(self):
        env, arm, spec = self.env, self.arm, self.top_spec
        T = env.part_state.pose(self.top)
        psi = yaw_of(T[:3, :3])
        a = min((psi + k * math.pi / 2 for k in range(4)), key=math.cos)   # wall normal most toward the robot
        n = np.array([math.cos(a), math.sin(a), 0.0])
        p = T[:3, 3] + n * (spec.size[0] / 2 - spec.rim_width / 2)
        p[2] = T[2, 3] + spec.top_z - RIM_PAD_DEPTH
        R = gripper_down(self._yaw_near(a - math.pi / 2))                    # fingers across the wall
        arm.gripper = -1.0
        self._transit(p[:2], R)
        self._check(arm.move_linear(p, R, speed=0.08, tol=0.002), "descend to the tabletop rim")
        arm.grip(True, 15)
        self._check(env.grasped(self.top), "clamp the tabletop rim")

    def place_top_corner(self):
        env, arm, spec = self.env, self.arm, self.top_spec
        cx, cy = env.task.extras["corner_xy"]
        half = spec.size[0] / 2 - spec.rim_width / 2
        # turn in place until the clamped wall faces -x (tabletop square to the fence), then slide
        R = gripper_down(self._yaw_near(math.pi / 2))
        arm.move_linear(arm.eef_pos, R, ang_speed=0.6, settle_steps=20)
        goal = np.array([self.W[0] + cx - half + CORNER_PUSH, self.W[1] + cy + CORNER_PUSH, arm.eef_pos[2]])
        arm.move_linear(goal, R, speed=0.06, settle_steps=15)                # the fence stops it short
        arm.grip(False, 12)
        arm.move_linear(arm.eef_pos + np.array([0.0, 0.0, 0.08]), arm.eef_R, speed=0.12)
        T = env.part_state.pose_W(self.top)
        err = float(np.linalg.norm(T[:2, 3] - np.asarray([cx, cy])))
        self.notes["corner_err_mm"] = round(1000 * err, 1)
        self._check(err < env.task.extras["corner_tol"], f"tabletop in the corner (off by {1000 * err:.0f} mm)")

    def pick_leg(self):
        env, arm = self.env, self.arm
        T = env.part_state.pose(self.leg)
        u = T[:3, 2]                                                      # long axis; the stud is at -u
        self._check(abs(u[2]) < 0.2, "find the leg lying down")
        R_down = gripper_down(math.atan2(u[1], u[0]))                     # fingers across the leg, site y along u
        R = axis_angle(R_down[:, 0], GRASP_TILT) @ R_down
        p = T[:3, 3].copy()
        self._transit(p[:2], R)
        self._check(arm.move_linear(p, R, speed=0.08, tol=0.002), "descend to the leg")
        arm.grip(True, 15)
        self._check(env.grasped(self.leg), "grasp the leg")
        arm.move_linear(np.r_[arm.eef_pos[:2], self.W[2] + LEG_LIFT_Z], arm.eef_R, speed=0.1)
        self._check(env.grasped(self.leg), "lift the leg")

    def _choose_hole(self):
        """The hole nearest the robot base, and its seated-child frame in the world."""
        env = self.env
        T_top = env.part_state.pose(self.top)
        base = np.asarray(env.task.scene.robot_base, float)
        holes = [h for h in env.screw.holes if h.parent == self.top]
        seats = {h.index: T_top @ h.T_local for h in holes}
        hole = min(holes, key=lambda h: float(np.linalg.norm(seats[h.index][:2, 3] - base[:2])))
        return hole, seats[hole.index]

    def insert_leg(self):
        env, arm, s = self.env, self.arm, self.screw_spec
        hole, seat = self._choose_hole()
        self.hole = hole
        axis = seat[:3, 2]
        tip_len = -self.leg_spec.tip_z
        catch = seat[:3, 3] + axis * s.travel                             # leg origin with the stud on the catch
        # Gripper orientation with the leg upright (pitch about the finger axis) and the approach
        # direction APPROACH_OFFSET off the base-to-hole line.
        T_leg = env.part_state.pose(self.leg)
        R0, p0 = arm.eef_R, arm.eef_pos
        a, u = R0[:, 0], T_leg[:3, 2]
        theta = math.atan2(float(np.dot(a, np.cross(u, axis))), float(np.dot(u, axis)))
        R_up = axis_angle(a, theta) @ R0
        base = np.asarray(env.task.scene.robot_base, float)
        d = catch[:2] - base[:2]
        app = R_up[:, 2]
        phi = math.atan2(d[1], d[0]) + APPROACH_OFFSET
        R = rot_z(_wrap(phi - math.atan2(app[1], app[0]), 2 * math.pi)) @ R_up
        off = R0.T @ (T_leg[:3, 3] - p0)                                  # leg origin in the gripper frame
        site_above = catch + axis * (INSERT_ABOVE + 0.04) - R @ off
        # Carry first and turn the leg upright in place over the tabletop: upright next to the
        # robot, where the leg was picked up, the wrist would be out of range.
        hi = np.r_[site_above[:2], self.W[2] + LEG_LIFT_Z]
        self._check(arm.move_linear(hi, R0), "carry the leg over the tabletop")
        self._check(arm.move_linear(hi, R, ang_speed=0.8), "turn the leg upright")
        self._check(env.grasped(self.leg), "keep hold of the leg while turning it upright")
        arm.move_linear(site_above, R, speed=0.08)

        def tip_target(z_off):                                            # stud tip on the hole axis
            return seat[:3, 3] + axis * (s.travel - tip_len + z_off)

        # square the leg to the hole axis and put the stud tip over the bore
        for _ in range(3):
            lz, tip = self._leg_axis_and_tip()
            v = np.cross(lz, axis)
            ang = math.asin(min(1.0, float(np.linalg.norm(v))))
            if ang > math.radians(0.3):
                R = axis_angle(v / np.linalg.norm(v), ang) @ arm.eef_R
            arm.move_to(arm.eef_pos + (tip_target(INSERT_ABOVE) - tip), R, tol=0.001, rot_tol=0.01, max_steps=40)
        lz, tip = self._leg_axis_and_tip()
        tilt = math.degrees(math.acos(float(np.clip(np.dot(lz, axis), -1.0, 1.0))))
        self.notes["upright_tilt_deg"] = round(tilt, 1)
        self._check(tilt < 5.0, f"hold the leg upright over the hole (tilt {tilt:.1f} deg)")
        # lower with closed-loop correction on the stud tip, aiming slightly below the catch
        z_off, rest = INSERT_ABOVE, 0
        step = INSERT_SPEED / env.control_freq
        for _ in range(200):
            z_off = max(z_off - step, -0.003)
            _, tip = self._leg_axis_and_tip()
            arm.step(arm.action_toward(arm.eef_pos + (tip_target(z_off) - tip), R))
            m = env.screw.measure(self.leg, hole)
            rest = rest + 1 if m.axial - s.travel < 0.0015 else 0
            if rest >= 5:
                break
        m = env.screw.measure(self.leg, hole)
        self.notes["insert"] = dict(axial_mm=round(1000 * (m.axial - s.travel), 2), radial_mm=round(1000 * m.radial, 2),
                                    tilt_deg=round(m.tilt_deg, 2))
        self._check(rest >= 5, f"rest the stud on the catch ({self.notes['insert']})")
        arm.grip(False, 12)
        arm.move_linear(arm.eef_pos - arm.eef_R[:, 2] * 0.06, arm.eef_R, speed=0.06)   # back the fingers out
        arm.move_linear(arm.eef_pos + np.array([0.0, 0.0, 0.06]), arm.eef_R, speed=0.1)
        self._check(env.screw.resting_hole(self.leg) is not None, "leave the leg standing in the hole")

    def screw_leg(self):
        env, arm = self.env, self.arm
        hole = env.screw.resting_hole(self.leg) or self.hole
        seat = env.part_state.pose(self.top) @ hole.T_local
        T = env.part_state.pose(self.leg)
        R = gripper_down(self._yaw_near(yaw_of(T[:3, :3]), period=math.pi / 2))   # fingers on two faces
        grasp = np.r_[seat[:2, 3], T[2, 3] + REGRASP_ABOVE]
        arm.gripper = -1.0
        self._transit(grasp[:2], R, z=LEG_LIFT_Z)                           # clear the standing leg
        self._check(arm.move_linear(grasp, R, speed=0.06, tol=0.002), "descend onto the standing leg")
        arm.grip(True, 15)
        self._check(env.grasped(self.leg), "regrasp the leg")
        rate = TWIST_RATE / env.control_freq
        seated_at, R_t = None, arm.eef_R
        for k in range(600):
            cand = rot_z(-rate) @ R_t                                       # tighten: clockwise seen from above
            if np.linalg.norm(_rot_error(arm.eef_R, cand)) <= MAX_TWIST_LEAD:
                R_t = cand
            arm.step(arm.action_toward(np.r_[seat[:2, 3], arm.eef_pos[2] - 0.001], R_t))
            if seated_at is None and env.screw.child_state(self.leg) == "seated":
                seated_at = k
            if seated_at is not None and k >= seated_at + 5:
                break
        self.notes["screw_steps"] = seated_at
        arm.grip(False, 12)
        arm.move_linear(arm.eef_pos + np.array([0.0, 0.0, 0.08]), arm.eef_R, speed=0.1)
        self._check(seated_at is not None, f"seat the leg (state {env.screw.child_state(self.leg)})")

    # ------------------------------------------------------------------ episode
    def run(self) -> ExpertResult:
        env = self.env
        try:
            for name, fn in (("grasp_top", self.grasp_top), ("place_top_corner", self.place_top_corner),
                             ("pick_leg", self.pick_leg), ("insert_leg", self.insert_leg), ("screw_leg", self.screw_leg)):
                fn()
                self.log(f"  {name:17s} done: env phase {env.phases.phase}, step {env.timestep}")
        except ExpertFailure as e:
            self.notes["failed_pose_err"] = self._pose_report()
            return ExpertResult(False, env.phases.phase, env.timestep, str(e), self.notes)
        return ExpertResult(bool(env._check_success()), env.phases.phase, env.timestep, None, self.notes)

    def _pose_report(self) -> str:
        """Arm joints nearest their limits (most failures are a joint running into its stop)."""
        env = self.env
        r, m, d = env.robots[0], env.sim.model._model, env.sim.data._data
        out = []
        for i, adr in enumerate(r._ref_joint_pos_indexes):
            j = int(np.flatnonzero(m.jnt_qposadr == adr)[0])
            if m.jnt_limited[j]:
                margin = min(d.qpos[adr] - m.jnt_range[j, 0], m.jnt_range[j, 1] - d.qpos[adr])
                if margin < 0.1:
                    out.append(f"joint {i + 1} within {margin:.2f} rad of its limit")
        return "; ".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--randomness", default="low", choices=["none", "low", "med", "high"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--record", metavar="DIR", default=None,
                    help="record the successful episodes (cameras on) as HDF5 demos under DIR, like the VR server does")
    ap.add_argument("--camera-size", type=int, default=224)
    args = ap.parse_args(argv)
    from fsim.env.furniture_env import make_env

    log = print if args.verbose else None
    ok, t0 = 0, time.time()
    if args.record:
        from fsim.collect import CollectConfig, Collector

        env = make_env(task="one_leg", randomness=args.randomness, seed=args.seed, camera_size=args.camera_size)
        collector = Collector(env, CollectConfig(out_dir=os.path.expanduser(args.record), operator="scripted_expert",
                                                 realtime=False), get_controller=lambda: (None, 1e9))
    else:
        env = make_env(task="one_leg", randomness=args.randomness, seed=args.seed,
                       has_offscreen_renderer=False, use_camera_obs=False)
    for ep in range(args.episodes):
        if args.record:
            out = {}
            outcome, path = collector.record_scripted(lambda step: out.update(res=OneLegExpert(env, log, step).run()))
            res, saved = out["res"], f"  -> {path}" if path else f"  ({outcome}, not saved)"
        else:
            env.reset()
            res, saved = OneLegExpert(env, log).run(), ""
        ok += res.success
        print(f"episode {ep:3d}: {'success' if res.success else 'FAIL ' + str(res.failed):60s} "
              f"phase {res.phase}/{len(env.task.phases)}  steps {res.steps}  {res.notes}{saved}", flush=True)
    print(f"{ok}/{args.episodes} successful ({args.randomness}), {time.time() - t0:.0f} s", flush=True)
    os._exit(0)                      # skip robosuite's EGL teardown


if __name__ == "__main__":
    main()
