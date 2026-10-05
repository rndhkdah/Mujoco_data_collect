"""Quest controller -> end-effector target -> robosuite OSC_POSE action.

Controls (right hand unless noted):
  grip (hold)      clutch: the end-effector follows the controller in position AND orientation,
                   relative to where both were when the grip was pressed
  trigger (hold)   close the gripper (release = open), as in FurnitureBench's Oculus teleop
  A (hold)         precision mode: smaller position/rotation gain (re-anchors, so no jump)
  stick x          twist about the tool axis (rate control; works with or without the clutch,
                   e.g. to screw a leg in with the Kinova's unlimited wrist)
  stick y          push / pull along the tool axis (rate control)
  B (press)        discard the episode
  left X (press)   manual phase mark
Messages older than `stale_after` seconds, or with an untracked controller, release the clutch.

The action is the teammate's P-law on the absolute target: clip(error * kp, -1, 1) with small
deadbands; robosuite's OSC_POSE turns it into deltas of up to 5 cm and 0.5 rad per step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fsim.geometry import mat_to_quat, quat_to_mat, wxyz_to_xyzw, xyzw_to_wxyz

# WebXR local-floor (x right, y up, -z forward) -> robot base frame (x forward, y left, z up),
# with the operator facing the robot's +x (the teammate's AXIS_FACING).
XR_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def rotvec(R: np.ndarray) -> np.ndarray:
    q = mat_to_quat(R)
    angle = 2.0 * math.acos(min(1.0, max(-1.0, q[0])))
    s = math.sqrt(max(1.0 - q[0] * q[0], 1e-12))
    return q[1:] / s * angle if angle > 1e-9 else np.zeros(3)


def axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = angle / 2.0
    return quat_to_mat(np.r_[math.cos(h), math.sin(h) * axis])


def scale_rotation(R: np.ndarray, s: float) -> np.ndarray:
    v = rotvec(R)
    a = np.linalg.norm(v)
    return np.eye(3) if a < 1e-12 else axis_angle(v / a, a * s)


@dataclass
class TeleopConfig:
    speed: float = 0.7                        # global scale on every teleop motion: hand following and stick rates
    pos_scale: float = 1.0
    rot_scale: float = 1.0
    precision_pos: float = 0.2
    precision_rot: float = 0.3
    twist_rate: float = math.radians(120.0)   # rad/s at full stick
    push_rate: float = 0.02                   # m/s at full stick
    stick_deadzone: float = 0.15
    max_twist_lead: float = math.radians(25.0)   # stick twist stops leading the arm by more than this
    max_push_lead: float = 0.01                  # and pushing by more than this (m)
    pos_kp: float = 12.0
    rot_kp: float = 2.5
    pos_deadband: float = 0.001
    rot_deadband: float = 0.01
    engage_threshold: float = 0.5
    stale_after: float = 0.15
    # Grip-site workspace in the robot world frame (keeps targets off the table and in reach).
    workspace_min: tuple = (-0.25, -0.45, 0.418)
    workspace_max: tuple = (0.55, 0.45, 0.95)


@dataclass
class TeleopOutput:
    action: np.ndarray            # 7-D OSC_POSE action
    action_fb: np.ndarray         # 8-D FurnitureBench convention: dpos (m), dquat xyzw (EE frame), gripper
    target_pos: np.ndarray
    target_R: np.ndarray
    gripper: float
    clutched: bool
    precision: bool
    stale: bool
    discard: bool = False
    phase_mark: bool = False
    raw: np.ndarray = field(default_factory=lambda: np.zeros(CONTROLLER_DIM, np.float32))


CONTROLLER_DIM = 18   # see controller_vector()


def controller_vector(msg: dict | None) -> np.ndarray:
    """Fixed-size record of the controller stream: right pos(3) quat xyzw(4) trigger grip a b
    stick(2) tracked, left x y stick(2)."""
    v = np.zeros(CONTROLLER_DIM, np.float32)
    if not msg:
        return v
    r, l = msg.get("right") or {}, msg.get("left") or {}
    v[0:3] = r.get("pos", (0, 0, 0))
    v[3:7] = r.get("quat", (0, 0, 0, 1))
    v[7], v[8] = r.get("trigger", 0.0), r.get("grip", 0.0)
    v[9], v[10] = float(bool(r.get("a"))), float(bool(r.get("b")))
    v[11:13] = (r.get("stick") or (0.0, 0.0))[:2]
    v[13] = float(bool(r.get("tracked", True)))
    v[14], v[15] = float(bool(l.get("x"))), float(bool(l.get("y")))
    v[16:18] = (l.get("stick") or (0.0, 0.0))[:2]
    return v


def head_pose(msg: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Headset pose (pos, R) in the robot world frame from a controller message, or None. The head
    frame is a MuJoCo camera frame: x right, y up, looking along -z."""
    h = (msg or {}).get("head")
    if not h:
        return None
    p, R = np.asarray(h["pos"], float), quat_to_mat(xyzw_to_wxyz(h["quat"]))
    if msg.get("frame", "xr") == "xr":
        p, R = XR_TO_ROBOT @ p, XR_TO_ROBOT @ R
    return p, R


class TeleopMapper:
    def __init__(self, cfg: TeleopConfig | None = None, control_hz: float = 20.0):
        self.cfg = cfg or TeleopConfig()
        self.dt = 1.0 / control_hz
        self.reset(np.zeros(3), np.eye(3))

    def reset(self, ee_pos, ee_R):
        self.target_pos = np.array(ee_pos, float)
        self.target_R = np.array(ee_R, float)
        self.anchor = None            # (controller pos, controller R, ee target pos, ee target R)
        self.anchor_precision = False
        self.gripper = -1.0
        self.prev_b = False
        self.prev_x = False
        self.stick_active = False

    def _controller_pose(self, msg) -> tuple[np.ndarray, np.ndarray]:
        r = msg["right"]
        p = np.asarray(r["pos"], float)
        R = quat_to_mat(xyzw_to_wxyz(r["quat"]))
        if msg.get("frame", "xr") == "xr":
            p, R = XR_TO_ROBOT @ p, XR_TO_ROBOT @ R @ XR_TO_ROBOT.T
        return p, R

    def update(self, msg: dict | None, age: float, ee_pos, ee_R) -> TeleopOutput:
        """msg: latest controller message (or None); age: seconds since it arrived."""
        cfg = self.cfg
        ee_pos, ee_R = np.asarray(ee_pos, float), np.asarray(ee_R, float)
        right = (msg or {}).get("right") or {}
        stale = msg is None or age > cfg.stale_after or not right.get("tracked", True)
        discard = mark = False
        precision = False
        if stale:
            self.anchor = None
            self.stick_active = False
            self.target_pos, self.target_R = ee_pos.copy(), ee_R.copy()
        else:
            left = msg.get("left") or {}
            b, x = bool(right.get("b")), bool(left.get("x"))
            discard, mark = b and not self.prev_b, x and not self.prev_x
            self.prev_b, self.prev_x = b, x
            self.gripper = 1.0 if float(right.get("trigger", 0.0)) > 0.5 else -1.0
            precision = bool(right.get("a"))
            clutch = float(right.get("grip", 0.0)) > cfg.engage_threshold
            if clutch:
                c_pos, c_R = self._controller_pose(msg)
                if self.anchor is None or precision != self.anchor_precision:
                    self.anchor = (c_pos, c_R, self.target_pos.copy() if self.anchor else ee_pos.copy(),
                                   self.target_R.copy() if self.anchor else ee_R.copy())
                    self.anchor_precision = precision
                c0_pos, c0_R, e0_pos, e0_R = self.anchor
                ps = cfg.speed * cfg.pos_scale * (cfg.precision_pos if precision else 1.0)
                rs = cfg.speed * cfg.rot_scale * (cfg.precision_rot if precision else 1.0)
                S = scale_rotation(c_R @ c0_R.T, rs)
                self.target_pos = e0_pos + ps * (c_pos - c0_pos)
                self.target_R = S @ e0_R
            sx, sy = (right.get("stick") or (0.0, 0.0))[:2]
            twist, push = abs(sx) > cfg.stick_deadzone, abs(sy) > cfg.stick_deadzone
            sticks = twist or push
            if not clutch:
                self.anchor = None
                # Idle, or a stick was just engaged: start from the arm. While a stick stays engaged
                # the target integrates on its own, so the arm turns at the commanded rate.
                if not (sticks and self.stick_active):
                    self.target_pos, self.target_R = ee_pos.copy(), ee_R.copy()
            self.stick_active = sticks
            tool = self.target_R[:, 2]
            dR = axis_angle(tool, sx * cfg.speed * cfg.twist_rate * self.dt) if twist else None
            if dR is not None and np.linalg.norm(rotvec(dR @ self.target_R @ ee_R.T)) <= cfg.max_twist_lead:
                self.target_R = dR @ self.target_R
                if self.anchor is not None:      # keep the twist when the clutch is held: target = dR S e0
                    c0_pos, c0_R, e0_pos, e0_R = self.anchor
                    self.anchor = (c0_pos, c0_R, e0_pos, S.T @ dR @ S @ e0_R)
            dp = tool * (-sy) * cfg.speed * cfg.push_rate * self.dt if push else None   # stick forward (negative y) pushes
            if dp is not None and np.linalg.norm(self.target_pos + dp - ee_pos) <= cfg.max_push_lead:
                self.target_pos = self.target_pos + dp
                if self.anchor is not None:
                    c0_pos, c0_R, e0_pos, e0_R = self.anchor
                    self.anchor = (c0_pos, c0_R, e0_pos + dp, e0_R)
        self.target_pos = np.clip(self.target_pos, cfg.workspace_min, cfg.workspace_max)

        a = np.zeros(7)
        perr = self.target_pos - ee_pos
        perr[np.abs(perr) < cfg.pos_deadband] = 0.0
        a[:3] = np.clip(perr * cfg.pos_kp, -1.0, 1.0)
        rerr = rotvec(self.target_R @ ee_R.T)
        rerr[np.abs(rerr) < cfg.rot_deadband] = 0.0
        a[3:6] = np.clip(rerr * cfg.rot_kp, -1.0, 1.0)
        a[6] = self.gripper
        dq = wxyz_to_xyzw(mat_to_quat(ee_R.T @ self.target_R))
        fb = np.r_[self.target_pos - ee_pos, dq, self.gripper]
        return TeleopOutput(action=a, action_fb=fb, target_pos=self.target_pos.copy(), target_R=self.target_R.copy(),
                            gripper=self.gripper, clutched=self.anchor is not None, precision=precision, stale=stale,
                            discard=discard, phase_mark=mark, raw=controller_vector(msg))
