"""Damped-least-squares IK for the robosuite arm (home pose, reach checks, scripted demos).

Works on a private MjData copy, so the env state is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from fsim.geometry import mat_to_quat, quat_to_mat


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float      # m
    rot_err: float      # rad
    ok: bool
    iters: int


def _rot_error(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking R_cur to R_tgt."""
    q = mat_to_quat(R_tgt @ R_cur.T)
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    s = np.sqrt(max(1.0 - q[0] * q[0], 1e-12))
    return q[1:] / s * angle if angle > 1e-9 else np.zeros(3)


def solve_ik(model: mujoco.MjModel, qpos: np.ndarray, site: str, joint_qadr, joint_dadr, target_pos,
             target_R=None, q0=None, iters: int = 400, damping: float = 1e-4, pos_tol: float = 1e-4,
             rot_tol: float = 1e-3, q_rest=None, rest_gain: float = 0.05) -> IKResult:
    """Solve for the arm joints (qadr/dadr index lists) so `site` reaches target_pos (and target_R).

    q_rest: optional posture the redundant (null-space) motion is pulled toward.
    """
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    qadr, dadr = np.asarray(joint_qadr), np.asarray(joint_dadr)
    if q0 is not None:
        d.qpos[qadr] = q0
    jac_p, jac_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    jnt = [int(np.flatnonzero(model.jnt_qposadr == a)[0]) for a in qadr]
    lo = np.where(model.jnt_limited[jnt], model.jnt_range[jnt, 0], -np.inf)
    hi = np.where(model.jnt_limited[jnt], model.jnt_range[jnt, 1], np.inf)
    target_pos = np.asarray(target_pos, float)
    for it in range(iters):
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        e_p = target_pos - d.site_xpos[sid]
        e_r = np.zeros(3) if target_R is None else _rot_error(d.site_xmat[sid].reshape(3, 3), target_R)
        converged = np.linalg.norm(e_p) < pos_tol and np.linalg.norm(e_r) < rot_tol
        mujoco.mj_jacSite(model, d, jac_p, jac_r, sid)
        J = jac_p[:, dadr] if target_R is None else np.vstack([jac_p[:, dadr], jac_r[:, dadr]])
        e = e_p if target_R is None else np.concatenate([e_p, e_r])
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(J.shape[0]))
        dq = J_pinv @ e
        null = np.zeros(len(dadr))
        if q_rest is not None:
            null = (np.eye(len(dadr)) - J_pinv @ J) @ (rest_gain * (np.asarray(q_rest) - d.qpos[qadr]))
        if converged and np.linalg.norm(null) < 1e-5:
            break
        d.qpos[qadr] = np.clip(d.qpos[qadr] + dq + null, lo, hi)
    mujoco.mj_kinematics(model, d)
    e_p = np.linalg.norm(target_pos - d.site_xpos[sid])
    e_r = 0.0 if target_R is None else float(np.linalg.norm(_rot_error(d.site_xmat[sid].reshape(3, 3), target_R)))
    return IKResult(d.qpos[qadr].copy(), float(e_p), e_r, bool(e_p < 2e-3 and e_r < np.radians(2)), it + 1)


def gripper_down(yaw: float = 0.0) -> np.ndarray:
    """Grip-site orientation with the approach axis (site z) pointing down; yaw rotates about world z.

    The Robotiq fingers open along the grip site's x axis. At yaw 0 that axis points along world
    +y, so the fingers close across a part lying along world x.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def quat_of(R: np.ndarray) -> np.ndarray:
    return mat_to_quat(R)


__all__ = ["IKResult", "solve_ik", "gripper_down", "quat_of", "quat_to_mat"]
