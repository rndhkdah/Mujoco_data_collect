"""Teleop mapping: clutch, frames, precision, sticks, staleness, buttons (pure numpy)."""
import math

import numpy as np

from fsim.geometry import mat_to_quat, rot_x, rot_z, wxyz_to_xyzw
from fsim.teleop.mapping import XR_TO_ROBOT, TeleopConfig, TeleopMapper, controller_vector, head_pose, rotvec

EE_POS = np.array([0.15, 0.0, 0.585])
EE_R = np.diag([1.0, -1.0, -1.0])      # gripper pointing down


def msg(pos=(0, 0, 0), R=np.eye(3), grip=1.0, trigger=0.0, a=False, b=False, stick=(0.0, 0.0), x=False,
        frame="robot", tracked=True):
    return {"type": "controller", "frame": frame,
            "right": {"pos": list(pos), "quat": list(wxyz_to_xyzw(mat_to_quat(R))), "grip": grip, "trigger": trigger,
                      "a": a, "b": b, "stick": list(stick), "tracked": tracked},
            "left": {"x": x, "y": False, "stick": [0.0, 0.0]}}


def mapper(speed=1.0):
    m = TeleopMapper(TeleopConfig(speed=speed), control_hz=20)
    m.reset(EE_POS, EE_R)
    return m


def test_clutch_follows_hand_translation_and_rotation():
    m = mapper()
    m.update(msg((0.0, 0.0, 0.0)), 0.0, EE_POS, EE_R)                      # anchor
    out = m.update(msg((0.05, -0.02, 0.01), rot_z(0.3)), 0.0, EE_POS, EE_R)
    assert np.allclose(out.target_pos, EE_POS + [0.05, -0.02, 0.01])
    assert np.allclose(out.target_R, rot_z(0.3) @ EE_R)
    assert out.clutched and out.action[0] > 0 and out.action[5] > 0       # moves +x, rotates +z


def test_xr_frame_is_mapped_to_robot_frame():
    m = mapper()
    m.update(msg((0, 0, 0), frame="xr"), 0.0, EE_POS, EE_R)
    out = m.update(msg((0.0, 0.0, -0.05), frame="xr"), 0.0, EE_POS, EE_R)    # hand forward (-z in WebXR)
    assert np.allclose(out.target_pos - EE_POS, [0.05, 0.0, 0.0])
    assert np.allclose(XR_TO_ROBOT @ [0, 1, 0], [0, 0, 1])                     # up stays up


def test_release_holds_and_reanchors_without_jump():
    m = mapper()
    m.update(msg((0, 0, 0)), 0.0, EE_POS, EE_R)
    m.update(msg((0.03, 0, 0)), 0.0, EE_POS, EE_R)
    out = m.update(msg((0.03, 0, 0), grip=0.0), 0.0, EE_POS, EE_R)            # released: hold the arm
    assert not out.clutched and np.allclose(out.target_pos, EE_POS) and np.allclose(out.action[:6], 0)
    out = m.update(msg((0.10, 0, 0)), 0.0, EE_POS, EE_R)                      # re-press far away: new anchor
    assert np.allclose(out.target_pos, EE_POS)


def test_precision_mode_scales_and_reanchors():
    m = mapper()
    m.update(msg((0, 0, 0)), 0.0, EE_POS, EE_R)
    m.update(msg((0.0, 0, 0), a=True), 0.0, EE_POS, EE_R)                    # entering precision re-anchors
    out = m.update(msg((0.10, 0, 0), a=True), 0.0, EE_POS, EE_R)
    assert np.allclose(out.target_pos - EE_POS, [0.02, 0, 0]) and out.precision


def test_stick_twists_about_tool_axis_and_pushes_along_it():
    m = mapper()
    out = None
    for _ in range(4):                                                          # no clutch, stick right (24 deg < lead cap)
        out = m.update(msg(grip=0.0, stick=(1.0, 0.0)), 0.0, EE_POS, EE_R)
    twist = rotvec(out.target_R @ EE_R.T)
    assert np.allclose(twist / np.linalg.norm(twist), EE_R[:, 2], atol=1e-6)   # about the tool (down) axis
    assert math.isclose(np.linalg.norm(twist), 4 * math.radians(120) / 20, rel_tol=1e-6)
    m = mapper()
    out = m.update(msg(grip=0.0, stick=(0.0, -1.0)), 0.0, EE_POS, EE_R)        # stick forward: push down the tool
    assert np.allclose(out.target_pos - EE_POS, EE_R[:, 2] * 0.02 / 20)


def test_twist_lead_is_limited():
    m = mapper()
    for _ in range(100):
        out = m.update(msg(grip=0.0, stick=(1.0, 0.0)), 0.0, EE_POS, EE_R)     # arm never turns
    assert np.linalg.norm(rotvec(out.target_R @ EE_R.T)) < math.radians(27)


def test_stale_or_untracked_input_releases_the_clutch():
    m = mapper()
    m.update(msg((0, 0, 0)), 0.0, EE_POS, EE_R)
    out = m.update(msg((0.05, 0, 0)), 0.5, EE_POS, EE_R)                       # 0.5 s old
    assert out.stale and not out.clutched and np.allclose(out.target_pos, EE_POS)
    out = m.update(msg((0.05, 0, 0), tracked=False), 0.0, EE_POS, EE_R)
    assert out.stale and np.allclose(out.action[:6], 0)


def test_buttons_trigger_on_press_only():
    m = mapper()
    assert m.update(msg(b=True, grip=0.0), 0.0, EE_POS, EE_R).discard
    assert not m.update(msg(b=True, grip=0.0), 0.0, EE_POS, EE_R).discard
    assert m.update(msg(x=True, grip=0.0), 0.0, EE_POS, EE_R).phase_mark
    out = m.update(msg(trigger=0.9, grip=0.0), 0.0, EE_POS, EE_R)
    assert out.gripper == 1.0 and out.action[6] == 1.0 and out.action_fb[7] == 1.0


def test_workspace_clamp_keeps_target_above_table():
    m = mapper()
    m.update(msg((0, 0, 0)), 0.0, EE_POS, EE_R)
    out = m.update(msg((0, 0, -1.0)), 0.0, EE_POS, EE_R)
    assert out.target_pos[2] >= TeleopConfig().workspace_min[2]


def test_fb_action_convention():
    m = mapper()
    m.update(msg((0, 0, 0)), 0.0, EE_POS, EE_R)
    out = m.update(msg((0.01, 0, 0), rot_x(0.2)), 0.0, EE_POS, EE_R)
    assert np.allclose(out.action_fb[:3], [0.01, 0, 0])
    q = out.action_fb[3:7]                                                       # xyzw, EE frame
    assert math.isclose(np.linalg.norm(q), 1.0, rel_tol=1e-6)
    assert controller_vector(msg()).shape == (18,)


def test_head_pose_reads_the_headset_camera_frame():
    assert head_pose(None) is None and head_pose(msg()) is None          # no headset in the message
    R = rot_z(0.4) @ rot_x(1.1)
    m = dict(msg(), head={"pos": [0.1, 0.2, 1.3], "quat": list(wxyz_to_xyzw(mat_to_quat(R)))})
    p, Rh = head_pose(m)
    assert np.allclose(p, [0.1, 0.2, 1.3]) and np.allclose(Rh, R, atol=1e-9)
    # raw XR frame: a level head looking along XR -z looks along the robot's +x, with robot z up
    p, Rh = head_pose({"frame": "xr", "head": {"pos": [0, 1.5, 0], "quat": [0, 0, 0, 1]}})
    assert np.allclose(p, [0, 0, 1.5]) and np.allclose(-Rh[:, 2], [1, 0, 0]) and np.allclose(Rh[:, 1], [0, 0, 1])


def test_global_speed_scales_hand_following_and_sticks():
    assert TeleopConfig().speed == 0.7
    m = mapper(speed=0.7)
    m.update(msg((0.0, 0.0, 0.0)), 0.0, EE_POS, EE_R)                      # anchor
    out = m.update(msg((0.1, 0.0, 0.0), rot_z(0.3)), 0.0, EE_POS, EE_R)
    assert np.allclose(out.target_pos - EE_POS, [0.07, 0, 0])
    assert np.isclose(np.linalg.norm(rotvec(out.target_R @ EE_R.T)), 0.21)
    m = mapper(speed=0.7)
    m.update(msg(grip=0.0, stick=(1.0, 0.0)), 0.0, EE_POS, EE_R)           # one step of full twist
    out = m.update(msg(grip=0.0, stick=(1.0, 0.0)), 0.0, EE_POS, EE_R)
    assert np.isclose(np.linalg.norm(rotvec(out.target_R @ EE_R.T)), 2 * 0.7 * TeleopConfig().twist_rate / 20)
