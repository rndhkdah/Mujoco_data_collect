import math

import numpy as np

from fsim.geometry import (inv_T, look_at_xyaxes, make_T, mat_to_quat, orthonormalize, quat_to_mat,
                           rot_x, rot_y, rot_z, wrap_angle, wxyz_to_xyzw, xyzw_to_wxyz)


def test_quat_mat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        assert np.allclose(mat_to_quat(quat_to_mat(q)), q, atol=1e-9)


def test_rotations_and_inverse():
    R = rot_z(0.3) @ rot_y(-1.1) @ rot_x(2.0)
    assert np.allclose(R @ R.T, np.eye(3))
    T = make_T((0.1, -0.2, 0.3), R)
    assert np.allclose(inv_T(T) @ T, np.eye(4))


def test_orthonormalize_projects_noisy_matrix():
    R = rot_z(0.4) + 1e-3 * np.ones((3, 3))
    Q = orthonormalize(R)
    assert np.allclose(Q @ Q.T, np.eye(3)) and np.linalg.det(Q) > 0


def test_wrap_angle_range():
    for a in np.linspace(-10, 10, 101):
        w = wrap_angle(a)
        assert -math.pi < w <= math.pi
        assert math.isclose(math.cos(w), math.cos(a), abs_tol=1e-9)
    assert wrap_angle(3 * math.pi) == math.pi


def test_quat_order_conversions():
    q = np.array([0.1, 0.2, 0.3, 0.9])
    assert np.allclose(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)


def test_look_at_camera_axes():
    pos, target = (0.9, 0.0, 0.65), (-1.0, 0.0, 0.3)
    ax = look_at_xyaxes(pos, target)
    x, y = ax[:3], ax[3:]
    z = np.cross(x, y)            # MuJoCo cameras look along -z
    f = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    assert np.allclose(-z, f, atol=1e-9)
    assert abs(np.dot(x, y)) < 1e-12 and y[2] > 0
