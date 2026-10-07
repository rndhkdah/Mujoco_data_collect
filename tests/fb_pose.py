"""FurnitureBench's pose helpers (Euler xyz as R = Rz Ry Rx), re-implemented for tests that compare
our specs with the design's numbers converted into our frames."""
import numpy as np

from fsim.geometry import make_T, rot_x, rot_y, rot_z


def get_mat(pos, angles) -> np.ndarray:
    x, y, z = angles
    return make_T(pos, rot_z(z) @ rot_y(y) @ rot_x(x))


def to_ours(T_fb_rel: np.ndarray, fb_T_parent: np.ndarray, fb_T_child: np.ndarray) -> np.ndarray:
    """A design parent->child pose expressed in our part frames (fb_T_*: our frame in theirs)."""
    return np.linalg.inv(fb_T_parent) @ T_fb_rel @ fb_T_child


def rot_cos_min(R1, R2) -> float:
    return float(min(np.dot(R1[:, i], R2[:, i]) for i in range(3)))
