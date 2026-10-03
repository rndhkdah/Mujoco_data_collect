"""The real server's collect thread (Linux + robosuite): env, EGL context and rendering share one thread."""
import os
import time

import numpy as np
import pytest

pytest.importorskip("robosuite")
pytest.importorskip("aiohttp")
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402

from fsim.server import Shared, parse_args, start_collect, stop_collect  # noqa: E402
from fsim.vr.scene_export import MSG_POSES  # noqa: E402


def roughness(img: np.ndarray) -> float:
    a = img.astype(np.int32)
    return float(np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2


def test_collect_thread_streams_rendered_camera_images(tmp_path):
    args = parse_args(["--camera-size", "64", "--out-dir", str(tmp_path), "--view", "3d", "--seed", "0"])
    shared = Shared(view=args.view)
    thread, collector_ref = start_collect(args, shared)
    try:
        assert shared.manifest["highlight"] == ["square_table_leg4_main"] and len(shared.manifest["robot_base"]) == 3
        frame = wrist = None
        deadline = time.monotonic() + 60.0
        while (frame is None or wrist is None) and time.monotonic() < deadline:
            with shared.lock:
                frame, wrist = shared.frame, shared.wrist
            time.sleep(0.05)
        assert frame is not None and frame[0] == MSG_POSES and wrist is not None
        img = cv2.imdecode(np.frombuffer(wrist[4:], np.uint8), cv2.IMREAD_COLOR)
        # Rendering through a context that is not current in the calling thread reads back noise
        # (roughness ~60); a rendered scene is mostly smooth (~8).
        assert img.shape == (64, 64, 3) and roughness(img) < 25.0
    finally:
        stop_collect(thread, collector_ref)
    assert not thread.is_alive()
