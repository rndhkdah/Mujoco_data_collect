"""Web server: WebSocket forwarding, frame/status push, disconnect safety (no env needed)."""
import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("cv2")

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from fsim.server import Shared, build_app, make_publisher, panel_frame  # noqa: E402
from fsim.spec import get_task  # noqa: E402


def fake_obs(size=64):
    rng = np.random.default_rng(0)
    return {k: rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            for k in ("front_image", "robot0_eye_in_hand_image", "rear_image")}


def test_panel_frame_layout():
    img = panel_frame(fake_obs(), "RECORD", {"steps": 12, "clutched": True, "phase_name": "pick_leg"}, "one_leg", tile=100)
    assert img.shape == (56 + 100, 3 * 100 + 2 * 4, 3) and img.dtype == np.uint8


def test_websocket_roundtrip_and_disconnect_clears_controller():
    async def run():
        shared = Shared()
        task = get_task("one_leg")
        make_publisher(shared, task, [])("READY", fake_obs(), {"flash": "NEW LAYOUT"})
        client = TestClient(TestServer(build_app(shared)))
        await client.start_server()
        try:
            ws = await client.ws_connect("/ws")
            got_frame = got_status = False
            for _ in range(10):
                m = await asyncio.wait_for(ws.receive(), timeout=2.0)
                if m.type.name == "BINARY":
                    got_frame = m.data[:2] == b"\xff\xd8"        # JPEG
                elif m.type.name == "TEXT":
                    s = json.loads(m.data)
                    got_status = s["state"] == "READY" and s["flash"] == "NEW LAYOUT"
                if got_frame and got_status:
                    break
            assert got_frame and got_status
            await ws.send_str(json.dumps({"type": "controller", "right": {"grip": 1.0}}))
            for _ in range(50):
                await asyncio.sleep(0.02)
                if shared.get_controller()[0] is not None:
                    break
            msg, age = shared.get_controller()
            assert msg["right"]["grip"] == 1.0 and age < 1.0
            await ws.close()
            for _ in range(50):
                await asyncio.sleep(0.02)
                if shared.get_controller()[0] is None:
                    break
            assert shared.get_controller()[0] is None
        finally:
            await client.close()

    asyncio.run(run())
