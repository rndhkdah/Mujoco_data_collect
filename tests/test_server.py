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


def test_3d_view_serves_scene_and_streams_pose_frames():
    import gzip
    from types import SimpleNamespace

    import mujoco

    from fsim.models.scene import build_scene_xml
    from fsim.server import MSG_WRIST_HEADER
    from fsim.vr.scene_export import MSG_POSES, PosePacker, export_scene, unpack_poses

    async def run():
        task = get_task("one_leg")
        m = mujoco.MjModel.from_xml_string(build_scene_xml(task, visual_meshes=True))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        shared = Shared(view="3d")
        shared.scene_glb, shared.manifest = export_scene(m, groups=(0, 1))
        shared.scene_glb_gz = gzip.compress(shared.scene_glb)
        packer = PosePacker(shared.manifest)
        env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(_data=d)))
        publish = make_publisher(shared, task, [], env, packer)
        target = np.r_[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
        for _ in range(2):                       # the wrist JPEG goes out every 2nd step
            publish("RECORD", fake_obs(), {"target": target, "gripper": 1.0, "clutched": True, "steps": 3})
        n = len(shared.manifest["bodies"])
        client = TestClient(TestServer(build_app(shared)))
        await client.start_server()
        try:
            r = await client.get("/")
            assert "Furniture Teleop 3D" in await r.text()
            assert r.headers["Cache-Control"] == "no-cache"        # a headset must not run a stale client
            r = await client.get("/manifest.json")
            assert (await r.json())["bodies"] == shared.manifest["bodies"]
            r = await client.get("/scene.glb", headers={"Accept-Encoding": "gzip"})
            assert r.status == 200 and (await r.read())[:4] == b"glTF"     # gzip is undone by the client
            r = await client.get("/static/vendor/three/three.module.js")
            assert r.status == 200 and "javascript" in r.headers["Content-Type"]
            assert r.headers["Cache-Control"] == "no-cache"
            ws = await client.ws_connect("/ws")
            pose = wrist = status = None
            for _ in range(10):
                msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                if msg.type.name == "BINARY" and msg.data[0] == MSG_POSES:
                    pose = unpack_poses(msg.data, n)
                elif msg.type.name == "BINARY":
                    wrist = msg.data
                elif msg.type.name == "TEXT":
                    status = json.loads(msg.data)
                if pose is not None and wrist is not None and status is not None:
                    break
            assert pose["seq"] == 2 and np.allclose(pose["target"], target) and np.allclose(pose["flags"], [1, 0, 0, 1])
            assert wrist[:4] == MSG_WRIST_HEADER and wrist[4:6] == b"\xff\xd8"
            assert status["state"] == "RECORD" and status["clutched"] is True
            await ws.close()
        finally:
            await client.close()

    asyncio.run(run())
