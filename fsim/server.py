#!/usr/bin/env python3
"""VR collection server (one process, like the teammate's tool).

A collect thread runs the env and the episode loop (fsim.collect); aiohttp on 127.0.0.1 serves the
WebXR page and a WebSocket: controller JSON in; JSON status out, plus
  --view 3d (default): body-pose frames + wrist-camera JPEGs; the page renders the scene from
                       /scene.glb + /manifest.json (fsim.vr.scene_export, static/vr3d.html)
  --view panel:        JPEG camera panel frames (static/panel.html)
The Quest reaches it through `adb reverse tcp:<port> tcp:<port>` (see stack.sh).

    MUJOCO_GL=egl python -m fsim.server --task one_leg --randomness low --num-demos 50 --operator me
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from aiohttp import web  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
PANEL_CAMERAS = ("front_image", "robot0_eye_in_hand_image", "rear_image")
SCENE_EXCLUDE = r"^(floor|wall_)"        # robosuite arena dressing: the headset shows the room or a plain backdrop


class Shared:
    def __init__(self, view: str = "panel"):
        self.lock = threading.Lock()
        self.view = view                   # "panel" (JPEG panel) or "3d" (pose frames + scene GLB)
        self.controller = None
        self.controller_time = 0.0
        self.frame = None                  # latest binary frame: JPEG panel, or pose frame in 3d mode
        self.frame_id = 0
        self.wrist = None                  # 3d mode: wrist-camera JPEG with a 4-byte type header
        self.wrist_id = 0
        self.scene_glb = None
        self.scene_glb_gz = None
        self.manifest = None
        self.status = {"state": "STARTING"}
        self.running = True

    def get_controller(self):
        with self.lock:
            msg, t = self.controller, self.controller_time
        return msg, (time.monotonic() - t) if msg is not None else 1e9

    def put_controller(self, msg):
        with self.lock:
            self.controller, self.controller_time = msg, time.monotonic()

    def clear_controller(self):
        with self.lock:
            self.controller = None


SHARED = web.AppKey("shared", Shared) if hasattr(web, "AppKey") else "shared"
MSG_WRIST_HEADER = bytes([3, 0, 0, 0])      # fsim.vr.scene_export.MSG_WRIST_JPEG + padding


def panel_frame(obs, state: str, info: dict, task_name: str, tile: int = 320) -> np.ndarray:
    """[front | wrist | rear] tiles under a status banner (robosuite frames are flipped for display)."""
    tiles = []
    for key in PANEL_CAMERAS:
        if key in obs:
            img = cv2.cvtColor(np.ascontiguousarray(obs[key][::-1]), cv2.COLOR_RGB2BGR)
            tiles.append(cv2.resize(img, (tile, tile), interpolation=cv2.INTER_LINEAR))
    if not tiles:
        tiles = [np.zeros((tile, tile, 3), np.uint8)]
    gap = np.zeros((tile, 4, 3), np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, gap, t])
    banner = np.full((56, row.shape[1], 3), 20, np.uint8)
    phase = info.get("phase_name") or ""
    line1 = f"[{state}] {task_name}  {phase}"
    flags = []
    if info.get("clutched"):
        flags.append("CLUTCH")
    if info.get("precision"):
        flags.append("PRECISION")
    if info.get("stale"):
        flags.append("NO CONTROLLER")
    line2 = f"step {info.get('steps', 0)}  saved {info.get('saved', 0)}  {' '.join(flags)}"
    color = {"RECORD": (0, 255, 0), "READY": (0, 220, 255), "SAVED": (0, 200, 0)}.get(state, (0, 0, 230))
    cv2.putText(banner, line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(banner, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    img = np.vstack([banner, row])
    if info.get("flash"):
        cv2.putText(img, info["flash"], (img.shape[1] // 2 - 90, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (0, 200, 0) if info["flash"] in ("SAVED", "SUCCESS") else (0, 0, 230), 4)
    return img


def make_publisher(shared: Shared, task, collector_ref: list, env=None, packer=None):
    """Per-step publisher: panel JPEG (panel view) or pose frame + periodic wrist JPEG (3d view)."""
    counter = [0]

    def publish(state, obs, info):
        info = dict(info)
        if "phase" in info and info["phase"] is not None and info["phase"] < len(task.phases):
            info["phase_name"] = task.phases[info["phase"]]
        elif "phase" in info:
            info["phase_name"] = "done"
        info["saved"] = collector_ref[0].saved if collector_ref else 0
        status = {"type": "status", "state": state, **{k: info[k] for k in
                  ("steps", "saved", "phase_name", "clutched", "precision", "stale", "flash", "path") if k in info}}
        counter[0] += 1
        frame = wrist = None
        if shared.view == "3d" and packer is not None:
            flags = [float(bool(info.get(k))) for k in ("clutched", "precision", "stale")] + [float(info.get("gripper", -1.0))]
            frame = packer.pack(env.sim.data._data, target=info.get("target"), flags=flags)
            cam = obs.get("robot0_eye_in_hand_image")
            if cam is not None and counter[0] % 2 == 0:
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(cam[::-1]), cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    wrist = MSG_WRIST_HEADER + jpg.tobytes()
        else:
            ok, jpg = cv2.imencode(".jpg", panel_frame(obs, state, info, task.name), [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = jpg.tobytes() if ok else None
        with shared.lock:
            if frame is not None:
                shared.frame, shared.frame_id = frame, shared.frame_id + 1
            if wrist is not None:
                shared.wrist, shared.wrist_id = wrist, shared.wrist_id + 1
            shared.status = status
    return publish


async def index(request):
    shared: Shared = request.app[SHARED]
    return web.FileResponse(os.path.join(STATIC, "vr3d.html" if shared.view == "3d" else "panel.html"))


async def scene_glb(request):
    shared: Shared = request.app[SHARED]
    if shared.scene_glb is None:
        raise web.HTTPNotFound()
    headers = {"Cache-Control": "no-store"}
    if shared.scene_glb_gz is not None and "gzip" in request.headers.get("Accept-Encoding", ""):
        headers["Content-Encoding"] = "gzip"
        return web.Response(body=shared.scene_glb_gz, content_type="model/gltf-binary", headers=headers)
    return web.Response(body=shared.scene_glb, content_type="model/gltf-binary", headers=headers)


async def manifest(request):
    shared: Shared = request.app[SHARED]
    if shared.manifest is None:
        raise web.HTTPNotFound()
    return web.json_response(shared.manifest)


async def ws_handler(request):
    shared: Shared = request.app[SHARED]
    ws = web.WebSocketResponse(max_msg_size=0, heartbeat=10.0)
    await ws.prepare(request)
    print("[ws] client connected", flush=True)

    async def sender():
        last_frame, last_wrist, last_status = -1, -1, None
        while not ws.closed:
            with shared.lock:
                fid, frame, status = shared.frame_id, shared.frame, shared.status
                wid, wrist = shared.wrist_id, shared.wrist
            try:
                if frame is not None and fid != last_frame:
                    last_frame = fid
                    await ws.send_bytes(frame)
                if wrist is not None and wid != last_wrist:
                    last_wrist = wid
                    await ws.send_bytes(wrist)
                if status is not last_status:
                    last_status = status
                    await ws.send_str(json.dumps(status))
            except Exception:
                break
            await asyncio.sleep(1.0 / 60.0)

    task = asyncio.create_task(sender())
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                if data.get("type") == "controller":
                    shared.put_controller(data)
    finally:
        task.cancel()
        shared.clear_controller()          # a dropped connection must not leave the clutch latched
        print("[ws] client disconnected", flush=True)
    return ws


@web.middleware
async def revalidate_cache(request, handler):
    """Without Cache-Control, browsers cache pages heuristically, and the headset would keep running an
    old client after an update. "no-cache" makes them revalidate; ETags keep that cheap."""
    resp = await handler(request)
    if not isinstance(resp, web.WebSocketResponse):
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


def build_app(shared: Shared) -> web.Application:
    app = web.Application(middlewares=[revalidate_cache])
    app[SHARED] = shared
    app.router.add_get("/", index)
    app.router.add_get("/scene.glb", scene_glb)
    app.router.add_get("/manifest.json", manifest)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", STATIC)
    return app


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="one_leg")
    ap.add_argument("--randomness", default="low", choices=["none", "low", "med", "high"])
    ap.add_argument("--num-demos", type=int, default=50)
    ap.add_argument("--out-dir", default=os.path.expanduser(os.environ.get("FSIM_DATA_DIR", "~/fsim_dataset")))
    ap.add_argument("--operator", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--camera-size", type=int, default=224)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--settle-capture", type=int, default=12)
    ap.add_argument("--view", default="3d", choices=["3d", "panel"],
                    help="3d: stereo WebXR scene from streamed poses; panel: the teammate-style camera panel")
    ap.add_argument("--record-operator", action="store_true",
                    help="also record the scene as seen from the operator's headset (obs/operator_rgb, "
                         "obs/operator_head_pose)")
    ap.add_argument("--auto-screw", action="store_true",
                    help="one_leg: once the leg stands released in a hole, the scripted expert regrasps it and "
                         "screws it in (recorded with the episode; B discards)")
    ap.add_argument("--autopilot", action="store_true",
                    help="the scripted expert (one_leg) drives the arm in real time instead of the VR operator, "
                         "to watch in the headset or a browser; nothing is saved")
    return ap.parse_args(argv)


def start_collect(args, shared: Shared) -> tuple[threading.Thread, list]:
    """Start the collect thread; returns (thread, [collector]) once the env (and 3d scene) is ready.

    Everything that touches MuJoCo, including the EGL offscreen context robosuite creates with the
    env, lives in that thread: a context only renders in the thread where it is current, and camera
    images (recorded and streamed) come out as noise when the env is built in another thread.
    """
    from fsim.collect import CollectConfig, Collector
    from fsim.env.furniture_env import make_env

    collector_ref: list = []
    ready, failed = threading.Event(), []

    def run():
        try:
            env = make_env(task=args.task, randomness=args.randomness, seed=args.seed, camera_size=args.camera_size)
            packer = None
            if args.view == "3d":
                from fsim.vr.scene_export import PosePacker, export_scene
                glb, manifest = export_scene(env.sim.model._model, groups=(1,), exclude=SCENE_EXCLUDE)
                manifest["task"] = env.task.name
                manifest["workspace_center"] = [float(v) for v in np.asarray(env.task.scene.W_origin) + np.array([0.2, 0.0, 0.0])]
                manifest["highlight"] = sorted({f"{t.child}_main" for t in env.task.targets})
                manifest["robot_base"] = [float(v) for v in env.sim.data.get_body_xpos(env.robots[0].robot_model.root_body)]
                shared.scene_glb, shared.scene_glb_gz, shared.manifest = glb, gzip.compress(glb, 6), manifest
                packer = PosePacker(manifest)
                print(f"[server] 3d scene: {len(manifest['bodies'])} bodies, {len(glb) / 1e6:.1f} MB GLB", flush=True)
            collector = Collector(env, CollectConfig(out_dir=args.out_dir, operator=args.operator, max_steps=args.max_steps,
                                                     settle_capture=args.settle_capture,
                                                     record_operator=args.record_operator, auto_screw=args.auto_screw),
                                  shared.get_controller, make_publisher(shared, env.task, collector_ref, env, packer))
            collector_ref.append(collector)
        except BaseException as e:
            failed.append(e)
            raise
        finally:
            ready.set()
        try:
            if args.autopilot:
                from fsim.env.expert import OneLegExpert
                while collector.running:
                    outcome, _ = collector.record_scripted(lambda step: OneLegExpert(env, step_fn=step).run(), save=False)
                    print(f"[autopilot] {outcome}", flush=True)
                    time.sleep(2.0)                # leave the result on screen before the next layout
            else:
                collector.run(args.num_demos)
        finally:
            with shared.lock:
                shared.status = {"type": "status", "state": "DONE", "saved": collector.saved}
            env.close()                    # frees the EGL context in the thread that owns it

    thread = threading.Thread(target=run, name="collect", daemon=True)
    thread.start()
    ready.wait()
    if failed:
        raise RuntimeError(f"env setup failed: {failed[0]!r}") from failed[0]
    return thread, collector_ref


def stop_collect(thread: threading.Thread, collector_ref: list, timeout: float = 10.0):
    """Stop the episode loop and wait for the collect thread to release MuJoCo/EGL: tearing them
    down at interpreter exit under a running render segfaults."""
    if collector_ref:
        collector_ref[0].running = False
    thread.join(timeout=timeout)


def main(argv=None):
    args = parse_args(argv)
    if (args.autopilot or args.auto_screw) and args.task != "one_leg":
        raise SystemExit("[server] --autopilot and --auto-screw only know the one_leg task")
    shared = Shared(view=args.view)
    try:
        thread, collector_ref = start_collect(args, shared)
    except RuntimeError as e:
        raise SystemExit(f"[server] {e}")
    print(f"[server] http://localhost:{args.port}  task {args.task} ({args.randomness}) -> {args.out_dir}", flush=True)
    try:
        web.run_app(build_app(shared), host="127.0.0.1", port=args.port, print=None)
    finally:
        stop_collect(thread, collector_ref)


if __name__ == "__main__":
    main()
