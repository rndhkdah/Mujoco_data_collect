#!/usr/bin/env python3
"""VR collection server (one process, like the teammate's tool).

A collect thread runs the env and the episode loop (fsim.collect); aiohttp on 127.0.0.1 serves the
WebXR page and a WebSocket: controller JSON in; JPEG panel frames + JSON status out. The Quest
reaches it through `adb reverse tcp:<port> tcp:<port>` (see stack.sh).

    MUJOCO_GL=egl python -m fsim.server --task one_leg --randomness low --num-demos 50 --operator me
"""
from __future__ import annotations

import argparse
import asyncio
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


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.controller = None
        self.controller_time = 0.0
        self.frame = None
        self.frame_id = 0
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
                    (0, 200, 0) if info["flash"] == "SAVED" else (0, 0, 230), 4)
    return img


def make_publisher(shared: Shared, task, collector_ref: list):
    def publish(state, obs, info):
        info = dict(info)
        if "phase" in info and info["phase"] is not None and info["phase"] < len(task.phases):
            info["phase_name"] = task.phases[info["phase"]]
        elif "phase" in info:
            info["phase_name"] = "done"
        info["saved"] = collector_ref[0].saved if collector_ref else 0
        ok, jpg = cv2.imencode(".jpg", panel_frame(obs, state, info, task.name), [cv2.IMWRITE_JPEG_QUALITY, 85])
        status = {"type": "status", "state": state, **{k: info[k] for k in
                  ("steps", "saved", "phase_name", "clutched", "precision", "stale", "flash", "path") if k in info}}
        with shared.lock:
            if ok:
                shared.frame, shared.frame_id = jpg.tobytes(), shared.frame_id + 1
            shared.status = status
    return publish


async def index(request):
    return web.FileResponse(os.path.join(STATIC, "panel.html"))


async def ws_handler(request):
    shared: Shared = request.app[SHARED]
    ws = web.WebSocketResponse(max_msg_size=0, heartbeat=10.0)
    await ws.prepare(request)
    print("[ws] client connected", flush=True)

    async def sender():
        last_frame, last_status = -1, None
        while not ws.closed:
            with shared.lock:
                fid, frame, status = shared.frame_id, shared.frame, shared.status
            try:
                if frame is not None and fid != last_frame:
                    last_frame = fid
                    await ws.send_bytes(frame)
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


def build_app(shared: Shared) -> web.Application:
    app = web.Application()
    app[SHARED] = shared
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", STATIC)
    return app


def main(argv=None):
    from fsim.collect import CollectConfig, Collector
    from fsim.env.furniture_env import make_env

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
    args = ap.parse_args(argv)

    shared = Shared()
    env = make_env(task=args.task, randomness=args.randomness, seed=args.seed, camera_size=args.camera_size)
    collector_ref: list = []
    collector = Collector(env, CollectConfig(out_dir=args.out_dir, operator=args.operator, max_steps=args.max_steps,
                                             settle_capture=args.settle_capture),
                          shared.get_controller, make_publisher(shared, env.task, collector_ref))
    collector_ref.append(collector)

    def run():
        try:
            collector.run(args.num_demos)
        finally:
            with shared.lock:
                shared.status = {"type": "status", "state": "DONE", "saved": collector.saved}
    threading.Thread(target=run, daemon=True).start()
    print(f"[server] http://localhost:{args.port}  task {args.task} ({args.randomness}) -> {args.out_dir}", flush=True)
    try:
        web.run_app(build_app(shared), host="127.0.0.1", port=args.port, print=None)
    finally:
        collector.running = False


if __name__ == "__main__":
    main()
