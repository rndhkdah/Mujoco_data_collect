#!/usr/bin/env python3
import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import json
import time
import argparse
import threading

import numpy as np
import cv2
import h5py
from scipy.spatial.transform import Rotation as R
from aiohttp import web

from kinova_circle_env import make_circle_env
from circle_schedule import (layered_grid_inits, z_layers_from, spans_from,
                             build_circle_schedule, spec_key,
                             voxel_inits, build_voxel_schedule)
from hdf5_writer import DemoWriter

HERE = os.path.dirname(os.path.abspath(__file__))
AXIS_FACING = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.controller = None
        self.frame = None
        self.frame_id = 0
        self.status = "starting"
        self.running = True


shared = Shared()


def _w2p(x, y, W, H, xr, yr):
    return (int((x - xr[0]) / (xr[1] - xr[0]) * W),
            int((1.0 - (y - yr[0]) / (yr[1] - yr[0])) * H))


def draw_birdeye_circle(size, cellA_xy, cellB_xy, cell_names, basket_xy, inits,
                        obj_names, spec, ee_xy, phase, done_n, total_n):
    W = H = size
    xr, yr = (-0.35, 0.35), (-0.30, 0.42)
    img = np.full((H, W, 3), 30, np.uint8)
    li = int(spec["obj_left_idx"]); tc = int(spec["target_cell"])
    a_name = obj_names[li]; b_name = obj_names[1 - li]
    cv2.circle(img, _w2p(-0.30, 0.0, W, H, xr, yr), 7, (0, 0, 255), -1)
    cv2.putText(img, "ROBOT", _w2p(-0.345, -0.04, W, H, xr, yr),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
    bx, by = basket_xy
    cv2.rectangle(img, _w2p(bx - 0.06, by + 0.06, W, H, xr, yr),
                  _w2p(bx + 0.06, by - 0.06, W, H, xr, yr), (255, 120, 0), 2)
    cv2.putText(img, "basket", _w2p(bx - 0.05, by - 0.08, W, H, xr, yr),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 160, 80), 1)
    for i, (tx, ty, _) in enumerate(inits):
        cur = (i == int(spec["init_idx"]))
        cv2.circle(img, _w2p(tx, ty, W, H, xr, yr), 6 if cur else 3,
                   (60, 60, 255) if cur else (140, 140, 70), -1 if not cur else 2)
    for xy, nm, cell in [(cellA_xy, a_name, 0), (cellB_xy, b_name, 1)]:
        p = _w2p(xy[0], xy[1], W, H, xr, yr)
        cv2.circle(img, p, 10, (0, 170, 230), -1)
        if cell == tc:
            cv2.circle(img, p, 14, (0, 230, 0), 2)
        cv2.putText(img, cell_names[cell] + ":" + nm[:4], (p[0] - 14, p[1] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
    cv2.circle(img, _w2p(ee_xy[0], ee_xy[1], W, H, xr, yr), 4, (0, 255, 255), -1)
    cv2.putText(img, "TOP VIEW", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(img, f"{phase}  {done_n}/{total_n}", (6, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1)
    return img


def collect_loop(args):
    Rmap = R.from_matrix(AXIS_FACING)
    obj_names = tuple(args.objects)
    center = tuple(args.obj_center)
    env = make_circle_env(args.view_size, obj_names=obj_names, obj_center=center,
                          obj_sep=args.obj_sep, obj_axis=args.obj_axis,
                          basket_xy=tuple(args.basket_xy))
    re = env

    spans = spans_from(args.grid_span, args.span_grow, args.n_layers)
    def z_layers_for(top_z):
        return z_layers_from(top_z + args.init_clearance, args.layer_dz, args.n_layers)
    def grid_for(c_xy, top_z):
        return layered_grid_inits(c_xy, z_layers_for(top_z), spans, spans)
    def fan_idx(target_cell, approach):
        return (1 - int(target_cell)) if int(approach) else int(target_cell)

    env_args = {"env_name": "KinovaTwoObjCircle", "type": 1,
                "env_kwargs": {"robots": "Kinova3", "objects": list(obj_names),
                               "controller": "OSC_POSE", "obj_center": list(center),
                               "obj_sep": args.obj_sep, "init_mode": args.init_mode}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    env.reset()

    voxel_ijk = None
    if args.init_mode == "voxel":
        dims = tuple(args.voxel_dims)
        drop_bottom = bool(args.voxel_drop_bottom)
        voxA = voxel_inits(re.cellA_xy, re.object_center_z(0), args.voxel_size, dims, drop_object_layer=drop_bottom)
        voxB = voxel_inits(re.cellB_xy, re.object_center_z(1), args.voxel_size, dims, drop_object_layer=drop_bottom)
        fans = [[np.array(v["xyz"]) for v in voxA], [np.array(v["xyz"]) for v in voxB]]
        voxel_ijk = [[v["ijk"] for v in voxA], [v["ijk"] for v in voxB]]
        reachable = [[], []]
        print("[voxel] reach pre-pass (dropping voxels the arm can't reach)...")
        for tc in (0, 1):
            re.set_episode(0, tc)
            for vi, tgt in enumerate(fans[tc]):
                o = env.reset(); hq = R.from_quat(np.asarray(o['robot0_eef_quat']))
                for _ in range(args.setup_steps):
                    ee = np.asarray(o['robot0_eef_pos']); er = R.from_quat(np.asarray(o['robot0_eef_quat']))
                    aa = np.zeros(7); aa[6] = -1.0; pe = tgt - ee
                    aa[:3] = np.clip(pe * args.pos_kp, -1, 1)
                    aa[3:6] = np.clip((hq * er.inv()).as_rotvec() * args.rot_kp, -1, 1)
                    o, _, _, _ = env.step(aa)
                    if np.linalg.norm(pe) < 0.01:
                        break
                if np.linalg.norm(tgt - np.asarray(o['robot0_eef_pos'])) < args.voxel_reach_tol:
                    reachable[tc].append(vi)
            print(f"[voxel] target {tc}: {len(reachable[tc])}/{len(fans[tc])} voxels reachable")
        full_schedule = build_voxel_schedule(reachable, repeats=args.repeats,
                                             swap_positions=bool(args.voxel_swap))
    else:
        n_init = args.n_layers * 9
        full_schedule = build_circle_schedule(n_init=n_init, repeats=args.repeats)
        fans = [grid_for(re.cellA_xy, re.object_anchor_z(0, args.init_anchor)),
                grid_for(re.cellB_xy, re.object_anchor_z(1, args.init_anchor))]
    total_n = len(full_schedule)
    model_xml = env.model.get_xml()
    writer = DemoWriter(args.out, env_args, model_xml)

    done_keys = set()
    if os.path.exists(args.out):
        with h5py.File(args.out, "r") as f:
            for k in f["data"]:
                if not k.startswith("demo_"):
                    continue
                a = f["data"][k].attrs
                done_keys.add((int(a["obj_left_idx"]), int(a["target_cell"]),
                               int(a.get("approach", 0)), int(a["init_idx"])))
    remaining = [s for s in full_schedule if spec_key(s) not in done_keys]
    done_start = writer.num_demos()
    print(f"[schedule] full={total_n} done={len(done_keys)} remaining={len(remaining)}")

    target_dt = 1.0 / 20.0
    PHASE_SETUP, PHASE_WAIT, PHASE_RECORD = "SETUP", "READY-grip to start", "RECORDING"

    def robot_joint_pos():
        sm = re.sim.model
        return np.array([re.sim.data.qpos[sm.jnt_qposadr[sm.joint_name2id(j)]]
                         for j in re.robots[0].robot_joints], np.float32)

    def make_vr_frame(obs, spec, phase, done_n, ee_pos, flash=None):
        agent = cv2.cvtColor(np.asarray(obs['agentview_image'])[::-1], cv2.COLOR_RGB2BGR)
        wrist = cv2.cvtColor(np.asarray(obs['robot0_eye_in_hand_image'])[::-1], cv2.COLOR_RGB2BGR)
        disp = max(agent.shape[0], 256)
        agent = cv2.resize(agent, (disp, disp), interpolation=cv2.INTER_NEAREST)
        wrist = cv2.resize(wrist, (disp, disp), interpolation=cv2.INTER_NEAREST)
        cv2.putText(agent, 'agent', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(wrist, 'wrist', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        bird = draw_birdeye_circle(disp, re.cellA_xy, re.cellB_xy, re.cell_names,
                                   re._basket_xy, fans[fan_idx(spec["target_cell"], spec["approach"])],
                                   obj_names, spec, ee_pos[:2], phase, done_n, total_n)
        sep = np.zeros((disp, 4, 3), np.uint8)
        row = np.hstack([agent, sep, wrist, sep, bird])
        banner = np.full((40, row.shape[1], 3), 20, np.uint8)
        cv2.putText(banner, f"[{phase}]  pick the {re.target_name.replace('_',' ')} -> basket",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        img = np.vstack([banner, row])
        if flash:
            color = (0, 200, 0) if flash == "SAVED" else (0, 0, 230)
            cv2.putText(img, flash, (row.shape[1] // 2 - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        return img

    def publish(img):
        ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with shared.lock:
                shared.frame = jpg.tobytes(); shared.frame_id += 1

    def get_ctrl():
        with shared.lock:
            return shared.controller

    def rec_obs(obs):
        return {
            "agentview_rgb": np.asarray(obs['agentview_image']),
            "eye_in_hand_rgb": np.asarray(obs['robot0_eye_in_hand_image']),
            "joint_states": robot_joint_pos(),
            "gripper_states": np.asarray(obs['robot0_gripper_qpos'], np.float32),
            "ee_pos": np.asarray(obs['robot0_eef_pos'], np.float32),
            "ee_quat": np.asarray(obs['robot0_eef_quat'], np.float32),
        }

    while shared.running:
        done_n = writer.num_demos()
        ptr = done_n - done_start
        if ptr >= len(remaining):
            with shared.lock:
                shared.status = "ALL EPISODES COLLECTED"
            obs = re._get_observations(force_update=True)
            publish(make_vr_frame(obs, remaining[-1] if remaining else full_schedule[-1],
                                  "DONE", done_n, np.asarray(obs['robot0_eef_pos']), flash="SAVED"))
            time.sleep(0.5)
            continue

        spec = remaining[ptr]
        re.set_episode(spec["obj_left_idx"], spec["target_cell"])
        li = spec["obj_left_idx"]
        obj_xy = re.object_positions(re._spec)
        if args.init_mode != "voxel":
            fans = [grid_for(obj_xy["A"], re.object_anchor_z(li, args.init_anchor)),
                    grid_for(obj_xy["B"], re.object_anchor_z(1 - li, args.init_anchor))]
        obs = env.reset()
        home_erot = R.from_quat(np.asarray(obs['robot0_eef_quat']))
        with shared.lock:
            shared.status = f"episode {done_n+1}/{total_n}: {re.language}"

        target_ee = np.array(fans[fan_idx(spec["target_cell"], spec["approach"])][spec["init_idx"]])
        for _ in range(args.setup_steps):
            ee_pos = np.asarray(obs['robot0_eef_pos'])
            ee_rot = R.from_quat(np.asarray(obs['robot0_eef_quat']))
            a = np.zeros(7); a[6] = -1.0
            perr = target_ee - ee_pos
            a[:3] = np.clip(perr * args.pos_kp, -1, 1)
            a[3:6] = np.clip((home_erot * ee_rot.inv()).as_rotvec() * args.rot_kp, -1, 1)
            obs, _, _, _ = env.step(a)
            publish(make_vr_frame(obs, spec, PHASE_SETUP, done_n, ee_pos))
            if np.linalg.norm(perr) < 0.01:
                break

        engaged = False
        while shared.running and not engaged:
            c = get_ctrl()
            ee_pos = np.asarray(obs['robot0_eef_pos'])
            ee_rot = R.from_quat(np.asarray(obs['robot0_eef_quat']))
            a = np.zeros(7); a[6] = -1.0
            a[:3] = np.clip((target_ee - ee_pos) * args.pos_kp, -1, 1)
            a[3:6] = np.clip((home_erot * ee_rot.inv()).as_rotvec() * args.rot_kp, -1, 1)
            obs, _, _, _ = env.step(a)
            publish(make_vr_frame(obs, spec, PHASE_WAIT, done_n, ee_pos))
            if c is not None and float(c.get('grip', 0.0)) > args.engage_threshold:
                engaged = True
            time.sleep(target_dt)

        obs_seq, actions, rewards, dones, states = [], [], [], [], []
        cur_mode = None
        anchor_cpos = anchor_crot = anchor_epos = anchor_erot = None
        prev_b = False
        outcome = None
        while shared.running and outcome is None:
            t0 = time.time()
            c = get_ctrl()
            ee_pos = np.asarray(obs['robot0_eef_pos'], np.float64)
            ee_rot = R.from_quat(np.asarray(obs['robot0_eef_quat']))
            action = np.zeros(7, np.float64); action[6] = -1.0

            b = bool(c.get('bbtn', False)) if c else False
            if b and not prev_b:
                outcome = "discard"; break
            prev_b = b

            if c is not None:
                action[6] = 1.0 if float(c.get('trigger', 0.0)) > 0.5 else -1.0
                grip = float(c.get('grip', 0.0))
                cpos = np.array(c.get('pos', [0, 0, 0]), np.float64)
                crot = R.from_quat(c.get('quat', [0, 0, 0, 1]))
                mode = 'rot' if bool(c.get('a', False)) else 'pos'
                if grip > args.engage_threshold:
                    if cur_mode != mode or anchor_cpos is None:
                        cur_mode = mode
                        anchor_cpos, anchor_crot = cpos, crot
                        anchor_epos, anchor_erot = ee_pos.copy(), ee_rot
                    else:
                        if mode == 'pos':
                            disp = Rmap.apply(cpos - anchor_cpos) * args.pos_scale
                            desired_epos = anchor_epos + disp
                            desired_erot = anchor_erot
                        else:
                            desired_epos = anchor_epos
                            dworld = Rmap * (crot * anchor_crot.inv()) * Rmap.inv()
                            desired_erot = dworld * anchor_erot
                        perr = desired_epos - ee_pos
                        perr[np.abs(perr) < 0.001] = 0.0
                        action[:3] = np.clip(perr * args.pos_kp, -1, 1)
                        rerr = (desired_erot * ee_rot.inv()).as_rotvec()
                        rerr[np.abs(rerr) < 0.01] = 0.0
                        action[3:6] = np.clip(rerr * args.rot_kp, -1, 1)
                else:
                    cur_mode = None; anchor_cpos = None

            obs_seq.append(rec_obs(obs))
            actions.append(action.astype(np.float32))
            states.append(re.sim.get_state().flatten())
            rewards.append(0.0); dones.append(0)
            obs, _, _, _ = env.step(action)

            if re._check_success() and action[6] < 0:
                hold = np.asarray(obs['robot0_eef_pos'], np.float64)
                hrot = R.from_quat(np.asarray(obs['robot0_eef_quat']))
                for _ in range(args.settle_capture):
                    obs_seq.append(rec_obs(obs))
                    cur = np.asarray(obs['robot0_eef_pos'], np.float64)
                    crot = R.from_quat(np.asarray(obs['robot0_eef_quat']))
                    sa = np.zeros(7, np.float64); sa[6] = -1.0
                    sa[:3] = np.clip((hold - cur) * args.pos_kp, -1, 1)
                    sa[3:6] = np.clip((hrot * crot.inv()).as_rotvec() * args.rot_kp, -1, 1)
                    actions.append(sa.astype(np.float32))
                    states.append(re.sim.get_state().flatten())
                    rewards.append(0.0); dones.append(0)
                    obs, _, _, _ = env.step(sa)
                    publish(make_vr_frame(obs, spec, PHASE_RECORD, done_n, cur))
                rewards[-1] = 1.0; dones[-1] = 1
                outcome = "save"

            publish(make_vr_frame(obs, spec, PHASE_RECORD, done_n, ee_pos))
            dt = time.time() - t0
            if dt < target_dt:
                time.sleep(target_dt - dt)
            if len(actions) >= args.max_steps:
                outcome = "discard"

        if outcome == "save" and len(actions) > 0:
            _ii = spec["init_idx"]; _tc = spec["target_cell"]
            if args.init_mode == "voxel":
                _vijk = voxel_ijk[_tc][_ii]
                _layer = int(_vijk[2]); _cil = int(_vijk[0]) * args.voxel_dims[1] + int(_vijk[1])
                _span = float(args.voxel_size); _nlay = int(args.voxel_dims[2])
            else:
                _layer = _ii // 9; _cil = _ii % 9
                _span = float(spans[_layer]); _nlay = int(args.n_layers)
            meta = {"obj_left_idx": spec["obj_left_idx"], "target_cell": _tc,
                    "target_idx": re._spec["target_idx"], "init_idx": _ii,
                    "approach": spec["approach"], "repeat": spec["repeat"], "success": 1,
                    "inverse": int(spec["approach"]), "init_mode": args.init_mode,
                    "target": re.target_name, "language": re.language,
                    "layer": _layer,
                    "cell_in_layer": _cil,
                    "n_layers": _nlay,
                    "init_xyz": [float(v) for v in target_ee],
                    "init_z": float(target_ee[2]),
                    "init_span": _span,
                    "init_clearance": float(args.init_clearance),
                    "init_anchor": args.init_anchor,
                    "start_cell": int(fan_idx(_tc, spec["approach"])),
                    "obj_a_xy": [float(v) for v in obj_xy["A"]],
                    "obj_b_xy": [float(v) for v in obj_xy["B"]],
                    "obj_a_top": float(re.object_top_z(li)),
                    "obj_b_top": float(re.object_top_z(1 - li)),
                    "obj_a_com": float(re.object_center_z(li)),
                    "obj_b_com": float(re.object_center_z(1 - li)),
                    "target_xyz": [float(v) for v in re.target_pos()],
                    "basket_xy": [float(v) for v in re._basket_xy]}
            if args.init_mode == "voxel":
                meta["voxel_ijk"] = [int(x) for x in voxel_ijk[_tc][_ii]]
                meta["voxel_size"] = float(args.voxel_size)
                meta["voxel_dims"] = [int(x) for x in args.voxel_dims]
            idx = writer.add_demo(obs_seq, actions, rewards, dones, states, meta)
            print(f"[save] demo_{idx} ({len(actions)} steps) {re.language} init{spec['init_idx']}")
            publish(make_vr_frame(obs, spec, "SAVED", writer.num_demos(), ee_pos, flash="SAVED"))
            time.sleep(0.6)
        else:
            print(f"[discard] retry ({len(actions)} steps)")
            publish(make_vr_frame(obs, spec, "DISCARD", done_n, ee_pos, flash="DISCARD"))
            time.sleep(0.6)

    env.close()


async def index(request):
    return web.FileResponse(os.path.join(HERE, 'static', 'index.html'))


async def ws_handler(request):
    import asyncio
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    print("[ws] client connected")

    async def sender():
        last = -1
        while not ws.closed:
            with shared.lock:
                fid, f = shared.frame_id, shared.frame
            if f is not None and fid != last:
                last = fid
                try:
                    await ws.send_bytes(f)
                except Exception:
                    break
            await asyncio.sleep(1.0 / 30.0)

    task = asyncio.create_task(sender())
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                if data.get('type') == 'controller':
                    with shared.lock:
                        shared.controller = data
    finally:
        task.cancel()
        print("[ws] client disconnected")
    return ws


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=os.path.join(os.environ.get('SIM_DATA_DIR', '~/kinova_dataset'),
                                                 'data', 'circle.hdf5'),
                   help='output HDF5, resumed if it exists (default $SIM_DATA_DIR/data/circle.hdf5)')
    p.add_argument('--objects', nargs=2, default=['red_cube', 'yellow_cube'])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--view-size', type=int, default=128)
    p.add_argument('--obj-center', nargs=2, type=float, default=[0.0, 0.20])
    p.add_argument('--obj-sep', type=float, default=0.07, help='half-distance between the two objects (m)')
    p.add_argument('--obj-axis', default='x', choices=['x', 'y'],
                   help='axis the two objects are separated along')
    p.add_argument('--inverse', action='store_true',
                   help='deprecated, ignored')
    p.add_argument('--basket-xy', nargs=2, type=float, default=[0.0, -0.18])
    p.add_argument('--init-anchor', default='com', choices=['top', 'com'],
                   help="what --init-clearance is measured from: object 'com' or 'top' surface")
    p.add_argument('--init-clearance', type=float, default=0.085, help='gap (m) from the --init-anchor up to the lowest init layer')
    p.add_argument('--n-layers', type=int, default=3, help='number of z layers in the init lattice (each a 3x3 grid)')
    p.add_argument('--layer-dz', type=float, default=0.05, help='z gap between init layers (m)')
    p.add_argument('--grid-span', type=float, default=0.02, help='lowest-layer 3x3 half-extent (m); grows by --span-grow per layer')
    p.add_argument('--span-grow', type=float, default=0.015, help='extra half-extent per higher layer (m)')
    p.add_argument('--init-mode', default='funnel', choices=['funnel', 'voxel'])
    p.add_argument('--voxel-size', type=float, default=0.05, help='voxel edge (m); ~ the object size')
    p.add_argument('--voxel-dims', nargs=3, type=int, default=[3, 3, 3], help='nx ny nz voxels around each object')
    p.add_argument('--voxel-reach-tol', type=float, default=0.02, help='voxel kept if EE reaches within this (m)')
    p.add_argument('--voxel-drop-bottom', type=int, default=0, help='1 = drop the bottom voxel layer (keep only layers above the object)')
    p.add_argument('--voxel-swap', type=int, default=1, help='1 = also collect with the two objects swapped between cells; 0 = one arrangement only')
    p.add_argument('--r-min', type=float, default=0.10)
    p.add_argument('--r-max', type=float, default=0.22)
    p.add_argument('--n-r', type=int, default=5)
    p.add_argument('--n-ang', type=int, default=5)
    p.add_argument('--fan-deg', type=float, default=110.0, help='fan angular width (<180)')
    p.add_argument('--pos-scale', type=float, default=1.0)
    p.add_argument('--pos-kp', type=float, default=12.0)
    p.add_argument('--rot-kp', type=float, default=2.5)
    p.add_argument('--engage-threshold', type=float, default=0.5)
    p.add_argument('--setup-steps', type=int, default=100)
    p.add_argument('--max-steps', type=int, default=1500)
    p.add_argument('--settle-capture', type=int, default=12)
    p.add_argument('--port', type=int, default=8002)
    p.add_argument('--config', default='cube',
                   help="environment preset in env_configs/<name>.json (geometry only); CLI flags override")
    from env_config import apply_config
    pre, _ = p.parse_known_args()
    cfgpath = apply_config(p, pre.config)
    args = p.parse_args()
    args.out = os.path.expanduser(args.out)
    if cfgpath:
        print(f"[config] env '{pre.config}' -> {cfgpath}")

    th = threading.Thread(target=collect_loop, args=(args,), daemon=True)
    th.start()

    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/ws', ws_handler)
    app.router.add_static('/static/', os.path.join(HERE, 'static'))
    print(f"[server] http://localhost:{args.port}  -> {args.out}")
    web.run_app(app, host='127.0.0.1', port=args.port, print=None)
    shared.running = False


if __name__ == '__main__':
    main()
