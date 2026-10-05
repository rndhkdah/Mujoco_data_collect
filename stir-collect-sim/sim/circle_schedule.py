#!/usr/bin/env python3
import numpy as np


def circle_inits(center_xy, z, r_min=0.10, r_max=0.22, n_r=5, n_ang=5,
                 ang_center_deg=180.0, ang_width_deg=110.0):
    pts = []
    rs = np.linspace(r_min, r_max, n_r)
    a0 = ang_center_deg - ang_width_deg / 2.0
    a1 = ang_center_deg + ang_width_deg / 2.0
    angs = np.linspace(a0, a1, n_ang)
    for r in rs:
        for a in angs:
            ar = np.radians(a)
            pts.append((float(center_xy[0] + r * np.cos(ar)),
                        float(center_xy[1] + r * np.sin(ar)),
                        float(z)))
    return pts


def layered_grid_inits(center_xy, z_layers, span_x=0.08, span_y=0.08):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    n = len(z_layers)
    sx = list(span_x) if hasattr(span_x, "__len__") else [span_x] * n
    sy = list(span_y) if hasattr(span_y, "__len__") else [span_y] * n
    pts = []
    for i, z in enumerate(z_layers):
        xs = (cx - sx[i], cx, cx + sx[i])
        ys = (cy - sy[i], cy, cy + sy[i])
        pts.extend((x, y, float(z)) for x in xs for y in ys)
    return pts


def z_layers_from(base_z, dz=0.06, n=3):
    return [float(base_z) + i * float(dz) for i in range(int(n))]


def spans_from(base, grow, n):
    return [float(base) + i * float(grow) for i in range(int(n))]


def voxel_inits(center_xy, base_z, voxel_size=0.05, dims=(3, 3, 3),
                include_object_cell=False, drop_object_layer=False):
    nx, ny, nz = dims
    cx, cy = float(center_xy[0]), float(center_xy[1])
    vs = float(voxel_size)
    oi, oj = nx // 2, ny // 2
    out = []
    for k in range(nz):
        if k == 0 and drop_object_layer:
            continue
        for i in range(nx):
            for j in range(ny):
                if (i, j, k) == (oi, oj, 0) and not include_object_cell:
                    continue
                x = cx + (i - (nx - 1) / 2.0) * vs
                y = cy + (j - (ny - 1) / 2.0) * vs
                z = base_z + k * vs
                out.append({"ijk": (i, j, k), "xyz": (x, y, z)})
    return out


def build_voxel_schedule(reachable_per_target, repeats=1, swap_positions=True):
    lefts = (0, 1) if swap_positions else (0,)
    specs = []
    for r in range(repeats):
        for li in lefts:
            for tc, vis in enumerate(reachable_per_target):
                for vi in vis:
                    specs.append({"obj_left_idx": li, "target_cell": tc, "approach": 0,
                                  "init_idx": int(vi), "repeat": r})
    return specs


def robot_facing_angle_deg(center_xy, base_xy=(-0.56, 0.0)):
    dx = base_xy[0] - center_xy[0]
    dy = base_xy[1] - center_xy[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def build_circle_schedule(n_init=27, repeats=1, cells_per_layer=9):
    n_layers = n_init // cells_per_layer
    specs = []
    for r in range(repeats):
        for layer in range(n_layers):
            for li in (0, 1):
                for tc in (0, 1):
                    for ap in (0, 1):
                        for cell in range(cells_per_layer):
                            ii = layer * cells_per_layer + cell
                            specs.append({"obj_left_idx": li, "target_cell": tc,
                                          "approach": ap, "init_idx": ii, "repeat": r})
    return specs


def spec_key(s):
    return (int(s["obj_left_idx"]), int(s["target_cell"]),
            int(s.get("approach", 0)), int(s["init_idx"]))


if __name__ == "__main__":
    pts = circle_inits((0.0, 0.10), 0.97)
    print("n inits:", len(pts), "(expect 25)")
    sched = build_circle_schedule()
    print("schedule:", len(sched), "(expect 100)")
    print("fan dir toward robot from (0,0.10):",
          round(robot_facing_angle_deg((0.0, 0.10)), 1), "deg")
