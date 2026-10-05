#!/usr/bin/env python3
import json
import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_configs")

GEOM_KEYS = {"objects", "obj_center", "obj_sep", "obj_axis", "basket_xy",
             "init_anchor", "init_clearance", "layer_dz", "n_layers",
             "grid_span", "span_grow",
             "init_mode", "voxel_size", "voxel_dims", "voxel_drop_bottom", "voxel_swap",
             "guideline_size", "init_half", "init_res", "init_z_layers", "cube_half",
             "init_y_min"}


def apply_config(parser, name):
    if not name:
        return None
    path = name if os.path.exists(name) else os.path.join(CONFIG_DIR, name + ".json")
    if not os.path.exists(path):
        if name == "cube":
            return None
        raise SystemExit(f"[config] env '{name}' not found (looked in {CONFIG_DIR}/)")
    with open(path) as f:
        cfg = json.load(f)
    parser.set_defaults(**{k: v for k, v in cfg.items() if k in GEOM_KEYS})
    return path
