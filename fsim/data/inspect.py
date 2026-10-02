#!/usr/bin/env python3
"""Summarise an episode or merged file: python -m fsim.data.inspect <file.hdf5>"""
from __future__ import annotations

import json
import sys

import h5py
import numpy as np


def describe(path: str) -> str:
    lines = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        env_args = json.loads(data.attrs["env_args"])
        lines.append(f"{path}: schema {data.attrs.get('schema', '?')}, env {env_args.get('env_name')}, "
                     f"total {data.attrs.get('total')} steps")
        for name in sorted((k for k in data if k.startswith("demo_")), key=lambda k: int(k.split("_")[1])):
            g = data[name]
            a = g.attrs
            rew = float(np.sum(g["rewards"][()])) if "rewards" in g else float("nan")
            lines.append(f"  {name}: {a['num_samples']} steps, success {bool(a.get('success'))}, reward {rew:g}, "
                         f"task {a.get('task')}, randomness {a.get('randomness')}, operator {a.get('operator', '')}")
        first = data[sorted(k for k in data if k.startswith("demo_"))[0]] if any(k.startswith("demo_") for k in data) else None
        if first is not None:
            for k in sorted(first.keys()):
                if k == "obs":
                    for ok in sorted(first["obs"].keys()):
                        d = first["obs"][ok]
                        lines.append(f"    obs/{ok:18s} {d.shape} {d.dtype}")
                else:
                    d = first[k]
                    lines.append(f"    {k:22s} {d.shape} {d.dtype}")
        if "mask" in f:
            for k in f["mask"]:
                lines.append(f"  mask/{k}: {len(f['mask'][k])} demos")
    return "\n".join(lines)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(describe(p))
