#!/usr/bin/env python3
"""Merge per-episode files into one robomimic-style HDF5 (data/demo_0..N-1, mask/<randomness>).

    python -m fsim.data.merge <episodes dir or files...> -o one_leg.hdf5 [--success-only]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np


def episode_files(inputs: list[str]) -> list[str]:
    files = []
    for p in inputs:
        files += sorted(glob.glob(os.path.join(p, "**", "*.hdf5"), recursive=True)) if os.path.isdir(p) else [p]
    return [f for f in files if not f.endswith(".tmp")]


def merge(files: list[str], out: str, success_only: bool = False) -> int:
    masks = defaultdict(list)
    n, total, env_args = 0, 0, None
    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as dst:
        data = dst.create_group("data")
        for path in files:
            with h5py.File(path, "r") as src:
                if env_args is None:
                    env_args = src["data"].attrs["env_args"]
                for name in sorted(k for k in src["data"] if k.startswith("demo_")):
                    g = src["data"][name]
                    if success_only and not bool(g.attrs.get("success", False)):
                        continue
                    new = f"demo_{n}"
                    src.copy(g, data, name=new)
                    data[new].attrs["source_file"] = os.path.basename(path)
                    masks[str(g.attrs.get("randomness", "unknown"))].append(new)
                    total += int(g.attrs["num_samples"])
                    n += 1
        data.attrs["env_args"] = env_args if env_args is not None else json.dumps({})
        data.attrs["total"] = total
        mask = dst.create_group("mask")
        for key, demos in masks.items():
            mask.create_dataset(key, data=np.array(demos, dtype="S"))
    os.replace(tmp, out)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--success-only", action="store_true")
    args = ap.parse_args(argv)
    files = episode_files(args.inputs)
    n = merge(files, args.out, args.success_only)
    print(f"merged {n} demos from {len(files)} files -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
