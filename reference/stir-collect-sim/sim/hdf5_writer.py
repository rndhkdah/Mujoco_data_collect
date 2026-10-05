#!/usr/bin/env python3
import importlib.util
import json
import os
import re
import xml.etree.ElementTree as ET

import h5py
import numpy as np

_ASSET_ROOTS = ("robosuite/models/assets/", "libero/libero/assets/", "/grippers/")


def portable_model_xml(xml):
    root = ET.fromstring(xml)
    for el in root.iter():
        for k, v in el.attrib.items():
            if not v.startswith("/"):
                continue
            p = os.path.normpath(v)
            for anchor in _ASSET_ROOTS:
                i = p.rfind(anchor)
                if i >= 0:
                    rel = p[i:] if anchor != "/grippers/" else "sim/assets" + p[i:]
                    el.set(k, rel)
                    break
            else:
                raise ValueError(f"model_file: absolute path outside the known asset roots: "
                                 f"<{el.tag} {k}={v!r}>")
    return ET.tostring(root, encoding="unicode")


def localize_model_xml(xml):
    import robosuite
    from kinova_env import menagerie_gripper_xml
    libero_pkg = importlib.util.find_spec("libero").submodule_search_locations[0]
    bases = {
        "robosuite/models/assets/": os.path.join(os.path.dirname(robosuite.__file__), "models", "assets") + "/",
        "libero/libero/assets/": os.path.join(libero_pkg, "libero", "assets") + "/",
        "sim/assets/grippers/": os.path.dirname(menagerie_gripper_xml()) + "/",
    }
    root = ET.fromstring(xml)
    for el in root.iter():
        for k, v in el.attrib.items():
            for rel, base in bases.items():
                if v.startswith(rel):
                    el.set(k, base + v[len(rel):])
                    break
    return ET.tostring(root, encoding="unicode")


_ABS_PATH = re.compile(r"""(?:^|["'\s=:,\[(])/[^/\s"']+/""")


def _check_no_abs_path(what, value):
    home = os.path.expanduser("~")
    for s in (value if isinstance(value, (list, tuple)) else [value]):
        if isinstance(s, bytes):
            s = s.decode("utf-8", "replace")
        if isinstance(s, str) and (_ABS_PATH.search(s) or home in s):
            raise ValueError(f"{what}: refusing to write an absolute path: {s[:120]!r}")


class DemoWriter:
    def __init__(self, path, env_args, model_xml=""):
        self.path = path
        self.model_xml = portable_model_xml(model_xml) if model_xml else ""
        env_args_json = json.dumps(env_args)
        _check_no_abs_path("env_args", env_args_json)
        with h5py.File(self.path, "a") as f:
            if "data" not in f:
                g = f.create_group("data")
                g.attrs["env_args"] = env_args_json
                g.attrs["total"] = 0
            self._n = self._count_demos(f["data"])

    @staticmethod
    def _count_demos(data_grp):
        return sum(1 for k in data_grp.keys() if k.startswith("demo_"))

    def num_demos(self):
        return self._n

    def add_demo(self, obs_seq, actions, rewards, dones, states, meta, extra_datasets=None):
        T = len(actions)
        if T == 0:
            raise ValueError("empty trajectory")
        for k, v in meta.items():
            _check_no_abs_path(f"meta[{k}]", v)

        cols = {k: np.stack([np.asarray(o[k]) for o in obs_seq]) for k in obs_seq[0]}

        with h5py.File(self.path, "a") as f:
            data = f["data"]
            name = f"demo_{self._n}"
            g = data.create_group(name)
            g.create_dataset("actions", data=np.asarray(actions, np.float32))
            g.create_dataset("rewards", data=np.asarray(rewards, np.float32))
            g.create_dataset("dones", data=np.asarray(dones, np.int64))
            g.create_dataset("states", data=np.asarray(states, np.float64))
            og = g.create_group("obs")
            for ds_name, arr in cols.items():
                if arr.ndim >= 3:
                    og.create_dataset(ds_name, data=arr.astype(np.uint8),
                                      compression="gzip", compression_opts=4)
                else:
                    og.create_dataset(ds_name, data=arr.astype(np.float32))
            if extra_datasets:
                for ds_name, arr in extra_datasets.items():
                    g.create_dataset(ds_name, data=np.asarray(arr))
            g.attrs["num_samples"] = T
            g.attrs["model_file"] = self.model_xml
            for k, v in meta.items():
                g.attrs[k] = v
            data.attrs["total"] = int(data.attrs.get("total", 0)) + T
            self._n += 1
        return self._n - 1
