"""3D client scene export and pose frames (standalone scene; runs on Windows and Linux)."""
import io

import mujoco
import numpy as np
import trimesh

from fsim.models.scene import build_scene_xml
from fsim.spec import get_task
from fsim.vr.scene_export import MSG_POSES, PosePacker, export_scene, unpack_poses

TASK = get_task("one_leg")


def model_and_data():
    m = mujoco.MjModel.from_xml_string(build_scene_xml(TASK, visual_meshes=True))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_glb_has_one_node_per_visual_body():
    m, _ = model_and_data()
    glb, manifest = export_scene(m, groups=(0, 1))
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    names = {b["name"] for b in manifest["bodies"]}
    assert {f"{p}_main" for p in TASK.part_names} <= names
    nodes = set(scene.graph.nodes_geometry)
    assert {b["node"] for b in manifest["bodies"]} <= nodes
    assert len(glb) < 20e6


def test_pose_frame_roundtrip_matches_simulation():
    m, d = model_and_data()
    _, manifest = export_scene(m, groups=(0, 1))
    packer = PosePacker(manifest)
    buf = packer.pack(d, target=np.r_[0.1, 0.2, 0.3, 1, 0, 0, 0], flags=[1, 0, 0, -1])
    out = unpack_poses(buf, len(manifest["bodies"]))
    assert out["type"] == MSG_POSES and out["seq"] == 1
    for i, b in enumerate(manifest["bodies"]):
        assert np.allclose(out["poses"][i, :3], d.xpos[b["id"]], atol=1e-6)
        assert np.allclose(out["poses"][i, 3:], d.xquat[b["id"]], atol=1e-6)
    assert np.allclose(out["target"], [0.1, 0.2, 0.3, 1, 0, 0, 0]) and np.allclose(out["flags"], [1, 0, 0, -1])
