#!/usr/bin/env python3
import numpy as np
import robosuite
from robosuite import load_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from scipy.spatial.transform import Rotation as R

from libero.libero.envs.objects import get_object_dict

from kinova_env import MenagerieRobotiq2f85


_CUBE_RGBA = {
    "red_cube":    [0.85, 0.12, 0.12, 1.0],
    "yellow_cube": [0.95, 0.85, 0.10, 1.0],
}
CUBE_HALF = 0.025


def make_object(name, inst_name):
    if name in _CUBE_RGBA:
        return BoxObject(name=inst_name, size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
                         rgba=_CUBE_RGBA[name], density=500, friction=[1.0, 0.05, 0.001])
    return get_object_dict()[name](name=inst_name)


def grid_3x3(center_xy, span_xy):
    cx, cy = center_xy
    sx, sy = span_xy
    xs = [cx - sx, cx, cx + sx]
    ys = [cy - sy, cy, cy + sy]
    return [(x, y) for y in ys for x in xs]


def _upright_quat_wxyz(obj, yaw=0.0):
    if abs(yaw) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = R.from_euler("z", yaw).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


HOME_QPOS = np.array([-0.003, 0.468, 0.005, 1.791, -0.003, 0.881, -1.568])


class KinovaTwoObjGridPick(Lift):

    def __init__(
        self,
        obj_names=("milk", "tomato_sauce"),
        obj_grid_center=(0.0, 0.10),
        obj_grid_span=(0.08, 0.07),
        basket_xy=(0.0, -0.18),
        success_radius=0.06,
        success_z_margin=0.11,
        success_speed=0.04,
        settle_steps=60,
        init_qpos=None,
        **kwargs,
    ):
        self._obj_names = tuple(obj_names)
        self.obj_grid = grid_3x3(obj_grid_center, obj_grid_span)
        self._basket_xy = tuple(basket_xy)
        self._success_radius = float(success_radius)
        self._success_z_margin = float(success_z_margin)
        self._success_speed = float(success_speed)
        self._settle_steps = int(settle_steps)
        self._init_qpos = np.array(init_qpos, dtype=float) if init_qpos is not None else HOME_QPOS.copy()
        self._spec = {"target_idx": 0, "target_cell": 0, "other_cell": 4}
        self._objs = []
        self._basket = None
        super().__init__(**kwargs)

    def set_episode(self, target_idx, target_cell, other_cell):
        self._spec = {
            "target_idx": int(target_idx),
            "target_cell": int(target_cell),
            "other_cell": int(other_cell),
        }

    @property
    def target_name(self):
        return self._obj_names[self._spec["target_idx"]]

    @property
    def language(self):
        return f"pick up the {self.target_name.replace('_', ' ')} and place it in the basket"

    def _load_model(self):
        super(Lift, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self._objs = [
            make_object(self._obj_names[0], "obj0"),
            make_object(self._obj_names[1], "obj1"),
        ]
        self._basket = get_object_dict()["basket"](name="basket")

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self._objs + [self._basket],
        )

    def _setup_references(self):
        super(Lift, self)._setup_references()
        if self._init_qpos is not None:
            self.robots[0].init_qpos = np.array(self._init_qpos)
        self.obj_body_ids = [self.sim.model.body_name2id(o.root_body) for o in self._objs]
        self.basket_body_id = self.sim.model.body_name2id(self._basket.root_body)

    def _setup_observables(self):
        return super(Lift, self)._setup_observables()

    _UPRIGHT_CANDS = {
        "identity": R.identity(),
        "Rx90": R.from_euler("x", np.pi / 2),
        "Rx-90": R.from_euler("x", -np.pi / 2),
        "Ry90": R.from_euler("y", np.pi / 2),
        "Ry-90": R.from_euler("y", -np.pi / 2),
    }

    def _body_local_corners(self, body_id):
        m = self.sim.model
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body_id:
                continue
            ab = m.geom_aabb[g]
            ctr, half = ab[:3], ab[3:]
            gp = m.geom_pos[g]
            gq = m.geom_quat[g]
            Rg = R.from_quat([gq[1], gq[2], gq[3], gq[0]]).as_matrix()
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        pts.append(gp + Rg @ (ctr + np.array([sx, sy, sz]) * half))
        return np.asarray(pts) if pts else np.zeros((1, 3))

    _UPRIGHT_OVERRIDE = {
        "tomato_sauce": R.from_euler("x", np.pi / 2),
    }

    def _upright_pose(self, body_id, force_identity=False, override=None):
        if not hasattr(self, "_pose_cache"):
            self._pose_cache = {}
        key = (body_id, force_identity, override is not None)
        if key in self._pose_cache:
            return self._pose_cache[key]
        corners = self._body_local_corners(body_id)
        if force_identity:
            Rc = R.identity()
        elif override is not None:
            Rc = override
        else:
            Rc = max(self._UPRIGHT_CANDS.values(),
                     key=lambda Rm: np.ptp((corners @ Rm.as_matrix().T)[:, 2]))
        zmin = (corners @ Rc.as_matrix().T)[:, 2].min()
        q = Rc.as_quat()
        pose = (np.array([q[3], q[0], q[1], q[2]]), -float(zmin))
        self._pose_cache[key] = pose
        return pose

    def object_height(self, idx):
        bid = self.obj_body_ids[idx]
        corners = self._body_local_corners(bid)
        override = self._UPRIGHT_OVERRIDE.get(self._obj_names[idx])
        if override is not None:
            Rc = override
        else:
            Rc = max(self._UPRIGHT_CANDS.values(),
                     key=lambda Rm: np.ptp((corners @ Rm.as_matrix().T)[:, 2]))
        return float(np.ptp((corners @ Rc.as_matrix().T)[:, 2]))

    def object_top_z(self, idx):
        return float(self.table_offset[2] + self.object_height(idx) + 0.002)

    def object_center_z(self, idx):
        return float(self.table_offset[2] + 0.002 + self.object_height(idx) / 2.0)

    def object_anchor_z(self, idx, anchor="top"):
        return self.object_center_z(idx) if anchor == "com" else self.object_top_z(idx)

    def _place_free(self, obj, body_id, xy, force_identity=False):
        override = None
        if not force_identity and obj in self._objs:
            name = self._obj_names[self._objs.index(obj)]
            override = self._UPRIGHT_OVERRIDE.get(name)
        quat, lift = self._upright_pose(body_id, force_identity, override)
        z = self.table_offset[2] + lift + 0.002
        self.sim.data.set_joint_qpos(
            obj.joints[0], np.concatenate([[xy[0], xy[1], z], quat]))

    def _reset_internal(self):
        super(Lift, self)._reset_internal()

        ti = self._spec["target_idx"]
        tc, oc = self._spec["target_cell"], self._spec["other_cell"]
        oi = 1 - ti
        self._place_free(self._objs[ti], self.obj_body_ids[ti], self.obj_grid[tc])
        self._place_free(self._objs[oi], self.obj_body_ids[oi], self.obj_grid[oc])
        self._place_free(self._basket, self.basket_body_id, self._basket_xy,
                         force_identity=True)
        self.sim.forward()
        for _ in range(self._settle_steps):
            self.sim.step()

    def target_pos(self):
        return np.array(self.sim.data.body_xpos[self.obj_body_ids[self._spec["target_idx"]]])

    def basket_pos(self):
        return np.array(self.sim.data.body_xpos[self.basket_body_id])

    def target_speed(self):
        m = self.sim.model
        jid = m.joint_name2id(self._objs[self._spec["target_idx"]].joints[0])
        adr = m.jnt_dofadr[jid]
        return float(np.linalg.norm(self.sim.data.qvel[adr:adr + 3]))

    def _check_success(self):
        op = self.target_pos()
        bp = self.basket_pos()
        dxy = np.linalg.norm(op[:2] - bp[:2])
        low = (op[2] - bp[2]) < self._success_z_margin
        settled = self.target_speed() < self._success_speed
        return bool(dxy < self._success_radius and low and settled)


def make_grid_env(view_size, obj_names=("milk", "tomato_sauce"), extra_cams=(),
                  obj_grid_center=(0.0, 0.10), obj_grid_span=(0.12, 0.10),
                  basket_xy=(0.0, -0.18)):
    controller = load_controller_config(default_controller="OSC_POSE")
    cams = ["agentview", "robot0_eye_in_hand"] + list(extra_cams)
    env = robosuite.make(
        env_name="KinovaTwoObjGridPick",
        robots="Kinova3",
        gripper_types="MenagerieRobotiq2f85",
        controller_configs=controller,
        obj_names=obj_names,
        obj_grid_center=tuple(obj_grid_center),
        obj_grid_span=tuple(obj_grid_span),
        basket_xy=tuple(basket_xy),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cams,
        camera_heights=view_size,
        camera_widths=view_size,
        control_freq=20,
        ignore_done=True,
        horizon=100000,
        reward_shaping=False,
    )
    return env


if __name__ == "__main__":
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = make_grid_env(256)
    env.set_episode(target_idx=0, target_cell=2, other_cell=6)
    obs = env.reset()
    re = env
    print("task:", re.language)
    print("action_dim:", re.action_dim)
    print("target rest pos:", np.round(re.target_pos(), 3))
    print("other  rest pos:", np.round(re.sim.data.body_xpos[re.obj_body_ids[1]], 3))
    print("basket pos:      ", np.round(re.basket_pos(), 3))
    print("success (start):", re._check_success())
    bp = re.basket_pos()
    re.sim.data.set_joint_qpos(
        re._objs[0].joints[0],
        np.concatenate([[bp[0], bp[1], bp[2] + 0.03], _upright_quat_wxyz(re._objs[0])]),
    )
    re.sim.forward()
    print("success (teleported into basket):", re._check_success())
    os._exit(0)
