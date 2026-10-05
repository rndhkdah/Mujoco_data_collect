#!/usr/bin/env python3
import numpy as np
import robosuite
from robosuite import load_controller_config
from robosuite.environments.manipulation.lift import Lift

from kinova_grid_env import KinovaTwoObjGridPick


class KinovaTwoObjCircle(KinovaTwoObjGridPick):

    def __init__(self, obj_center=(0.0, 0.10), obj_sep=0.12, obj_axis="y", **kwargs):
        self._obj_center = tuple(obj_center)
        self._obj_sep = float(obj_sep)
        self._obj_axis = obj_axis
        if obj_axis == "x":
            self._cellA_xy = (obj_center[0] - obj_sep, obj_center[1])
            self._cellB_xy = (obj_center[0] + obj_sep, obj_center[1])
            self._cell_names = ("4", "6")
        else:
            self._cellA_xy = (obj_center[0], obj_center[1] - obj_sep)
            self._cellB_xy = (obj_center[0], obj_center[1] + obj_sep)
            self._cell_names = ("2", "8")
        super().__init__(**kwargs)
        self._spec = {"obj_left_idx": 0, "target_cell": 0, "target_idx": 0}

    def set_episode(self, obj_left_idx, target_cell):
        li = int(obj_left_idx)
        tc = int(target_cell)
        self._spec = {
            "obj_left_idx": li,
            "target_cell": tc,
            "target_idx": li if tc == 0 else 1 - li,
        }

    @property
    def cellA_xy(self):
        return self._cellA_xy

    @property
    def cellB_xy(self):
        return self._cellB_xy

    @property
    def cell_names(self):
        return self._cell_names

    def object_positions(self, spec):
        return {"A": self._cellA_xy, "B": self._cellB_xy}

    def _reset_internal(self):
        super(Lift, self)._reset_internal()
        li = int(self._spec.get("obj_left_idx", 0))
        pos = self.object_positions(self._spec)
        self._place_free(self._objs[li], self.obj_body_ids[li], pos["A"])
        self._place_free(self._objs[1 - li], self.obj_body_ids[1 - li], pos["B"])
        self._place_free(self._basket, self.basket_body_id, self._basket_xy,
                         force_identity=True)
        self.sim.forward()
        for _ in range(self._settle_steps):
            self.sim.step()


def make_circle_env(view_size, obj_names=("milk", "tomato_sauce"),
                    obj_center=(0.0, 0.10), obj_sep=0.12, obj_axis="y",
                    basket_xy=(0.0, -0.18), extra_cams=()):
    controller = load_controller_config(default_controller="OSC_POSE")
    cams = ["agentview", "robot0_eye_in_hand"] + list(extra_cams)
    env = robosuite.make(
        env_name="KinovaTwoObjCircle",
        robots="Kinova3",
        gripper_types="MenagerieRobotiq2f85",
        controller_configs=controller,
        obj_names=obj_names,
        obj_center=obj_center,
        obj_sep=obj_sep,
        obj_axis=obj_axis,
        basket_xy=basket_xy,
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
    env = make_circle_env(128)
    for li, tc in [(0, 0), (0, 1), (1, 0)]:
        env.set_episode(li, tc)
        env.reset()
        print(f"obj_left={li} target_cell={tc} -> target={env.target_name} "
              f"@ {np.round(env.target_pos(), 3)}  cellA{env.cell_names[0]}={np.round(env.cellA_xy,3)} "
              f"cellB{env.cell_names[1]}={np.round(env.cellB_xy,3)}  basket={np.round(env.basket_pos(),3)}")
    os._exit(0)
