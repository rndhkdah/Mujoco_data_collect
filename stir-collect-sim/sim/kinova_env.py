#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET

import numpy as np
import robosuite
from robosuite import load_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial, xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import GRIPPER_MAPPING


GRIPPER_XML = "grippers/robotiq_2f85_menagerie.xml"


def menagerie_gripper_xml():
    here = os.path.dirname(os.path.abspath(__file__))
    env_root = os.environ.get("KINOVA_SIM_ASSETS")
    candidates = [
        os.path.join(os.path.expanduser(env_root), GRIPPER_XML) if env_root else None,
        os.path.join(here, "assets", GRIPPER_XML),
        xml_path_completion(GRIPPER_XML),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    shown = [p or f"$KINOVA_SIM_ASSETS/{GRIPPER_XML} (KINOVA_SIM_ASSETS is unset)"
             for p in candidates]
    raise FileNotFoundError(
        "Menagerie Robotiq 2F-85 gripper XML not found. Looked for:\n  "
        + "\n  ".join(shown)
        + "\nPlace robotiq_2f85_menagerie.xml and its meshes as described in "
        + os.path.join(here, "assets", "README.md"))


class MenagerieRobotiq2f85(GripperModel):

    def __init__(self, idn=0):
        super().__init__(menagerie_gripper_xml(), idn=idn)
        self.current_action = np.zeros(1)

    def format_action(self, action):
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.05

    @property
    def dof(self):
        return 1

    @property
    def init_qpos(self):
        return np.zeros(8)

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["left_pad1", "left_pad2"],
            "right_finger": ["right_pad1", "right_pad2"],
            "left_fingerpad": ["left_pad1"],
            "right_fingerpad": ["right_pad1"],
        }


GRIPPER_MAPPING["MenagerieRobotiq2f85"] = MenagerieRobotiq2f85


class KinovaPickToGoal(Lift):

    def __init__(self, goal_xy=(0.0, -0.15), goal_radius=0.06, **kwargs):
        self._goal_xy = tuple(goal_xy)
        self._goal_radius = float(goal_radius)
        super().__init__(**kwargs)

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

        redwood = CustomMaterial(
            texture="WoodRed", tex_name="redwood", mat_name="redwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        self.cube = BoxObject(
            name="cube",
            size=[0.025, 0.025, 0.04],
            rgba=[1, 0, 0, 1],
            material=redwood,
            density=400,
            friction=[1.5, 0.05, 0.0008],
        )

        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.cube,
            x_range=[-0.04, 0.04],
            y_range=[0.08, 0.14],
            rotation=0.0,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

        gx, gy = self._goal_xy
        gz = mujoco_arena.table_offset[2] + 0.002
        site = ET.Element("site", {
            "name": "goal_area",
            "pos": f"{gx} {gy} {gz}",
            "size": f"{self._goal_radius} 0.001",
            "type": "cylinder",
            "rgba": "0.1 0.9 0.1 0.45",
            "group": "1",
        })
        self.model.worldbody.append(site)

    def _check_success(self):
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        gx, gy = self._goal_xy
        dxy = np.linalg.norm(cube_pos[:2] - np.array([gx, gy]))
        table_h = self.model.mujoco_arena.table_offset[2]
        on_table = cube_pos[2] < table_h + 0.05
        return bool(dxy < self._goal_radius and on_table)


def make_kinova_env(view_size):
    controller = load_controller_config(default_controller="OSC_POSE")
    env = robosuite.make(
        env_name="KinovaPickToGoal",
        robots="Kinova3",
        gripper_types="MenagerieRobotiq2f85",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=view_size,
        camera_widths=view_size,
        control_freq=20,
        ignore_done=True,
        horizon=100000,
        reward_shaping=True,
    )

    class _Wrap:
        def __init__(self, e): self.env = e
        def reset(self): return self.env.reset()
        def step(self, a): return self.env.step(a)
        def close(self): return self.env.close()

    return _Wrap(env), "Kinova Gen3: pick the cube and place it in the green goal area"


if __name__ == "__main__":
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    env, desc = make_kinova_env(256)
    print("task:", desc)
    obs = env.reset()
    print("action_dim:", env.env.action_dim)
    for k in ["agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat"]:
        print(f"  {k}:", None if k not in obs else np.asarray(obs[k]).shape)
    a = np.zeros(env.env.action_dim); a[-1] = -1
    for _ in range(5):
        obs, r, d, i = env.step(a)
    print("stepped OK; success:", env.env._check_success())
    os._exit(0)
