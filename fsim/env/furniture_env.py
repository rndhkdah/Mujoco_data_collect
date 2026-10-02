"""KinovaFurnitureEnv: FurnitureBench-style assembly tasks in robosuite 1.4 (Kinova3 + Robotiq 2F-85).

Per step: robosuite runs the OSC controller every physics substep; `_pre_action` also runs the
helical screw manager there. `reward` is the number of newly assembled pairs (latching), and
`info` carries the phase label and screw events. Reset samples a layout (none/low/med/high),
places and settles the parts while holding the arm, and re-arms all managers.
"""
from __future__ import annotations

import copy
import os
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.tasks import ManipulationTask

import fsim.env.gripper  # noqa: F401  (registers FsimRobotiq2f85)
from fsim.env.arena import FurnitureArena
from fsim.env.gripper import menagerie_gripper_xml
from fsim.env.objects import build_part_objects
from fsim.geometry import mat_to_quat
from fsim.models import names
from fsim.models.meshes import build_task_meshes, generated_dir
from fsim.models.parts import build_screw_equalities
from fsim.sim.assembly import AssemblyChecker
from fsim.sim.phases import PhaseTracker
from fsim.sim.reset import ResetSampler, place_and_settle
from fsim.sim.screw import HelicalScrewManager
from fsim.sim.state import PartState
from fsim.spec import get_task
from fsim.spec.types import TaskSpec

# Gripper pointing down, 17 cm above the table between the legs and the tabletop
# (solved by tools/reach_check.py, which also checks every grasp/insertion pose of one_leg).
HOME_QPOS = (-0.3381, 0.4968, 0.4154, 1.7152, -0.2339, 0.9787, -2.9775)

_GENERATED_RE = re.compile(r"[/\\]([A-Za-z0-9_]+-[0-9a-f]{10})[/\\]([^/\\]+)$")


class KinovaFurnitureEnv(SingleArmEnv):
    def __init__(
        self,
        robots="Kinova3",
        task: str | TaskSpec = "one_leg",
        randomness: str = "low",
        high_index: int | None = None,
        seed: int | None = None,
        gripper_types="FsimRobotiq2f85",
        controller_configs=None,
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="front",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=3000,
        ignore_done=True,
        hard_reset=False,
        camera_names=("front", "robot0_eye_in_hand", "rear"),
        camera_heights=224,
        camera_widths=224,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        home_qpos=None,
        settle_steps: int = 150,
        integrator: str = "implicitfast",
    ):
        self.task = get_task(task) if isinstance(task, str) else task
        self.randomness = randomness
        self.high_index = high_index
        self.rng = np.random.default_rng(seed)
        self.home_qpos = np.asarray(home_qpos if home_qpos is not None else HOME_QPOS, float) \
            if (home_qpos is not None or HOME_QPOS is not None) else None
        self.settle_steps = settle_steps
        self.integrator = integrator
        self.sampler = ResetSampler(self.task)
        self.checker = AssemblyChecker(self.task)
        self.layout = self.task.layout
        self.phases = PhaseTracker(self.task, grasped=self.grasped)
        self._event_cursor = 0
        self.controller_configs_arg = copy.deepcopy(controller_configs)   # JSON-safe copy for env_args
        super().__init__(
            robots=robots, env_configuration="default", controller_configs=controller_configs,
            mount_types=None, gripper_types=gripper_types, initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs, has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera, render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh, render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq, horizon=horizon, ignore_done=ignore_done, hard_reset=hard_reset,
            camera_names=list(camera_names), camera_heights=camera_heights, camera_widths=camera_widths,
            camera_depths=camera_depths, camera_segmentations=camera_segmentations,
            renderer=renderer, renderer_config=renderer_config,
        )

    # ------------------------------------------------------------------ model
    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(self.task.scene.robot_base)
        arena = FurnitureArena(self.task)
        arena.set_origin([0, 0, 0])
        self.part_objects = build_part_objects(self.task)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=self.part_objects)
        for eq in build_screw_equalities(self.task):
            self.model.equality.append(eq)

    def _setup_references(self):
        super()._setup_references()
        m, d = self.sim.model._model, self.sim.data._data
        if self.integrator == "implicitfast":
            m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
        self.part_state = PartState(m, d, self.task)
        self.screw = HelicalScrewManager(m, d, self.task)
        self.part_geoms = {}
        for p, b in self.part_state.body_id.items():
            self.part_geoms[p] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)
                                  if m.geom_bodyid[g] == b and m.geom_contype[g] | m.geom_conaffinity[g]]
        if self.home_qpos is not None:
            self.robots[0].init_qpos = self.home_qpos.copy()

    # ------------------------------------------------------------------ reset
    def _hold_robot(self):
        r = self.robots[0]
        d = self.sim.data._data
        d.qpos[r._ref_joint_pos_indexes] = self._held_q
        d.qvel[r._ref_joint_vel_indexes] = 0.0
        if r._ref_gripper_joint_pos_indexes is not None:
            d.qpos[r._ref_gripper_joint_pos_indexes] = self._held_gq
            d.qvel[r._ref_gripper_joint_vel_indexes] = 0.0

    def _reset_internal(self):
        super()._reset_internal()
        m, d = self.sim.model._model, self.sim.data._data
        self.screw.reset()
        self.checker.reset()
        self._event_cursor = 0
        r = self.robots[0]
        self._held_q = d.qpos[r._ref_joint_pos_indexes].copy()
        self._held_gq = d.qpos[r._ref_gripper_joint_pos_indexes].copy() if r._ref_gripper_joint_pos_indexes is not None else None
        self.layout = self.sampler.sample(self.rng, self.randomness, self.high_index)
        place_and_settle(m, d, self.part_state, self.layout, settle_steps=self.settle_steps, after_step=self._hold_robot)
        self._hold_robot()
        mujoco.mj_forward(m, d)
        self.phases.reset(self.parts_poses_W())

    # ------------------------------------------------------------------ step hooks
    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        self.screw.update()

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        poses_W = self.parts_poses_W()
        info["skill"] = bool(self.phases.update(poses_W, self.screw, self.checker.assembled))
        info["phase"] = self.phases.phase
        info["screw_events"] = self.screw.events[self._event_cursor:]
        self._event_cursor = len(self.screw.events)
        info["success"] = self._check_success()
        return reward, done, info

    def reward(self, action=None):
        return float(self.checker.update(self.part_state.poses()))

    def _check_success(self):
        return self.checker.all_assembled

    # ------------------------------------------------------------------ task helpers
    def grasped(self, part: str) -> bool:
        return bool(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.part_geoms[part]))

    def parts_poses_W(self) -> dict[str, np.ndarray]:
        return {p: self.part_state.pose_W(p) for p in self.task.part_names}

    def parts_pose_vector(self) -> np.ndarray:
        """Part poses in the table frame W as one vector: [pos(3), quat xyzw(4)] per part."""
        out = []
        for p in self.task.part_names:
            T = self.part_state.pose_W(p)
            q = mat_to_quat(T[:3, :3])
            out += [*T[:3, 3], q[1], q[2], q[3], q[0]]
        return np.asarray(out, np.float32)

    def eef_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """(linear, angular) velocity of the grip site in the world frame."""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model._model, self.sim.data._data, mujoco.mjtObj.mjOBJ_SITE,
                                 self.robots[0].eef_site_id, res, 0)
        return res[3:].copy(), res[:3].copy()

    def gripper_width(self) -> float:
        """Gap between the finger pad surfaces (m)."""
        m, d = self.sim.model._model, self.sim.data._data
        ids = getattr(self, "_pad_ids", None)
        if ids is None:
            ids = self._pad_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"gripper0_{s}_pad1") for s in ("left", "right")]
        return float(np.linalg.norm(d.geom_xpos[ids[0]] - d.geom_xpos[ids[1]]) - 2 * m.geom_size[ids[0]][1])

    def fsim_snapshot(self) -> dict:
        """Task state not contained in the MuJoCo state (for recording and replay)."""
        return {"screw": self.screw.snapshot(), "assembled": sorted(map(list, self.checker.assembled)),
                "phase": self.phases.phase,
                "gripper_action": float(self.robots[0].gripper.current_action[0])}

    def restore_snapshot(self, snap: dict):
        """Re-apply task state recorded by fsim_snapshot (after restoring the MuJoCo state)."""
        self.screw.restore(snap["screw"])
        self.checker.assembled = {tuple(p) for p in snap.get("assembled", [])}
        self.phases.phase = int(snap.get("phase", 0))
        self.robots[0].gripper.current_action = np.array([snap.get("gripper_action", -1.0)])

    # ------------------------------------------------------------------ replay support
    def edit_model_xml(self, xml_str):
        """Relocate asset paths of a recorded model: robosuite assets, generated part meshes and the
        Menagerie gripper. Generated meshes are rebuilt if missing (the hash in the path must match)."""
        import robosuite

        rs_root = os.path.dirname(robosuite.__file__)
        root = ET.fromstring(xml_str)
        asset = root.find("asset")
        grippers = os.path.dirname(menagerie_gripper_xml())
        for el in list(asset.iter("mesh")) + list(asset.iter("texture")):
            path = el.get("file")
            if path is None:
                continue
            parts = re.split(r"[/\\]", path)
            if "robosuite" in parts:
                i = max(k for k, v in enumerate(parts) if v == "robosuite")
                el.set("file", os.path.join(rs_root, *parts[i + 1:]))
            elif "grippers" in parts:
                i = max(k for k, v in enumerate(parts) if v == "grippers")
                el.set("file", os.path.join(grippers, *parts[i + 1:]))
            else:
                m = _GENERATED_RE.search(path)
                if m:
                    target = os.path.join(generated_dir(), m.group(1), m.group(2))
                    if not os.path.isfile(target):
                        build_task_meshes(self.task)
                    if not os.path.isfile(target):
                        raise FileNotFoundError(f"{target}: recorded with a different part spec ({m.group(1)})")
                    el.set("file", target)
        return ET.tostring(root, encoding="unicode")


def make_env(task="one_leg", randomness="low", seed=None, camera_size=224, **kwargs) -> KinovaFurnitureEnv:
    """Construct the env with robosuite's OSC_POSE controller (the teammate's action convention)."""
    from robosuite import load_controller_config

    return KinovaFurnitureEnv(task=task, randomness=randomness, seed=seed,
                              controller_configs=load_controller_config(default_controller="OSC_POSE"),
                              camera_heights=camera_size, camera_widths=camera_size, **kwargs)
