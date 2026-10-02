"""MuJoCo Menagerie Robotiq 2F-85 for robosuite 1.4 (built by fsim/assets/build_gripper.py).

menagerie_gripper_xml() and the action handling follow the teammate's stir-collect-sim/sim/kinova_env.py
(MenagerieRobotiq2f85). FsimRobotiq2f85 adds the knobs the furniture tasks need: grip force,
pad friction for twisting parts, and an opening cap (the design caps the gripper at 65 mm).
"""
from __future__ import annotations

import os

import numpy as np
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion

GRIPPER_XML = "grippers/robotiq_2f85_menagerie.xml"


def menagerie_gripper_xml() -> str:
    """First existing copy of the generated gripper XML ($FSIM_ASSETS, fsim/assets, $KINOVA_SIM_ASSETS, robosuite)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [os.path.join(os.path.expanduser(os.environ[k]), GRIPPER_XML)
                  for k in ("FSIM_ASSETS", "KINOVA_SIM_ASSETS") if os.environ.get(k)]
    candidates += [os.path.join(here, "assets", GRIPPER_XML), xml_path_completion(GRIPPER_XML)]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Menagerie Robotiq 2F-85 gripper XML not found; run: python fsim/assets/build_gripper.py --out fsim/assets\n"
        "Looked for:\n  " + "\n  ".join(candidates))


class FsimRobotiq2f85(GripperModel):
    """Robotiq 2F-85 with furniture-task settings.

    Actions follow robosuite's convention (-1 open, +1 close) and ramp by `speed` per physics
    substep like the teammate's gripper. `open_limit` (in [-1, 1]) caps the opening.
    """

    FORCE_RANGE = 5.0        # tendon force limit (Menagerie default 5)
    PAD_FRICTION = 1.0       # sliding friction of the finger pads
    PAD_PRIORITY = 2         # above the parts/environment (priority 1), so pad contacts use PAD_FRICTION
    OPEN_LIMIT = -0.484      # most-open command: 65 mm between the pads (tools/gripper_calib.py; -1 = 85 mm)

    def __init__(self, idn=0):
        super().__init__(menagerie_gripper_xml(), idn=idn)
        self.current_action = np.zeros(1)
        for act in self.actuator:
            act.set("forcerange", f"{-self.FORCE_RANGE} {self.FORCE_RANGE}")
        pads = {self.naming_prefix + n for group in ("left_finger", "right_finger") for n in self._important_geoms[group]}
        for geom in self.worldbody.iter("geom"):
            if geom.get("name") in pads:
                geom.set("friction", f"{self.PAD_FRICTION} 0.005 0.0001")
                geom.set("priority", str(self.PAD_PRIORITY))

    def format_action(self, action):
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), self.OPEN_LIMIT, 1.0)
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


GRIPPER_MAPPING["FsimRobotiq2f85"] = FsimRobotiq2f85
