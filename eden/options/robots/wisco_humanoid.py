"""Wisco humanoid V2 (31 DoF) configuration.

The asset is a CAD-exported URDF (``wiscohumanoids``) whose joint limits, damping and effort limits come
from MyActuator data sheets; the mesh tree next to it is required (``meshes/*.stl``).
"""

from __future__ import annotations

from typing import Any, ClassVar

from genesis.typing import Vec3FType

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions
from eden.utils.actuator import NATURAL_FREQ


# 31 actuated joints: both legs, the 3-DoF waist, the 2-DoF head, both 7-DoF arms. The order is Eden's
# convention for humanoids (legs, waist, head, arms) so a term written against G1Dofs29 / Oli indexes
# the same joints with the same indices.
WISCO_DOFS_NAME: list[str] = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "torso_yaw",
    "torso_roll",
    "torso_pitch",
    "head_yaw",
    "head_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_forearm_yaw",
    "left_wrist_pitch",
    "left_wrist_roll",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_forearm_yaw",
    "right_wrist_pitch",
    "right_wrist_roll",
]


# The ``*_sole`` frames are the foot's ground-contact plane: massless fixed frames 4.4 cm below the
# ankle-roll links (``0.0502, 0.0001, -0.7015`` in the pelvis frame at the zero pose). They are what the
# robot stands on, so they feed both ``feet_links_name`` and ``links_to_keep`` — Genesis merges fixed
# childless links away, and a merged-away foot fails only at deploy time, inside the velocity estimator.
WISCO_FEET_LINKS_NAME: tuple[str, ...] = ("left_sole", "right_sole")

# The most proximal link of each shoulder: its frame origin sits on the shoulder-pitch axis and does not
# translate as the arm swings, so a bring-up gantry hung there stays put through an arm-throwing clip.
WISCO_GANTRY_LINKS_NAME: tuple[str, ...] = ("left_shoulder_pitch_link", "right_shoulder_pitch_link")

# ``ee_links_name[0]`` is the IK frame; the rigs pick the per-side wrist for a hand mount.
WISCO_EE_LINKS_NAME: list[str] = ["left_hand", "right_hand"]

# Sensor frames a task resolves by name. Both are massless fixed links, so they must be named here or
# Genesis merges them into ``head`` / ``pelvis`` and every lookup fails.
WISCO_SENSOR_LINKS_NAME: tuple[str, ...] = ("imu_link", "head_camera", "head_camera_optical")


# Reflected rotor inertia (armature) per joint group, keyed by DOF pattern. PLACEHOLDER VALUES: the URDF
# carries the vendor's damping, friction and effort limits, but URDF has no armature element and the
# data sheets' rotor inertia is not in the repo, so these scale with the motor's effort class (120 / 86 /
# 60 / 35 / 10 N*m) after the Unitree N7520 ratio of armature to peak torque (~1.9e-4 kg*m^2 per N*m).
# Replace each with ``reflected_inertia(motor, gearbox)`` once ``training/model/motor_specs.json`` is
# available; every kp / kd below follows from this dict.
#
# These are deliberately NOT an ``ActuatorSpecOptions`` / ``dofs_spec``: a spec's ``DAMPING`` (0.0 by
# default) *overrides* the per-joint damping the asset authors, so mapping a spec here would silently
# zero the vendor's joint friction model. Add the spec together with an explicit ``default_dofs_damping``
# per joint if you want the torque-speed actuator modifiers.
_WISCO_ARMATURE: dict[str, float] = {
    # 120 N*m class.
    "*_hip_pitch": 0.022,
    "*_hip_roll": 0.022,
    "*_knee": 0.022,
    "torso_yaw": 0.022,
    # 86 / 83 / 78 N*m class.
    "*_ankle_pitch": 0.015,
    "torso_roll": 0.015,
    "torso_pitch": 0.015,
    # 60 N*m class.
    "*_hip_yaw": 0.011,
    "*_shoulder_pitch": 0.011,
    "*_shoulder_roll": 0.011,
    # 35 / 34 N*m class.
    "*_ankle_roll": 0.0065,
    "*_shoulder_yaw": 0.0065,
    "*_elbow": 0.0065,
    "*_forearm_yaw": 0.0065,
    # 10 N*m class.
    "head_*": 0.002,
    "*_wrist_*": 0.002,
}

# Raibert heuristic (``eden.utils.actuator``) at the global 10 Hz closed-loop bandwidth, with the
# over-damped damping ratio of 2.0. Written against the armature dict rather than an ActuatorSpecOptions
# because no spec is mapped (see above): kp = A*w^2, kd = 2*zeta*A*w.
_ACTION_DAMPING_RATIO: float = 2.0


def _kp(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _kd(armature: float) -> float:
    return 2.0 * _ACTION_DAMPING_RATIO * armature * NATURAL_FREQ


@ROBOT_REGISTRY.register()
class WiscoHumanoidV2(RobotOptions):
    """Wisco humanoid V2, a 31-DoF free-base humanoid (40.55 kg total)."""

    file: str = "humanoids/wisco_humanoid_v2/robot.urdf"

    is_fixed_base: bool = False

    links_to_keep: list[str] = [*WISCO_FEET_LINKS_NAME, *WISCO_SENSOR_LINKS_NAME]
    feet_links_name: ClassVar[tuple[str, ...]] = WISCO_FEET_LINKS_NAME
    gantry_links_name: ClassVar[tuple[str, ...]] = WISCO_GANTRY_LINKS_NAME
    ee_links_name: ClassVar[list[str]] = WISCO_EE_LINKS_NAME

    dofs_name: list[str] = WISCO_DOFS_NAME

    # Straight legs at the zero pose put both soles 0.7015 m below the pelvis link frame (forward
    # kinematics over the URDF's joint origins), so the pelvis spawns 0.5 mm above a feet-flat stance.
    # No nominal stance ships with the asset: a reset that wants a bent knee should also lower this by
    # ~0.1 m per 0.3 rad of knee, and compensate the CoM shift the bent pose carries backwards.
    default_root_pos: Vec3FType = (0.0, 0.0, 0.702)
    default_dofs_pos: dict[str, float] = {"*": 0.0}

    default_dofs_kp: dict[str, float] = {pattern: _kp(a) for pattern, a in _WISCO_ARMATURE.items()}
    default_dofs_kd: dict[str, float] = {pattern: _kd(a) for pattern, a in _WISCO_ARMATURE.items()}
    default_dofs_armature: dict[str, float] = dict(_WISCO_ARMATURE)

    # Effort limits are NOT restated: every joint's ``<limit effort>`` is in the asset (120 N*m hip /
    # knee / torso-yaw down to 10 N*m head / wrist), and Genesis writes it as the DOF's force range.
    metadata: ClassVar[dict[str, Any]] = {
        # No nominal height ships either; this is the zero-pose pelvis height plus the head link's
        # 0.593 m offset, i.e. the standing silhouette without the head mesh's own extent.
        "robot_height": 1.30,
    }
