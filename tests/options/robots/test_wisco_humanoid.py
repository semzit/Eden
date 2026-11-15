"""Wisco humanoid V2: the preset agrees with its asset, and its spawn pose stands on the soles.

Nothing here builds a scene: the asset is a local-only URDF whose mesh tree is fetched separately, so
these tests read the XML directly and skip when it is not on disk (the same contract as ``test_init``).
"""

from __future__ import annotations

import itertools
import os
from xml.etree import ElementTree

import numpy as np
import pytest

from eden.options import robots  # noqa: F401 -- populates ROBOT_REGISTRY
from eden.options.catalog import ROBOT_REGISTRY
from eden.options.robots.wisco_humanoid import (
    WISCO_DOFS_NAME,
    WISCO_EE_LINKS_NAME,
    WISCO_FEET_LINKS_NAME,
    WISCO_GANTRY_LINKS_NAME,
    WISCO_SENSOR_LINKS_NAME,
    WiscoHumanoidV2,
)
from eden.utils.actuator import NATURAL_FREQ
from eden.utils.string import resolve_matching_names_values


def _asset_path() -> str:
    """The preset's file under the asset root, or a skip: resolving it must never reach the network."""
    options = WiscoHumanoidV2()
    path = os.path.join(options.resolve_local_dir(), options.file)
    if not os.path.isfile(path):
        pytest.skip(
            f"{options.file!r} is not cached under {options.resolve_local_dir()}; "
            "place the URDF and its meshes/ tree there to run the asset-backed checks"
        )
    return path


@pytest.fixture(scope="module")
def urdf() -> ElementTree.Element:
    return ElementTree.parse(_asset_path()).getroot()


def test_the_preset_registers_under_its_snake_cased_class_name():
    assert ROBOT_REGISTRY.get("wisco_humanoid_v2") is WiscoHumanoidV2


def test_the_declared_dofs_are_exactly_the_assets_revolute_joints(urdf):
    """Order is Eden's, membership is the asset's: a renamed joint must fail here, not at build."""
    revolute = {j.get("name") for j in urdf.findall("joint") if j.get("type") == "revolute"}
    assert set(WISCO_DOFS_NAME) == revolute
    assert len(WISCO_DOFS_NAME) == len(set(WISCO_DOFS_NAME)) == 31


def test_every_dof_is_covered_by_exactly_one_armature_pattern():
    """``resolve_matching_names_values`` refuses an ambiguous name and a pattern that matches nothing."""
    pattern_map = WiscoHumanoidV2().default_dofs_armature
    idx, names, _ = resolve_matching_names_values(pattern_map, WISCO_DOFS_NAME)
    assert sorted(idx) == list(range(len(WISCO_DOFS_NAME))), f"uncovered DOFs: {set(WISCO_DOFS_NAME) - set(names)}"
    # Disjointness is what the matcher cannot check for us: two patterns matching one DOF silently
    # let the last dict entry win, and the DOF's real armature disappears.
    matched = {pattern: set(_names_matching(pattern)) for pattern in pattern_map}
    overlapping = [
        (a, b, sorted(matched[a] & matched[b]))
        for a, b in itertools.combinations(matched, 2)
        if matched[a] & matched[b]
    ]
    assert not overlapping, f"patterns match the same DOF: {overlapping}"


def _names_matching(pattern: str) -> list[str]:
    return resolve_matching_names_values({pattern: None}, WISCO_DOFS_NAME)[1]


def test_gains_are_the_raibert_heuristic_of_the_armatures_at_10_hz():
    """kp = A*w^2 and kd = 2*zeta*A*w with zeta=2, the over-damped ratio `eden.utils.actuator` uses."""
    robot = WiscoHumanoidV2()
    for pattern, armature in robot.default_dofs_armature.items():
        assert robot.default_dofs_kp[pattern] == pytest.approx(armature * NATURAL_FREQ**2)
        assert robot.default_dofs_kd[pattern] == pytest.approx(4.0 * armature * NATURAL_FREQ)


def test_effort_limits_are_left_to_the_asset():
    """Every joint has ``<limit effort>`` in the URDF, so restating them in the preset could only drift."""
    assert WiscoHumanoidV2().default_dofs_force_limits == {}


@pytest.mark.parametrize(
    "field, names",
    [
        ("feet_links_name", WISCO_FEET_LINKS_NAME),
        ("gantry_links_name", WISCO_GANTRY_LINKS_NAME),
        ("ee_links_name", WISCO_EE_LINKS_NAME),
        ("links_to_keep", WISCO_SENSOR_LINKS_NAME),
    ],
)
def test_declared_links_exist_in_the_asset(urdf, field, names):
    links = {link.get("name") for link in urdf.findall("link")}
    assert [name for name in names if name not in links] == []


def test_the_feet_are_kept_through_the_fixed_link_merge():
    robot = WiscoHumanoidV2()
    assert [foot for foot in robot.feet_links_name if foot not in robot.links_to_keep] == []


def test_the_zero_pose_stands_the_soles_on_the_ground(urdf):
    """``default_root_pos`` is a spawn height, and the sole offset belongs to the asset: pin them together.

    A CAD re-export that changes a leg link length moves the soles without moving anything a reviewer
    reads, and the robot then spawns buried in the floor or hovering.
    """
    for side in ("left", "right"):
        sole_z = _pelvis_frame_position(urdf, f"{side}_sole")[2]
        assert WiscoHumanoidV2().default_root_pos[2] + sole_z == pytest.approx(0.0, abs=2e-3), (
            f"{side} sole sits {sole_z:.4f} m below the pelvis at the zero pose; "
            f"default_root_pos[2] should be about {-sole_z:.4f}"
        )


def _pelvis_frame_position(urdf: ElementTree.Element, link_name: str) -> np.ndarray:
    """``link_name``'s origin in the pelvis frame at the zero pose (every revolute angle 0)."""
    by_child = {j.find("child").get("link"): j for j in urdf.findall("joint")}

    def pose(link: str) -> tuple[np.ndarray, np.ndarray]:
        if link not in by_child:
            return np.eye(3), np.zeros(3)
        joint = by_child[link]
        origin = joint.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()])
        rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
        parent_R, parent_p = pose(joint.find("parent").get("link"))
        # No revolute rotation survives the zero pose, so only the joint origin's own rpy rotates.
        return parent_R @ _rpy_to_matrix(rpy), parent_p + parent_R @ xyz

    return pose(link_name)[1]


def _rpy_to_matrix(rpy: list[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rot_z @ rot_y @ rot_x
