"""A mesh-only product must not enable an empty screw action space."""

import pytest
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.connection import Screw_Connection

from disassembly_pomdp.env_builder import Cell, build_env
from disassembly_pomdp.gripping_and_removing_capability import (
    GrippingAndRemovingCapability,
)
from disassembly_pomdp.screwing_capability import ScrewingCapability
from tests.conftest import FakePart


@pytest.mark.parametrize("with_screw", [False, True])
def test_enabled_tools_follow_product_connections(with_screw):
    assembly = Assembly("mesh_product")
    assembly.parts = {name: FakePart(name) for name in ("base", "target")}
    if with_screw:
        assembly.connections = [Screw_Connection("target", "base", screw_head="H5")]
    env = build_env(
        None, assembly=assembly, base_parts=["base"], target_parts=["target"]
    )
    expected = (
        [ScrewingCapability, GrippingAndRemovingCapability]
        if with_screw
        else [GrippingAndRemovingCapability]
    )
    assert [type(cap) for cap in env.capabilities] == expected
    assert env.initial_mounted_tool == ("H5" if with_screw else "default_gripper")


def test_product_without_screws_on_gripper_only_cell():
    assembly = Assembly("mesh_product")
    assembly.parts = {name: FakePart(name) for name in ("base", "target")}
    env = build_env(
        None,
        assembly=assembly,
        cell=Cell(screwdrivers=()),
        base_parts=["base"],
        target_parts=["target"],
    )
    assert env.initial_mounted_tool == "default_gripper"
