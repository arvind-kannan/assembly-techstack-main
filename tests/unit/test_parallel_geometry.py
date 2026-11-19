"""Geometry screening uses real meshes, independently of robot planners."""

from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent
from disassembly_pomdp.parallel_geometry import (
    ParallelGeometrySettings,
    check_parallel_geometry,
)
from tests.unit.test_graph_parallel import agent_for


def scene(positions, size=2):
    parts = {}
    for name, position in positions.items():
        mesh = o3d.geometry.TriangleMesh.create_box(size, size, size)
        mesh.translate(np.asarray(position) - size / 2)
        parts[name] = SimpleNamespace(name=name, mesh=mesh)
    return SimpleNamespace(parts=parts)


def ops(a=(0, 1, 0), b=(0, 1, 0), duration_b=1):
    return [
        dict(target_part="A", direction=a, action_time=1),
        dict(target_part="B", direction=b, action_time=duration_b),
    ]


def test_separated_pulls_pass_without_mutating_meshes():
    assembly = scene({"A": (0, 0, 0), "B": (20, 0, 0)})
    before = np.asarray(assembly.parts["A"].mesh.vertices).copy()
    accepted, details = check_parallel_geometry(
        assembly, ops(), ParallelGeometrySettings()
    )
    assert accepted and details["samples"] > 1
    np.testing.assert_array_equal(before, np.asarray(assembly.parts["A"].mesh.vertices))


def test_surface_gap_proximity_is_configurable():
    assembly = scene({"A": (0, 0, 0), "B": (6, 0, 0)})
    assert (
        check_parallel_geometry(assembly, ops(), ParallelGeometrySettings())[1][
            "reason"
        ]
        == "parts_too_close"
    )
    assert check_parallel_geometry(
        assembly, ops(), ParallelGeometrySettings(min_separation=3)
    )[0]


def test_crossing_rays_reject_before_mesh_path(monkeypatch):
    assembly = scene({"A": (-10, 0, 0), "B": (0, -10, 0)})

    def unexpected(*args):
        pytest.fail("mesh path should not run after ray rejection")

    monkeypatch.setattr("disassembly_pomdp.parallel_geometry._collision", unexpected)
    assert (
        check_parallel_geometry(
            assembly, ops((1, 0, 0), (0, 1, 0)), ParallelGeometrySettings()
        )[1]["reason"]
        == "removal_rays_intersect"
    )


def test_mesh_volume_collision_with_nonintersecting_centre_rays():
    assembly = scene({"A": (-10, 0, 0), "B": (0, -10, 1.5)})
    accepted, details = check_parallel_geometry(
        assembly, ops((1, 0, 0), (0, 1, 0)), ParallelGeometrySettings(ray_clearance=0.1)
    )
    assert not accepted and details["reason"] == "moving_parts_collision"


def test_obstacle_further_along_path_rejected():
    assembly = scene({"A": (0, 0, 0), "B": (20, 0, 0), "Wall": (0, 10, 0)})
    accepted, details = check_parallel_geometry(
        assembly, ops(), ParallelGeometrySettings(stationary_penetration=0)
    )
    assert not accepted and details["reason"] == "stationary_part_collision"


def test_missing_direction_and_bad_geometry_fail_closed():
    assembly = scene({"A": (0, 0, 0), "B": (20, 0, 0)})
    operations = ops()
    del operations[0]["direction"]
    assert not check_parallel_geometry(
        assembly, operations, ParallelGeometrySettings()
    )[0]
    assembly.parts["A"].mesh = None
    assert not check_parallel_geometry(assembly, ops(), ParallelGeometrySettings())[0]


def test_sample_limit_rejects_without_coarsening():
    assembly = scene({"A": (0, 0, 0), "B": (20, 0, 0)})
    assert (
        check_parallel_geometry(
            assembly, ops(), ParallelGeometrySettings(max_samples=2)
        )[1]["reason"]
        == "parallel_sample_limit"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_size": 0},
        {"min_separation": -1},
        {"ray_clearance": float("nan")},
        {"max_samples": 1},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ParallelGeometrySettings(**kwargs)


def test_agent_rejects_close_parts_before_applying_removals():
    agent = agent_for(["A", "B"])
    geometry = scene({"A": (0, 0, 0), "B": (6, 0, 0), "Base": (30, 0, 0)})
    for name, part in agent.base_assembly.parts.items():
        part.mesh = geometry.parts[name].mesh
    steps = agent.available_steps(agent.base_assembly)
    for _, attrs in steps:
        attrs["direction"] = [0, 1, 0]
    assert (
        DisassemblyGraphAgent._try_parallel_actions(agent, agent.base_assembly, steps)
        is None
    )
    assert agent.last_parallel_check["reason"] == "parts_too_close"
    assert len(agent.base_assembly.parts) == 3


def test_three_parts_and_different_completion_times():
    assembly = scene({"A": (0, 0, 0), "B": (20, 0, 0), "C": (40, 0, 0)})
    operations = ops(duration_b=2)
    operations.append(dict(target_part="C", direction=[0, 1, 0], action_time=3))
    assert check_parallel_geometry(assembly, operations, ParallelGeometrySettings())[0]


@pytest.mark.parametrize(
    "direction,duration", [([0, 0, 0], 1), ([0, 1, 0], 0), ([float("nan"), 0, 1], 1)]
)
def test_invalid_direction_or_duration(direction, duration):
    assembly = scene({"A": (0, 0, 0), "B": (20, 0, 0)})
    operations = ops()
    operations[0].update(direction=direction, action_time=duration)
    assert not check_parallel_geometry(
        assembly, operations, ParallelGeometrySettings()
    )[0]


def test_agent_accepts_clear_group_and_records_screening():
    agent = agent_for(["A", "B"])
    geometry = scene({"A": (0, 0, 0), "B": (20, 0, 0), "Base": (40, 0, 0)})
    for name, part in agent.base_assembly.parts.items():
        part.mesh = geometry.parts[name].mesh
    steps = agent.available_steps(agent.base_assembly)
    for _, attrs in steps:
        attrs["direction"] = [0, 1, 0]
    result = agent._try_parallel_actions(agent.base_assembly, steps)
    assert result is not None
    successor, attrs = result
    assert set(successor.parts) == {"Base"}
    assert not successor.connections
    assert attrs["feasibility"] == "geometry_screened"
    assert attrs["geometry_check"]["samples"] > 1
    assert attrs["action_time"] == 12
    assert set(agent.base_assembly.parts) == {"A", "B", "Base"}
