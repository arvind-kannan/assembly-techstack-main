from __future__ import annotations

import numpy as np
import pytest
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.connection import Screw_Connection

from tests.conftest import (
    DummyScrewingOperation,
    FakePart,
    build_capability_with_planner,
    raw_action_for,
    set_dof,
    set_visibility,
    successor_without_screw,
)

# ---------------------------------------------------------------------
# Action-space construction and stable global indexing
# ---------------------------------------------------------------------


def test_build_action_space_uses_tools_and_global_screw_connections(
    capability,
    two_screw_assembly,
):
    space = capability.build_action_space(two_screw_assembly)

    assert list(space.nvec) == [2, 2]
    assert capability._global_screw_conn_keys == [
        ("screw_1", "base_1", "H5"),
        ("screw_2", "base_1", "T6"),
    ]


def test_global_connection_index_remains_stable_after_connection_removed(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")

    # Simulate that the screw connection has already been removed.
    simple_screw_assembly.connections = []

    action = capability.decode_action(simple_screw_assembly, raw)

    assert list(capability.action_space.nvec) == [2, 1]
    assert action.connection is None
    assert action.screw_part_name == "screw_1"
    assert action.base_part_name == "base_1"


# ---------------------------------------------------------------------
# Decoding raw actions
# ---------------------------------------------------------------------


def test_decode_action_returns_semantic_screwing_action(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")

    action = capability.decode_action(simple_screw_assembly, raw)

    assert action.tool_name == "H5"
    assert isinstance(action.connection, Screw_Connection)
    assert action.screw_part_name == "screw_1"
    assert action.base_part_name == "base_1"
    assert action.operation is None


def test_decode_action_rejects_out_of_range_tool_index(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)

    with pytest.raises(IndexError):
        capability.decode_action(simple_screw_assembly, np.array([99, 0]))


def test_decode_action_rejects_out_of_range_connection_index(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)

    with pytest.raises(IndexError):
        capability.decode_action(simple_screw_assembly, np.array([0, 99]))


# ---------------------------------------------------------------------
# Target reference and target presence
# ---------------------------------------------------------------------


def test_get_target_ref_returns_part_when_screw_exists(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")

    target = capability.get_target_ref(simple_screw_assembly, raw)

    assert target is not None
    assert target.kind == "part"
    assert target.name == "screw_1"


def test_get_target_ref_distinguishes_never_existed_screw(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")

    # Fresh capability has not seen screw_1 before.
    simple_screw_assembly.parts.pop("screw_1", None)

    target = capability.get_target_ref(simple_screw_assembly, raw)

    assert target is not None
    assert target.kind == "part_never_existed"
    assert target.name == "screw_1"


def test_get_target_ref_treats_removed_previous_screw_as_part(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")

    first_target = capability.get_target_ref(simple_screw_assembly, raw)
    assert first_target.kind == "part"

    simple_screw_assembly.parts.pop("screw_1", None)
    second_target = capability.get_target_ref(simple_screw_assembly, raw)

    assert second_target.kind == "part"
    assert second_target.name == "screw_1"


def test_is_target_present_checks_assembly_part_presence(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")
    target = capability.get_target_ref(simple_screw_assembly, raw)

    assert capability.is_target_present(simple_screw_assembly, target)

    simple_screw_assembly.parts.pop("screw_1", None)

    assert not capability.is_target_present(simple_screw_assembly, target)


# ---------------------------------------------------------------------
# Pre-decode mask reasons
# ---------------------------------------------------------------------


def test_predecode_mask_wrong_mounted_tool(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")

    masked, reason, details = capability.predecode_mask_reason(
        simple_screw_assembly,
        raw,
        current_tool_name="T6",
    )

    assert masked
    assert reason == "wrong_tool"
    assert details["mounted"] == "T6"
    assert details["required"] == "H5"


def test_predecode_mask_wrong_screw_head(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "T6", "screw_1", "base_1", "H5")

    masked, reason, details = capability.predecode_mask_reason(
        simple_screw_assembly,
        raw,
        current_tool_name="T6",
    )

    assert masked
    assert reason == "wrong_tool"
    assert details["why"] == "head_mismatch"
    assert details["tool"] == "T6"
    assert details["head"] == "H5"


def test_predecode_mask_missing_head():
    from disassembly_pomdp.screwing_capability import ScrewingCapability

    asm = Assembly("missing_head_assembly")
    asm.parts = {
        "screw_1": FakePart("screw_1"),
        "base_1": FakePart("base_1"),
    }
    asm.connections = [Screw_Connection("screw_1", "base_1", None)]
    asm._fcl_objects = {}

    cap = ScrewingCapability(["H5"], operation_cls=DummyScrewingOperation)
    cap.build_action_space(asm)

    masked, reason, details = cap.predecode_mask_reason(
        asm,
        np.array([0, 0]),
        current_tool_name="H5",
    )

    assert masked
    assert reason == "wrong_tool"
    assert details["why"] == "missing_head"


def test_predecode_mask_empty_dof(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")

    set_dof(simple_screw_assembly, [])

    masked, reason, details = capability.predecode_mask_reason(
        simple_screw_assembly,
        raw,
        current_tool_name="H5",
    )

    assert masked
    assert reason == "geometrically_infeasible"
    assert details["why"] == "empty_dof"


def test_predecode_mask_not_visible(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")

    set_dof(simple_screw_assembly, [np.array([0.0, 0.0, 1.0])])
    set_visibility(simple_screw_assembly, False)

    masked, reason, details = capability.predecode_mask_reason(
        simple_screw_assembly,
        raw,
        current_tool_name="H5",
    )

    assert masked
    assert reason == "geometrically_infeasible"
    assert details["why"] == "not_visible"


def test_predecode_mask_accepts_valid_action(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")

    set_dof(simple_screw_assembly, [np.array([0.0, 0.0, 1.0])])
    set_visibility(simple_screw_assembly, True)

    masked, reason, details = capability.predecode_mask_reason(
        simple_screw_assembly,
        raw,
        current_tool_name="H5",
    )

    assert not masked
    assert reason is None
    assert details == {}


def test_predecode_mask_blocks_a_robot_that_holds_a_part_per_the_cell_state(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")

    set_dof(simple_screw_assembly, [np.array([0.0, 0.0, 1.0])])
    set_visibility(simple_screw_assembly, True)

    masked, reason, details = capability.predecode_mask_reason(
        simple_screw_assembly,
        raw,
        current_tool_name="H5",
        cell_state=CellState(holds={"robot_1": "part_1"}),
    )

    assert masked
    assert reason == "agent_busy"
    assert details["agent"] == "robot_1"


# ---------------------------------------------------------------------
# Operation construction
# ---------------------------------------------------------------------


def test_make_operation_requires_operation_class(simple_screw_assembly):
    from disassembly_pomdp.screwing_capability import ScrewingCapability

    cap = ScrewingCapability(["H5"], operation_cls=None)
    cap.build_action_space(simple_screw_assembly)
    action = cap.decode_action(simple_screw_assembly, np.array([0, 0]))

    with pytest.raises(RuntimeError):
        cap.make_operation(simple_screw_assembly, action)


def test_make_operation_creates_screwing_operation_shell(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")
    action = capability.decode_action(simple_screw_assembly, raw)

    op = capability.make_operation(simple_screw_assembly, action)

    assert isinstance(op, DummyScrewingOperation)
    assert op.data["type"] == "screwing"
    assert op.data["tool_name"] == "H5"
    assert op.data["screw_part_name"] == "screw_1"
    assert op.data["base_part_name"] == "base_1"
    assert op.data["connection"] == action.connection
    assert op.data["cost"] == 2.0
    assert op.data["planned"] is False


# ---------------------------------------------------------------------
# Planning wrapper, motion program, and time estimation
# ---------------------------------------------------------------------


def test_plan_motion_stores_planner_successor_and_motion_program(
    simple_screw_assembly,
    dummy_settings,
):
    def plan_func(assembly, op):
        op.data["planned"] = True
        op.data["direction"] = np.array([0.0, 0.0, 2.0])
        op.data["path_duration"] = 4.0
        op.data["path"] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
            ]
        )
        return op, successor_without_screw(assembly, "screw_1")

    cap = build_capability_with_planner(plan_func, dummy_settings)
    cap.build_action_space(simple_screw_assembly)
    raw = raw_action_for(cap, "H5", "screw_1", "base_1", "H5")
    action = cap.decode_action(simple_screw_assembly, raw)

    rng = np.random.default_rng(0)
    planned_action = cap.plan_motion(simple_screw_assembly, action, rng=rng)

    op = planned_action.operation
    assert op is not None
    assert op.data["planned"] is True
    assert "result_assembly" in op.data
    assert np.allclose(op.data["direction"], np.array([0.0, 0.0, 1.0]))
    assert 5.0 <= op.data["screw_search_duration"] <= 10.0
    assert "motion_program" in op.data
    assert op.data["motion_program"][0]["type"] == "screw_core"
    assert op.data["motion_program"][-1]["type"] == "dropoff"
    assert op.data["motion_duration"] > op.data["path_duration"]


def test_plan_motion_falls_back_to_dof_direction(
    simple_screw_assembly,
    dummy_settings,
):
    def plan_func(assembly, op):
        op.data["planned"] = True
        op.data["path_duration"] = 1.0
        op.data["path"] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return op

    set_dof(simple_screw_assembly, [np.array([0.0, 0.0, 3.0])])

    cap = build_capability_with_planner(plan_func, dummy_settings)
    cap.build_action_space(simple_screw_assembly)
    raw = raw_action_for(cap, "H5", "screw_1", "base_1", "H5")
    action = cap.decode_action(simple_screw_assembly, raw)

    planned_action = cap.plan_motion(simple_screw_assembly, action)

    assert np.allclose(planned_action.operation.data["direction"], [0.0, 0.0, 1.0])


def test_estimate_action_time_uses_motion_duration_when_available(
    capability,
    simple_screw_assembly,
    dummy_settings,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")
    action = capability.decode_action(simple_screw_assembly, raw)
    action.operation = DummyScrewingOperation({"motion_duration": 12.5})

    assert (
        capability.estimate_action_time(
            simple_screw_assembly,
            action,
            dummy_settings,
        )
        == 12.5
    )


def test_estimate_action_time_uses_symbolic_fallback_without_motion_program(
    capability,
    simple_screw_assembly,
    dummy_settings,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")
    action = capability.decode_action(simple_screw_assembly, raw)

    t = capability.estimate_action_time(
        simple_screw_assembly,
        action,
        dummy_settings,
    )

    assert t > 0.0


# ---------------------------------------------------------------------
# Deterministic transition and semantic post-check
# ---------------------------------------------------------------------


def test_deterministic_transition_accepts_successor_only_if_screw_removed(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")
    action = capability.decode_action(simple_screw_assembly, raw)

    result_assembly = successor_without_screw(simple_screw_assembly, "screw_1")
    action.operation = DummyScrewingOperation(
        {
            "result_assembly": result_assembly,
            "path": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        }
    )

    out = capability.deterministic_transition(simple_screw_assembly, action)

    assert out is result_assembly
    assert out.get_part_by_name("screw_1") is None


def test_deterministic_transition_rejects_a_planned_noop_successor(
    simple_screw_assembly,
    dummy_settings,
):
    def plan_func(assembly, op):
        op.data["direction"] = np.array([0.0, 0.0, 1.0])
        noop_successor = Assembly("noop_successor")
        noop_successor.parts = dict(assembly.parts)
        noop_successor.connections = list(assembly.connections)
        noop_successor._fcl_objects = {}
        return op, noop_successor

    capability = build_capability_with_planner(plan_func, dummy_settings)
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")
    action = capability.plan_motion(
        simple_screw_assembly, capability.decode_action(simple_screw_assembly, raw)
    )

    out = capability.deterministic_transition(simple_screw_assembly, action)

    assert (
        "result_assembly" not in action.operation.data
    )  # the core judged it no result
    assert out.get_part_by_name("screw_1") is not None
    assert out is not simple_screw_assembly


def test_deterministic_transition_without_successor_keeps_state_unchanged(
    capability,
    simple_screw_assembly,
):
    capability.build_action_space(simple_screw_assembly)
    raw = raw_action_for(capability, "H5", "screw_1", "base_1", "H5")
    action = capability.decode_action(simple_screw_assembly, raw)
    action.operation = DummyScrewingOperation({})

    out = capability.deterministic_transition(simple_screw_assembly, action)

    assert out is not simple_screw_assembly
    assert out.get_part_by_name("screw_1") is not None
