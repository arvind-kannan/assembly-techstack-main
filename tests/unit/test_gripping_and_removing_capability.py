from __future__ import annotations

import numpy as np
import pytest
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.connection import Screw_Connection

from tests.conftest import (
    DummyGrippingAndRemoving,
    FakePart,
    build_gripping_and_removing_capability_with_planner,
    raw_manip_action_for,
    set_dof,
    set_visibility,
    successor_without_part,
)

# ---------------------------------------------------------------------
# Action-space construction and from_assembly
# ---------------------------------------------------------------------


def test_build_action_space_uses_grippers_and_global_part_names(
    manip_capability,
    simple_manip_assembly,
):
    space = manip_capability.build_action_space(simple_manip_assembly)

    assert list(space.nvec) == [2, 4]
    assert manip_capability.global_part_names == [
        "part_1",
        "part_2",
        "base_1",
        "buffer_milled_1",
    ]


def test_action_space_property_requires_build_before_access(
    gripping_and_removing_settings,
):
    from disassembly_pomdp.gripping_and_removing_capability import (
        GrippingAndRemovingCapability,
    )

    cap = GrippingAndRemovingCapability(
        name="gripping_and_removing",
        available_grippers=["gripper_a"],
        global_part_names=["part_1"],
        cell_settings=gripping_and_removing_settings,
        operation_cls=DummyGrippingAndRemoving,
    )

    with pytest.raises(RuntimeError):
        _ = cap.action_space


def test_from_assembly_excludes_screw_parts_and_adds_buffer_slots(
    gripping_and_removing_settings,
):
    from disassembly_pomdp.gripping_and_removing_capability import (
        GrippingAndRemovingCapability,
    )

    asm = Assembly("assembly_with_screw")
    asm.parts = {
        "screw_1": FakePart("screw_1"),
        "part_1": FakePart("part_1"),
        "part_2": FakePart("part_2"),
    }
    asm.connections = [Screw_Connection("screw_1", "part_1", "H5")]
    asm._fcl_objects = {}

    cap = GrippingAndRemovingCapability.from_assembly(
        assembly=asm,
        available_grippers=["gripper_a"],
        buffer_fraction=0.5,
        cell_settings=gripping_and_removing_settings,
        operation_cls=DummyGrippingAndRemoving,
        base_parts=(),
    )

    assert "screw_1" not in cap.global_part_names
    assert "part_1" in cap.global_part_names
    assert "part_2" in cap.global_part_names
    assert any(name.startswith("buffer_milled_") for name in cap.global_part_names)


# ---------------------------------------------------------------------
# Decoding raw actions
# ---------------------------------------------------------------------


def test_decode_action_returns_semantic_gripping_and_removing_action(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")

    action = manip_capability.decode_action(simple_manip_assembly, raw)

    assert action.tool_name == "gripper_a"
    assert action.part_name == "part_1"
    assert action.gripper_id == 0
    assert action.part_index == manip_capability.global_part_names.index("part_1")
    assert action.operation is None


def test_decode_action_rejects_out_of_range_tool_index(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)

    with pytest.raises(IndexError):
        manip_capability.decode_action(simple_manip_assembly, np.array([99, 0]))


def test_decode_action_rejects_out_of_range_part_index(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)

    with pytest.raises(IndexError):
        manip_capability.decode_action(simple_manip_assembly, np.array([0, 99]))


# ---------------------------------------------------------------------
# Target reference and target presence
# ---------------------------------------------------------------------


def test_get_target_ref_returns_part_when_part_exists(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")

    target = manip_capability.get_target_ref(simple_manip_assembly, raw)

    assert target is not None
    assert target.kind == "part"
    assert target.name == "part_1"


def test_get_target_ref_returns_buffer_slot_for_missing_buffer_name(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "buffer_milled_1")

    target = manip_capability.get_target_ref(simple_manip_assembly, raw)

    assert target is not None
    assert target.kind == "buffer_slot"
    assert target.name == "buffer_milled_1"


def test_get_target_ref_distinguishes_never_existed_part(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_2")

    simple_manip_assembly.parts.pop("part_2", None)

    target = manip_capability.get_target_ref(simple_manip_assembly, raw)

    assert target is not None
    assert target.kind == "part_never_existed"
    assert target.name == "part_2"


def test_get_target_ref_treats_removed_previous_part_as_part(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")

    first_target = manip_capability.get_target_ref(simple_manip_assembly, raw)
    assert first_target.kind == "part"

    simple_manip_assembly.parts.pop("part_1", None)
    second_target = manip_capability.get_target_ref(simple_manip_assembly, raw)

    assert second_target.kind == "part"
    assert second_target.name == "part_1"


def test_is_target_present_accepts_buffer_slot(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "buffer_milled_1")
    target = manip_capability.get_target_ref(simple_manip_assembly, raw)

    assert manip_capability.is_target_present(simple_manip_assembly, target)


def test_is_target_present_checks_real_part_presence(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")
    target = manip_capability.get_target_ref(simple_manip_assembly, raw)

    assert manip_capability.is_target_present(simple_manip_assembly, target)

    simple_manip_assembly.parts.pop("part_1", None)

    assert not manip_capability.is_target_present(simple_manip_assembly, target)


# ---------------------------------------------------------------------
# Pre-decode mask reasons
# ---------------------------------------------------------------------


def test_predecode_mask_base_part(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "base_1")

    masked, reason, details = manip_capability.predecode_mask_reason(
        simple_manip_assembly,
        raw,
        current_tool_name="gripper_a",
    )

    assert masked
    assert reason == "base_part"
    assert details["part"] == "base_1"


def test_predecode_mask_wrong_mounted_tool(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")

    masked, reason, details = manip_capability.predecode_mask_reason(
        simple_manip_assembly,
        raw,
        current_tool_name="gripper_b",
    )

    assert masked
    assert reason == "wrong_tool"
    assert details["mounted"] == "gripper_b"
    assert details["required"] == "gripper_a"


def test_predecode_mask_part_missing(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")

    simple_manip_assembly.parts.pop("part_1", None)

    masked, reason, details = manip_capability.predecode_mask_reason(
        simple_manip_assembly,
        raw,
        current_tool_name="gripper_a",
    )

    assert masked
    assert reason == "geometrically_infeasible"
    assert details["why"] == "part_missing"


def test_predecode_mask_empty_dof(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")

    set_dof(simple_manip_assembly, [])

    masked, reason, details = manip_capability.predecode_mask_reason(
        simple_manip_assembly,
        raw,
        current_tool_name="gripper_a",
    )

    assert masked
    assert reason == "geometrically_infeasible"
    assert details["why"] == "empty_dof"


def test_predecode_mask_not_visible(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")

    set_dof(simple_manip_assembly, [np.array([0.0, 0.0, 1.0])])
    set_visibility(simple_manip_assembly, False)

    masked, reason, details = manip_capability.predecode_mask_reason(
        simple_manip_assembly,
        raw,
        current_tool_name="gripper_a",
    )

    assert masked
    assert reason == "geometrically_infeasible"
    assert details["why"] == "not_visible"


def test_predecode_mask_accepts_valid_action(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")

    set_dof(simple_manip_assembly, [np.array([0.0, 0.0, 1.0])])
    set_visibility(simple_manip_assembly, True)

    masked, reason, details = manip_capability.predecode_mask_reason(
        simple_manip_assembly,
        raw,
        current_tool_name="gripper_a",
    )

    assert not masked
    assert reason is None
    assert details == {}


# ---------------------------------------------------------------------
# Operation construction
# ---------------------------------------------------------------------


def test_make_operation_requires_operation_class(
    simple_manip_assembly,
    gripping_and_removing_settings,
):
    from disassembly_pomdp.gripping_and_removing_capability import (
        GrippingAndRemovingCapability,
    )

    cap = GrippingAndRemovingCapability(
        name="gripping_and_removing",
        available_grippers=["gripper_a"],
        global_part_names=["part_1"],
        cell_settings=gripping_and_removing_settings,
        operation_cls=None,
    )
    cap.build_action_space(simple_manip_assembly)
    action = cap.decode_action(simple_manip_assembly, np.array([0, 0]))

    with pytest.raises(RuntimeError):
        cap.make_operation(simple_manip_assembly, action)


def test_make_operation_creates_gripping_and_removing_operation_shell(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")
    action = manip_capability.decode_action(simple_manip_assembly, raw)

    op = manip_capability.make_operation(simple_manip_assembly, action)

    assert isinstance(op, DummyGrippingAndRemoving)
    assert op.data["type"] == "gripping"
    assert op.data["tool_name"] == "gripper_a"
    assert op.data["part_name"] == "part_1"
    assert op.data["part_index"] == manip_capability.global_part_names.index("part_1")
    assert op.data["cost"] == 1.5
    assert op.data["planned"] is False


# ---------------------------------------------------------------------
# Planning wrapper, motion program, and time estimation
# ---------------------------------------------------------------------


def test_plan_motion_stores_planner_successor_and_motion_program(
    simple_manip_assembly,
    gripping_and_removing_settings,
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
        return op, successor_without_part(assembly, "part_1")

    cap = build_gripping_and_removing_capability_with_planner(
        plan_func,
        gripping_and_removing_settings,
    )
    cap.build_action_space(simple_manip_assembly)

    raw = raw_manip_action_for(cap, "gripper_a", "part_1")
    action = cap.decode_action(simple_manip_assembly, raw)

    planned_action = cap.plan_motion(simple_manip_assembly, action)

    op = planned_action.operation
    assert op is not None
    assert op.data["planned"] is True
    assert "result_assembly" in op.data
    assert np.allclose(op.data["direction"], np.array([0.0, 0.0, 1.0]))
    assert "motion_program" in op.data
    assert op.data["motion_program"][0]["type"] == "grip_core"
    assert op.data["motion_program"][-1]["type"] == "dropoff"
    assert op.data["motion_duration"] == pytest.approx(1.5 + 4.0 + 3.0)


def test_plan_motion_falls_back_to_dof_direction(
    simple_manip_assembly,
    gripping_and_removing_settings,
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

    set_dof(simple_manip_assembly, [np.array([0.0, 0.0, 3.0])])

    cap = build_gripping_and_removing_capability_with_planner(
        plan_func,
        gripping_and_removing_settings,
    )
    cap.build_action_space(simple_manip_assembly)

    raw = raw_manip_action_for(cap, "gripper_a", "part_1")
    action = cap.decode_action(simple_manip_assembly, raw)

    planned_action = cap.plan_motion(simple_manip_assembly, action)

    assert np.allclose(planned_action.operation.data["direction"], [0.0, 0.0, 1.0])


def test_estimate_action_time_uses_motion_duration_when_available(
    manip_capability,
    simple_manip_assembly,
    gripping_and_removing_settings,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")
    action = manip_capability.decode_action(simple_manip_assembly, raw)
    action.operation = DummyGrippingAndRemoving({"motion_duration": 12.5})

    assert (
        manip_capability.estimate_action_time(
            simple_manip_assembly,
            action,
            gripping_and_removing_settings,
        )
        == 12.5
    )


def test_estimate_action_time_uses_symbolic_fallback_without_motion_program(
    manip_capability,
    simple_manip_assembly,
    gripping_and_removing_settings,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")
    action = manip_capability.decode_action(simple_manip_assembly, raw)

    t = manip_capability.estimate_action_time(
        simple_manip_assembly,
        action,
        gripping_and_removing_settings,
    )

    assert t > 0.0


# ---------------------------------------------------------------------
# Deterministic transition and semantic post-check
# ---------------------------------------------------------------------


def test_deterministic_transition_accepts_successor_only_if_part_removed(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")
    action = manip_capability.decode_action(simple_manip_assembly, raw)

    result_assembly = successor_without_part(simple_manip_assembly, "part_1")
    action.operation = DummyGrippingAndRemoving(
        {
            "result_assembly": result_assembly,
            "path": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        }
    )

    out = manip_capability.deterministic_transition(simple_manip_assembly, action)

    assert out is result_assembly
    assert out.get_part_by_name("part_1") is None


def test_deterministic_transition_rejects_a_planned_noop_successor(
    simple_manip_assembly,
    gripping_and_removing_settings,
):
    def plan_func(assembly, op):
        op.data["direction"] = np.array([0.0, 0.0, 1.0])
        noop_successor = Assembly("noop_successor")
        noop_successor.parts = dict(assembly.parts)
        noop_successor.connections = list(assembly.connections)
        noop_successor._fcl_objects = {}
        return op, noop_successor

    cap = build_gripping_and_removing_capability_with_planner(
        plan_func, gripping_and_removing_settings
    )
    cap.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(cap, "gripper_a", "part_1")
    action = cap.plan_motion(
        simple_manip_assembly, cap.decode_action(simple_manip_assembly, raw)
    )

    out = cap.deterministic_transition(simple_manip_assembly, action)

    assert (
        "result_assembly" not in action.operation.data
    )  # the core judged it no result
    assert out.get_part_by_name("part_1") is not None
    assert out is not simple_manip_assembly


def test_deterministic_transition_without_successor_keeps_state_unchanged(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_1")
    action = manip_capability.decode_action(simple_manip_assembly, raw)
    action.operation = DummyGrippingAndRemoving({})

    out = manip_capability.deterministic_transition(simple_manip_assembly, action)

    assert out is not simple_manip_assembly
    assert out.get_part_by_name("part_1") is not None


# ---------------------------------------------------------------------
# Cell state: holds and mounted tools
# ---------------------------------------------------------------------


def test_cell_state_transition_releases_the_arm_after_a_successful_removal(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    action = manip_capability.decode_action(
        simple_manip_assembly,
        raw_manip_action_for(manip_capability, "gripper_a", "part_1"),
    )
    cell_state = CellState(holds={"robot_1": "part_1", "robot_2": "part_2"})
    removed = successor_without_part(simple_manip_assembly, "part_1")

    after = manip_capability.cell_state_transition(cell_state, action, removed)

    assert after.holds == {"robot_2": "part_2"}
    assert cell_state.holds == {"robot_1": "part_1", "robot_2": "part_2"}


def test_cell_state_transition_keeps_the_hold_when_the_part_was_not_removed(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    action = manip_capability.decode_action(
        simple_manip_assembly,
        raw_manip_action_for(manip_capability, "gripper_a", "part_1"),
    )
    cell_state = CellState(holds={"robot_1": "part_1"})

    after = manip_capability.cell_state_transition(
        cell_state, action, simple_manip_assembly
    )

    assert after == cell_state


def test_mask_blocks_a_robot_that_holds_a_part_per_the_cell_state(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_2")

    masked, reason, details = manip_capability.predecode_mask_reason(
        simple_manip_assembly, raw, cell_state=CellState(holds={"robot_1": "part_1"})
    )

    assert (masked, reason, details["agent"]) == (True, "agent_busy", "robot_1")


def test_mask_blocks_a_part_another_robot_holds_per_the_cell_state(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_2")

    masked, reason, details = manip_capability.predecode_mask_reason(
        simple_manip_assembly, raw, cell_state=CellState(holds={"robot_2": "part_2"})
    )

    assert (masked, reason, details["why"]) == (
        True,
        "agent_busy",
        "part_held_by_other",
    )


def test_mask_reads_the_mounted_tool_from_the_cell_state(
    manip_capability,
    simple_manip_assembly,
):
    manip_capability.build_action_space(simple_manip_assembly)
    raw = raw_manip_action_for(manip_capability, "gripper_a", "part_2")

    masked, reason, details = manip_capability.predecode_mask_reason(
        simple_manip_assembly,
        raw,
        current_tool_name="gripper_a",
        cell_state=CellState(mounted_tools={"robot_1": "gripper_b"}),
    )

    assert (masked, reason) == (True, "wrong_tool")
    assert details == {"mounted": "gripper_b", "required": "gripper_a"}
