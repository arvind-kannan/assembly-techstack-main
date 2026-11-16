from __future__ import annotations

import numpy as np

from tests.conftest import (
    build_gripping_and_removing_capability_with_planner,
    raw_manip_action_for,
    set_dof,
    set_visibility,
    successor_without_part,
)


def test_valid_gripping_and_removing_pipeline_reaches_successor_state(
    simple_manip_assembly,
    gripping_and_removing_settings,
):
    """
    Minimal integration test:
    action space -> decode -> mask -> plan_motion -> deterministic_transition.
    """

    def plan_func(assembly, op):
        op.data["planned"] = True
        op.data["direction"] = np.array([0.0, 0.0, 1.0])
        op.data["path_duration"] = 2.0
        op.data["path"] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
            ]
        )
        return op, successor_without_part(assembly, "part_1")

    set_dof(simple_manip_assembly, [np.array([0.0, 0.0, 1.0])])
    set_visibility(simple_manip_assembly, True)

    cap = build_gripping_and_removing_capability_with_planner(
        plan_func,
        gripping_and_removing_settings,
    )
    cap.build_action_space(simple_manip_assembly)

    raw = raw_manip_action_for(cap, "gripper_a", "part_1")

    masked, reason, details = cap.predecode_mask_reason(
        simple_manip_assembly,
        raw,
        current_tool_name="gripper_a",
    )
    assert not masked
    assert reason is None
    assert details == {}

    action = cap.decode_action(simple_manip_assembly, raw)
    action = cap.plan_motion(simple_manip_assembly, action)
    next_assembly = cap.deterministic_transition(simple_manip_assembly, action)

    assert next_assembly.get_part_by_name("part_1") is None
    assert action.operation is not None
    assert action.operation.data["motion_duration"] > 0.0


def test_noop_gripping_and_removing_successor_is_rejected_by_post_check(
    simple_manip_assembly,
    gripping_and_removing_settings,
):
    """
    Planner returns a successor, but the target part is still present.
    The semantic post-check must reject this no-op transition.
    """

    def plan_func(assembly, op):
        op.data["planned"] = True
        op.data["direction"] = np.array([0.0, 0.0, 1.0])
        op.data["path_duration"] = 2.0
        op.data["path"] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return op, assembly

    set_dof(simple_manip_assembly, [np.array([0.0, 0.0, 1.0])])
    set_visibility(simple_manip_assembly, True)

    cap = build_gripping_and_removing_capability_with_planner(
        plan_func,
        gripping_and_removing_settings,
    )
    cap.build_action_space(simple_manip_assembly)

    raw = raw_manip_action_for(cap, "gripper_a", "part_1")
    action = cap.decode_action(simple_manip_assembly, raw)
    action = cap.plan_motion(simple_manip_assembly, action)

    next_assembly = cap.deterministic_transition(simple_manip_assembly, action)

    assert next_assembly is not simple_manip_assembly
    assert next_assembly.get_part_by_name("part_1") is not None


def test_gripping_and_removing_wrong_tool_is_masked_before_planning(
    simple_manip_assembly,
    gripping_and_removing_settings,
):
    planner_calls = {"n": 0}

    def plan_func(assembly, op):
        planner_calls["n"] += 1
        return op

    cap = build_gripping_and_removing_capability_with_planner(
        plan_func,
        gripping_and_removing_settings,
    )
    cap.build_action_space(simple_manip_assembly)

    raw = raw_manip_action_for(cap, "gripper_a", "part_1")

    masked, reason, details = cap.predecode_mask_reason(
        simple_manip_assembly,
        raw,
        current_tool_name="gripper_b",
    )

    assert masked
    assert reason == "wrong_tool"
    assert details["mounted"] == "gripper_b"
    assert planner_calls["n"] == 0


def test_gripping_and_removing_empty_dof_is_masked_as_geometric_infeasibility(
    simple_manip_assembly,
    gripping_and_removing_settings,
):
    planner_calls = {"n": 0}

    def plan_func(assembly, op):
        planner_calls["n"] += 1
        return op

    set_dof(simple_manip_assembly, [])

    cap = build_gripping_and_removing_capability_with_planner(
        plan_func,
        gripping_and_removing_settings,
    )
    cap.build_action_space(simple_manip_assembly)

    raw = raw_manip_action_for(cap, "gripper_a", "part_1")

    masked, reason, details = cap.predecode_mask_reason(
        simple_manip_assembly,
        raw,
        current_tool_name="gripper_a",
    )

    assert masked
    assert reason == "geometrically_infeasible"
    assert details["why"] == "empty_dof"
    assert planner_calls["n"] == 0
