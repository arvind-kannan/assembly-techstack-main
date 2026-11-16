from __future__ import annotations

import numpy as np

from tests.conftest import (
    build_capability_with_planner,
    raw_action_for,
    set_dof,
    set_visibility,
    successor_without_screw,
)


def test_valid_screwing_pipeline_reaches_successor_state(
    simple_screw_assembly,
    dummy_settings,
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
        return op, successor_without_screw(assembly, "screw_1")

    set_dof(simple_screw_assembly, [np.array([0.0, 0.0, 1.0])])
    set_visibility(simple_screw_assembly, True)

    cap = build_capability_with_planner(plan_func, dummy_settings)
    cap.build_action_space(simple_screw_assembly)

    raw = raw_action_for(cap, "H5", "screw_1", "base_1", "H5")

    masked, reason, details = cap.predecode_mask_reason(
        simple_screw_assembly,
        raw,
        current_tool_name="H5",
    )
    assert not masked
    assert reason is None
    assert details == {}

    action = cap.decode_action(simple_screw_assembly, raw)
    action = cap.plan_motion(
        simple_screw_assembly, action, rng=np.random.default_rng(1)
    )
    next_assembly = cap.deterministic_transition(simple_screw_assembly, action)

    assert next_assembly.get_part_by_name("screw_1") is None
    assert action.operation is not None
    assert action.operation.data["motion_duration"] > 0.0


def test_noop_planner_successor_is_rejected_by_post_check(
    simple_screw_assembly,
    dummy_settings,
):
    """
    Planner returns a successor, but the screw is still present.
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

    set_dof(simple_screw_assembly, [np.array([0.0, 0.0, 1.0])])
    set_visibility(simple_screw_assembly, True)

    cap = build_capability_with_planner(plan_func, dummy_settings)
    cap.build_action_space(simple_screw_assembly)

    raw = raw_action_for(cap, "H5", "screw_1", "base_1", "H5")
    action = cap.decode_action(simple_screw_assembly, raw)
    action = cap.plan_motion(simple_screw_assembly, action)

    next_assembly = cap.deterministic_transition(simple_screw_assembly, action)

    assert next_assembly is not simple_screw_assembly
    assert next_assembly.get_part_by_name("screw_1") is not None


def test_wrong_tool_is_masked_before_planning(
    simple_screw_assembly,
    dummy_settings,
):
    planner_calls = {"n": 0}

    def plan_func(assembly, op):
        planner_calls["n"] += 1
        return op

    cap = build_capability_with_planner(plan_func, dummy_settings)
    cap.build_action_space(simple_screw_assembly)

    raw = raw_action_for(cap, "H5", "screw_1", "base_1", "H5")

    masked, reason, details = cap.predecode_mask_reason(
        simple_screw_assembly,
        raw,
        current_tool_name="T6",
    )

    assert masked
    assert reason == "wrong_tool"
    assert details["mounted"] == "T6"
    assert planner_calls["n"] == 0


def test_empty_dof_is_masked_as_geometric_infeasibility(
    simple_screw_assembly,
    dummy_settings,
):
    planner_calls = {"n": 0}

    def plan_func(assembly, op):
        planner_calls["n"] += 1
        return op

    set_dof(simple_screw_assembly, [])

    cap = build_capability_with_planner(plan_func, dummy_settings)
    cap.build_action_space(simple_screw_assembly)

    raw = raw_action_for(cap, "H5", "screw_1", "base_1", "H5")

    masked, reason, details = cap.predecode_mask_reason(
        simple_screw_assembly,
        raw,
        current_tool_name="H5",
    )

    assert masked
    assert reason == "geometrically_infeasible"
    assert details["why"] == "empty_dof"
    assert planner_calls["n"] == 0
