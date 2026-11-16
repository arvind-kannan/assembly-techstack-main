from __future__ import annotations

import copy
import importlib

import numpy as np
import pytest

from disassembly_pomdp.eol_operator import PartEOLOperatorV2
from tests.conftest import (
    FakeEnvCapability,
    FakeObserver,
    FakePart,
)

# ---------------------------------------------------------------------
# Robust import helper
# ---------------------------------------------------------------------

MODULE_CANDIDATES = [
    "disassembly_pomdp.disassembly_pomdp_gym",
    "disassembly_pomdp.env",
    "disassembly_pomdp.environment",
]


def _load_env_module():
    last_error = None
    for module_name in MODULE_CANDIDATES:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            last_error = exc

    raise ModuleNotFoundError(
        "Could not import DisassemblyPOMDPEnv. "
        "Please add the actual module path to MODULE_CANDIDATES."
    ) from last_error


_env_module = _load_env_module()
DisassemblyPOMDPEnv = _env_module.DisassemblyPOMDPEnv
_remove_part_and_connections_inplace = _env_module._remove_part_and_connections_inplace


# ---------------------------------------------------------------------
# Construction / reset
# ---------------------------------------------------------------------


def test_env_requires_observer_in_pomdp_mode(simple_env_assembly, env_settings):
    cap = FakeEnvCapability(part_names=["part_1"])

    with pytest.raises(ValueError):
        DisassemblyPOMDPEnv(
            assembly=simple_env_assembly,
            capabilities=[cap],
            cell_settings=env_settings,
            target_part_names=["part_1"],
            belief_mode="pomdp",
            observer=None,
        )


def test_env_builds_global_padded_action_space(simple_env_assembly, env_settings):
    cap_a = FakeEnvCapability(
        name="cap_a",
        tools=["H5"],
        part_names=["part_1", "part_2"],
    )
    cap_b = FakeEnvCapability(
        name="cap_b",
        tools=["H5", "tool_b", "tool_c"],
        part_names=["part_1"],
    )

    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap_a, cap_b],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1", "part_2"}),
        initial_mounted_tool="H5",
    )

    assert list(env.action_space.nvec) == [2, 3, 2]


def test_reset_returns_observer_based_initial_observation(
    simple_env_assembly, env_settings
):
    observer = FakeObserver({"part_1"})
    cap = FakeEnvCapability(part_names=["part_1", "part_2"])

    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=observer,
        initial_mounted_tool="H5",
    )

    obs, info = env.reset(seed=0)

    assert observer.reset_calls == 1
    assert observer.observe_calls == ["reset"]
    assert obs["accepted"] == 1
    assert np.allclose(obs["parts"], np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(obs["tool"], np.array([1.0], dtype=np.float32))
    assert info["measurement_debug_reset"]["stage"] == "reset"


def test_reset_in_fo_mode_uses_truth_parts_without_observer(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(part_names=["part_1", "part_2"])

    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="fo",
        observer=None,
        initial_mounted_tool="H5",
    )

    obs, _info = env.reset(seed=0)

    assert np.allclose(obs["parts"], np.array([1.0, 1.0], dtype=np.float32))


def test_rejected_first_action_in_fully_observable_mode_observes_the_true_parts(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(part_names=["part_1", "part_2"])
    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="fo",
        observer=None,
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    obs, _reward, _terminated, _truncated, info = env.step(np.array([9, 0, 0]))

    assert info["illegal_action"] is True
    assert obs["accepted"] == 0
    assert np.allclose(obs["parts"], np.array([1.0, 1.0], dtype=np.float32))


# ---------------------------------------------------------------------
# Pre-decode reject paths
# ---------------------------------------------------------------------


def test_step_rejects_capability_index_out_of_range(simple_env_assembly, env_settings):
    cap = FakeEnvCapability(part_names=["part_1"])
    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1"}),
        rejected_penalty=123.0,
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(np.array([9, 0, 0]))

    assert reward == pytest.approx(-123.0)
    assert not terminated
    assert not truncated
    assert obs["accepted"] == 0
    assert info["illegal_action"] is True
    assert info["illegal_reason"] == "out_of_range"
    assert info["why"] == "cap_idx_out_of_range"


def test_step_rejects_local_padding_out_of_range(simple_env_assembly, env_settings):
    cap = FakeEnvCapability(tools=["H5"], part_names=["part_1"])
    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1"}),
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 99]))

    assert obs["accepted"] == 0
    assert info["illegal_action"] is True
    assert info["illegal_reason"] == "out_of_range"


def test_step_rejects_missing_target_as_masked_out(simple_env_assembly, env_settings):
    cap = FakeEnvCapability(part_names=["part_1"])
    simple_env_assembly.parts.pop("part_1", None)

    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver(set()),
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert obs["accepted"] == 0
    assert info["illegal_action"] is True
    assert info["illegal_reason"] == "masked_out"
    assert info["mask_reason"] == "target_missing"


def test_step_rejects_capability_specific_mask_reason(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(
        part_names=["part_1"],
        mask_reason="geometrically_infeasible",
        mask_details={"why": "empty_dof"},
    )
    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1"}),
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert obs["accepted"] == 0
    assert info["illegal_action"] is True
    assert info["illegal_reason"] == "masked_out"
    assert info["mask_reason"] == "geometrically_infeasible"
    assert info["why"] == "empty_dof"


def test_step_rejected_before_exec_gate_records_gate_not_evaluated(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(part_names=["part_1"], mask_reason="wrong_tool")
    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1"}),
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)
    # Record left by an earlier step whose exec gate failed.
    cap.set_last_exec_gate(ok=False, stage="exec_gate", reason="bad_direction")

    _obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["illegal_action"] is True
    assert info["exec_gate_ok"] is None
    assert cap.get_last_exec_gate()["ok"] is None


# ---------------------------------------------------------------------
# Accepted action / transition behavior
# ---------------------------------------------------------------------


def test_successful_step_removes_target_and_terminates(
    simple_env_assembly, env_settings
):
    observer = FakeObserver({"part_2"})
    cap = FakeEnvCapability(
        tools=["H5"],
        part_names=["part_1"],
        remove_on_success=True,
        p_exec=1.0,
        op_duration=2.0,
    )

    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=observer,
        noop_penalty=100.0,
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))

    assert info["success"] is True
    assert info["illegal_action"] is False
    assert terminated is True
    assert truncated is False
    assert env.assembly.get_part_by_name("part_1") is None
    assert reward == pytest.approx(-2.0)
    assert obs["accepted"] == 1


def test_step_on_a_shallow_copied_assembly_deep_copies_without_error(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(
        tools=["H5"],
        part_names=["part_1"],
        remove_on_success=True,
        p_exec=1.0,
        op_duration=2.0,
    )
    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_2"}),
        noop_penalty=100.0,
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)
    env.assembly = copy.copy(env.assembly)

    _, _, _, _, info = env.step(np.array([0, 0, 0]))

    assert info["success"] is True


def test_noop_successor_is_classified_as_semantic_failure(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(
        tools=["H5"],
        part_names=["part_1"],
        remove_on_success=False,
        p_exec=1.0,
        op_duration=2.0,
    )

    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1", "part_2"}),
        noop_penalty=50.0,
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    _obs, reward, terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["success"] is False
    assert info["noop"] is True
    assert info["failure_stage"] == "semantic"
    assert info["noop_reason"] == "sampled_but_no_effect"
    assert terminated is False
    assert reward == pytest.approx(-(2.0 + 50.0))


def test_exec_gate_bad_direction_fails_before_attempt(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(
        tools=["H5"],
        part_names=["part_1"],
        operation_direction=np.array([0.0, 0.0, 0.0]),
        p_exec=1.0,
        op_duration=2.0,
    )

    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1", "part_2"}),
        noop_penalty=50.0,
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    _obs, reward, terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["success"] is False
    assert info["attempted"] is False
    assert info["exec_gate_ok"] is False
    assert info["failure_stage"] == "exec_gate"
    assert info["exec_fail_reason"] in {"bad_direction", "missing_direction"}
    assert terminated is False
    assert reward == pytest.approx(-(2.0 + 50.0))


def test_exec_gate_feasibility_failure_is_reported(simple_env_assembly, env_settings):
    cap = FakeEnvCapability(
        tools=["H5"],
        part_names=["part_1"],
        operation_feasible=False,
        p_exec=1.0,
        op_duration=2.0,
    )

    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1", "part_2"}),
        noop_penalty=50.0,
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    _obs, reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["success"] is False
    assert info["attempted"] is False
    assert info["exec_gate_ok"] is False
    assert info["failure_stage"] == "exec_gate"
    assert info["exec_fail_reason"] == "check_feasability_failed"
    assert reward == pytest.approx(-(2.0 + 50.0))


def test_step_truncates_at_max_steps_when_not_terminated(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(
        tools=["H5"],
        part_names=["part_1"],
        remove_on_success=False,
        p_exec=1.0,
        op_duration=2.0,
    )

    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1", "part_2"}),
        max_steps=1,
        noop_penalty=10.0,
        truncation_penalty=99.0,
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    _obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))

    assert terminated is False
    assert truncated is True
    assert info["truncation_penalty"] == pytest.approx(99.0)
    assert reward == pytest.approx(-(2.0 + 10.0 + 99.0))


# ---------------------------------------------------------------------
# Helper behavior
# ---------------------------------------------------------------------


def test_remove_part_and_connections_inplace_removes_referencing_connections(
    simple_env_assembly,
):
    conn = type("Conn", (), {})()
    conn.part_name1 = "part_1"
    conn.part_name2 = "part_2"
    conn.connection_type = "fixed"
    simple_env_assembly.connections = [conn]

    changed = _remove_part_and_connections_inplace(simple_env_assembly, "part_1")

    assert changed is True
    assert simple_env_assembly.get_part_by_name("part_1") is None
    assert simple_env_assembly.connections == []


# ---------------------------------------------------------------------
# Cell state
# ---------------------------------------------------------------------


def _env_with(simple_env_assembly, env_settings, cap):
    return DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_2"}),
        initial_mounted_tool="H5",
    )


def test_reset_mounts_the_initial_tool_in_the_cell_state(
    simple_env_assembly, env_settings
):
    env = _env_with(
        simple_env_assembly,
        env_settings,
        FakeEnvCapability(tools=["H5"], part_names=["part_1"]),
    )

    env.reset(seed=0)

    assert env.cell_state.mounted_tools == {"robot_1": "H5"}
    assert env.cell_state.holds == {}


def test_step_takes_the_cell_state_after_the_transition(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(tools=["H5"], part_names=["part_1"], holds_on_success=True)
    env = _env_with(simple_env_assembly, env_settings, cap)
    env.reset(seed=0)

    env.step(np.array([0, 0, 0]))

    assert env.cell_state.holds == {"robot_1": "part_1"}

    env.reset(seed=0)

    assert env.cell_state.holds == {}


def test_gripping_action_that_keeps_the_part_succeeds_and_is_kept(
    simple_env_assembly, env_settings
):
    """A gripping action keeps the part and changes only the cell state; it is not reverted (#61)."""
    cap = FakeEnvCapability(
        tools=["H5"],
        part_names=["part_1"],
        remove_on_success=False,
        holds_on_success=True,
    )
    env = _env_with(simple_env_assembly, env_settings, cap)
    env.reset(seed=0)

    _obs, _reward, terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["success"] is True
    assert info["noop"] is False
    assert env.cell_state.holds == {"robot_1": "part_1"}
    assert env.assembly.get_part_by_name("part_1") is not None
    assert terminated is False


class _AddsPartCapability(FakeEnvCapability):
    """Keeps its target part and adds a new one, like an assembly capability would."""

    def deterministic_transition(self, assembly, action):
        out = super().deterministic_transition(assembly, action)
        out.parts["part_3"] = FakePart("part_3")
        return out


def test_action_that_keeps_the_part_and_changes_the_assembly_succeeds(
    simple_env_assembly, env_settings
):
    cap = _AddsPartCapability(
        tools=["H5"], part_names=["part_1"], remove_on_success=False
    )
    env = _env_with(simple_env_assembly, env_settings, cap)
    env.reset(seed=0)

    _obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["success"] is True
    assert env.assembly.get_part_by_name("part_1") is not None
    assert env.assembly.get_part_by_name("part_3") is not None


# ---------------------------------------------------------------------
# EOL belief update after an attempt (#62)
# ---------------------------------------------------------------------


def _env_with_eol(simple_env_assembly, env_settings, *, freeze_updates, remove):
    cap = FakeEnvCapability(
        tools=["H5"],
        part_names=["part_1"],
        remove_on_success=remove,
        p_exec=1.0,
        op_duration=2.0,
    )
    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1", "part_2"}),
        eol_op=PartEOLOperatorV2(
            p_free_success=0.9, default_p_stuck=0.2, freeze_updates=freeze_updates
        ),
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)
    return env


def _force_draw(env, monkeypatch, value):
    """Make the success draw of the next step come out as ``value``."""
    monkeypatch.setattr(
        env, "rng", type("Draw", (), {"random": lambda self: value})(), raising=False
    )


def test_failed_attempt_raises_the_stuck_belief_and_reports_the_event(
    simple_env_assembly, env_settings, monkeypatch
):
    env = _env_with_eol(
        simple_env_assembly, env_settings, freeze_updates=False, remove=True
    )
    prior = env._eol_op.get_p_stuck("part_1")
    _force_draw(env, monkeypatch, 0.999)

    _obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["success"] is False
    assert info["failure_stage"] == "stochastic"
    event = info["eol_event"]
    assert event.part == "part_1"
    assert event.success is False
    assert event.p_stuck_prior == pytest.approx(prior)
    assert event.p_stuck_post > prior
    assert env._eol_op.get_p_stuck("part_1") == pytest.approx(event.p_stuck_post)


def test_successful_attempt_lowers_the_stuck_belief(
    simple_env_assembly, env_settings, monkeypatch
):
    env = _env_with_eol(
        simple_env_assembly, env_settings, freeze_updates=False, remove=True
    )
    prior = env._eol_op.get_p_stuck("part_1")
    _force_draw(env, monkeypatch, 0.0)

    _obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["success"] is True
    assert info["eol_event"].success is True
    assert env._eol_op.get_p_stuck("part_1") < prior


def test_frozen_belief_keeps_the_stuck_belief_but_still_reports_the_event(
    simple_env_assembly, env_settings, monkeypatch
):
    env = _env_with_eol(
        simple_env_assembly, env_settings, freeze_updates=True, remove=True
    )
    prior = env._eol_op.get_p_stuck("part_1")
    _force_draw(env, monkeypatch, 0.999)

    _obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["eol_event"].p_stuck_post == pytest.approx(prior)
    assert env._eol_op.get_p_stuck("part_1") == pytest.approx(prior)


def test_attempt_that_changed_nothing_counts_as_a_failed_attempt(
    simple_env_assembly, env_settings, monkeypatch
):
    env = _env_with_eol(
        simple_env_assembly, env_settings, freeze_updates=False, remove=False
    )
    prior = env._eol_op.get_p_stuck("part_1")
    _force_draw(env, monkeypatch, 0.0)

    _obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["failure_stage"] == "semantic"
    assert info["eol_event"].success is False
    assert env._eol_op.get_p_stuck("part_1") > prior


def test_attempt_stopped_by_the_exec_gate_leaves_the_belief_alone(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(
        tools=["H5"], part_names=["part_1"], operation_feasible=False, p_exec=1.0
    )
    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=FakeObserver({"part_1", "part_2"}),
        eol_op=PartEOLOperatorV2(freeze_updates=False),
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)
    prior = env._eol_op.get_p_stuck("part_1")

    _obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert info["attempted"] is False
    assert "eol_event" not in info
    assert env._eol_op.get_p_stuck("part_1") == pytest.approx(prior)
