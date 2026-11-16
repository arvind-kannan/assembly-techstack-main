from __future__ import annotations

import numpy as np

from tests.conftest import FakeEnvCapability, FakeObserver
from tests.unit.test_disassembly_pomdp_env import DisassemblyPOMDPEnv


def test_env_pipeline_reset_step_observe_success(simple_env_assembly, env_settings):
    """
    Minimal integration test:
    reset -> observer measurement -> accepted step -> transition -> post-observation.
    """
    observer = FakeObserver({"part_1", "part_2"})
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
        initial_mounted_tool="H5",
    )

    obs0, info0 = env.reset(seed=0)

    assert obs0["accepted"] == 1
    assert info0["measurement_debug_reset"]["stage"] == "reset"

    # After removing part_1, only part_2 remains visible.
    observer.detected_parts = {"part_2"}

    obs1, reward, terminated, truncated, info1 = env.step(np.array([0, 0, 0]))

    assert info1["success"] is True
    assert reward == -2.0
    assert terminated is True
    assert truncated is False
    assert env.assembly.get_part_by_name("part_1") is None
    assert observer.observe_calls == ["reset", "post"]
    assert np.allclose(obs1["parts"], np.array([0.0, 1.0], dtype=np.float32))


def test_env_pipeline_rejected_action_keeps_assembly_state(
    simple_env_assembly, env_settings
):
    """
    Minimal integration test:
    reset -> rejected out-of-range step -> state unchanged -> feedback observation.
    """
    observer = FakeObserver({"part_1", "part_2"})
    cap = FakeEnvCapability(
        tools=["H5"],
        part_names=["part_1"],
        remove_on_success=True,
    )

    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        belief_mode="pomdp",
        observer=observer,
        rejected_penalty=33.0,
        initial_mounted_tool="H5",
    )

    env.reset(seed=0)
    parts_before = set(env.assembly.parts.keys())

    obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 99]))

    assert set(env.assembly.parts.keys()) == parts_before
    assert reward == -33.0
    assert terminated is False
    assert truncated is False
    assert obs["accepted"] == 0
    assert info["illegal_action"] is True
    assert info["illegal_reason"] == "out_of_range"
