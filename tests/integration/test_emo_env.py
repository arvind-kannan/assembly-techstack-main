"""The disassembly POMDP env on EMO with its real plan functions (#49: was the sanity checks in examples/build_env.py)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from disassembly_pomdp.disassembly_pomdp_gym import DisassemblyPOMDPEnv
from disassembly_pomdp.env_builder import build_env, load_product, load_stuck_part_rules
from disassembly_pomdp.eol_operator import EOLUnifiedSampler, PartEOLOperatorV2
from disassembly_pomdp.gripping_and_removing_capability import (
    GrippingAndRemovingCapability,
)
from disassembly_pomdp.screwing_capability import ScrewingCapability

EMO = Path(__file__).resolve().parents[2] / "emo_sr_v2"
SCREWS = ["04_screw_1", "05_screw_2", "06_screw_3", "07_screw_4"]
UPPER_LID = "01_upper_lid"
ROTOR = "03_rotor_short"


def emo_env(
    *,
    p_missing_screw=0.0,
    p_stuck_screw=0.0,
    p_stuck_upper=0.0,
    initial_mounted_tool=None,
) -> DisassemblyPOMDPEnv:
    """EMO on the default cell like `build_env`, but with the given EOL conditions."""
    stock = build_env(EMO)
    assembly, _base, targets = load_product(EMO)
    return DisassemblyPOMDPEnv(
        assembly=assembly,
        capabilities=stock.capabilities,
        cell_settings=stock.cell_settings,
        target_part_names=targets,
        world_sampler=EOLUnifiedSampler(
            p_missing_screw=p_missing_screw,
            p_stuck_screw=p_stuck_screw,
            p_stuck_upper=p_stuck_upper,
            rules=load_stuck_part_rules(EMO),
        ),
        eol_op=PartEOLOperatorV2(freeze_updates=True),
        belief_mode="fo",
        initial_mounted_tool=initial_mounted_tool or stock.initial_mounted_tool,
    )


def _cap_index(env, cap_type) -> int:
    return next(
        i for i, cap in enumerate(env.capabilities) if isinstance(cap, cap_type)
    )


def unscrew_action(env, screw: str) -> np.ndarray:
    """The screwing action on `screw` with the tool its head needs."""
    cap_idx = _cap_index(env, ScrewingCapability)
    n_tools, n_targets = env._cap_action_spaces[cap_idx].nvec
    for tool in range(int(n_tools)):
        for target in range(int(n_targets)):
            action = env.capabilities[cap_idx].decode_action(
                env.assembly, np.array([tool, target])
            )
            if (
                action.screw_part_name == screw
                and env.capabilities[cap_idx].available_screwdrivers[tool]
                == action.tool_name
            ):
                return np.array([cap_idx, tool, target])
    raise LookupError(screw)


def gripping_and_removing_action(env, part: str) -> np.ndarray:
    cap_idx = _cap_index(env, GrippingAndRemovingCapability)
    return np.array([cap_idx, 0, env.capabilities[cap_idx].name_to_part_index[part]])


def parts(env) -> set[str]:
    return set(env.assembly.parts)


def test_missing_screws_are_gone_at_reset_with_their_connections():
    env = emo_env(p_missing_screw=1.0)

    _obs, info = env.reset(seed=0)

    assert info["eol_missing_sampled_parts"] == SCREWS
    assert parts(env).isdisjoint(SCREWS)
    assert not [
        c
        for c in env.assembly.connections
        if {c.part_name1, c.part_name2} & set(SCREWS)
    ]


def test_action_on_a_missing_screw_is_masked_as_target_missing():
    env = emo_env(p_missing_screw=1.0)
    env.reset(seed=0)
    action = unscrew_action(env, SCREWS[0])

    _obs, _reward, _term, _trunc, info = env.step(action)

    assert info["illegal_action"] is True
    assert info["illegal_reason"] == "masked_out"
    assert info["mask_reason"] == "target_missing"


def test_stuck_screws_are_fixed_and_cannot_be_unscrewed():
    env = emo_env(p_stuck_screw=1.0)
    _obs, info = env.reset(seed=0)

    _obs, _reward, _term, _trunc, step_info = env.step(unscrew_action(env, SCREWS[0]))

    assert info["eol_stuck_sampled_parts"] == SCREWS
    assert info["eol_fixed_count"] == 8  # each screw fixed to both lids
    assert step_info["success"] is False
    assert SCREWS[0] in parts(env)


def test_stuck_upper_lid_is_fixed_to_the_stator():
    env = emo_env(p_stuck_upper=1.0)

    _obs, info = env.reset(seed=0)

    assert info["eol_stuck_sampled_parts"] == [UPPER_LID]
    assert [
        {c.part_name1, c.part_name2}
        for c in env.assembly.connections
        if c.connection_type == "fixed"
    ] == [{UPPER_LID, "02_stator"}]


def test_action_needing_another_tool_changes_the_tool_first():
    env = emo_env(initial_mounted_tool="H6")
    env.reset(seed=0)
    action = unscrew_action(env, SCREWS[0])  # EMO screws have H5 heads

    _obs, _reward, _term, _trunc, info = env.step(action)

    assert info["illegal_action"] is False
    assert info["mask_was_toolchange_override"] is True
    assert info["predecode_toolchange_hint"]["required"] == "H5"
    assert info["semantic_action"].tool_name == "H5"
    assert (info["tool_before"], info["tool_after"]) == ("H6", "H5")
    assert env.current_mounted_tool == "H5"
    assert info["toolchange_time"] == pytest.approx(
        80.0
    )  # default cell: 10 to storage + 60 change + 10 back
    assert info["success"] is True


def test_time_estimate_of_each_scripted_action_is_finite_and_positive():
    env = build_env(EMO)
    env.reset(seed=0)

    for action in [unscrew_action(env, s) for s in SCREWS] + [
        gripping_and_removing_action(env, UPPER_LID),
        gripping_and_removing_action(env, ROTOR),
    ]:
        cap = env.capabilities[int(action[0])]
        semantic = cap.plan_motion(
            env.assembly, cap.decode_action(env.assembly, action[1:]), rng=env.np_random
        )
        estimate = cap.estimate_action_time(
            env.assembly, semantic, env.cell_settings, env.current_mounted_tool
        )
        assert np.isfinite(estimate) and estimate > 0.0
        motion_duration = semantic.operation.data.get("motion_duration")
        if motion_duration is not None:
            assert motion_duration <= estimate + 1e-6


@pytest.mark.slow
def test_scripted_policy_removes_screws_upper_lid_and_rotor():
    env = build_env(EMO)
    env.reset(seed=0)
    terminated = False

    for part in SCREWS + [UPPER_LID, ROTOR]:
        action = (
            unscrew_action(env, part)
            if part in SCREWS
            else gripping_and_removing_action(env, part)
        )
        _obs, _reward, terminated, _trunc, info = env.step(action)
        assert info["success"] is True, (
            part,
            info.get("illegal_reason"),
            info.get("failure_stage"),
        )
        assert part not in parts(env)

    assert terminated


@pytest.mark.slow
def test_random_walk_stays_in_the_observation_space_and_success_removes_the_target():
    env = build_env(EMO)
    obs, _info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    rng = np.random.default_rng(0)
    successes = 0

    for _ in range(20):
        cap_idx = int(rng.integers(env.action_space.nvec[0]))
        sub = [int(rng.integers(n)) for n in env._cap_action_spaces[cap_idx].nvec]
        obs, _reward, terminated, truncated, info = env.step(np.array([cap_idx, *sub]))
        assert env.observation_space.contains(obs)
        if info.get("success"):
            successes += 1
            assert info["target_part"] not in parts(env)
        if terminated or truncated:
            break

    assert successes > 0  # else the success check above never ran


# --- toolchange time of the env (#160) ---------------------------------------
# Distinct values per leg, so a start pose that picks the wrong leg fails the sums.
DISTINCT_LEGS = dict(
    toolchange_time_assembly_to_storage=7.0,
    toolchange_time_dropoff_to_storage=11.0,
    toolchange_time_at_toolstorage=60.0,
    toolchange_time_storage_to_assembly=13.0,
)


@pytest.mark.parametrize(
    ("start_pose", "start_leg", "first_step"),
    [
        (
            "assembly",
            "toolchange_time_assembly_to_storage",
            "transit_assembly_to_storage",
        ),
        ("dropoff", "toolchange_time_dropoff_to_storage", "transit_dropoff_to_storage"),
    ],
)
def test_wrong_tool_toolchange_is_one_start_leg_plus_change_plus_return(
    start_pose, start_leg, first_step
):
    env = build_env(EMO)
    env.cell_settings = replace(env.cell_settings, **DISTINCT_LEGS)
    settings = env.cell_settings

    time, program = env._toolchange_prefix(
        start_pose=start_pose, tool_before="H6", tool_after="H5"
    )

    assert time == pytest.approx(
        getattr(settings, start_leg)
        + settings.toolchange_time_at_toolstorage
        + settings.toolchange_time_storage_to_assembly
    )
    assert [step["type"] for step in program] == [
        first_step,
        "toolchange",
        "transit_storage_to_assembly_start",
    ]


@pytest.mark.parametrize("start_pose", ["assembly", "dropoff"])
@pytest.mark.parametrize("tool_after", ["H5", None])
def test_no_toolchange_when_the_required_tool_is_mounted_or_none_needed(
    start_pose, tool_after
):
    env = build_env(EMO)

    assert env._toolchange_prefix(
        start_pose=start_pose, tool_before="H5", tool_after=tool_after
    ) == (0.0, [])
