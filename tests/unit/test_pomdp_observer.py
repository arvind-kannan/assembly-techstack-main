from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from tests.conftest import FakePart

# ---------------------------------------------------------------------
# Robust import helper
# ---------------------------------------------------------------------

MODULE_CANDIDATES = [
    "disassembly_pomdp.pomdp_observer",
    "disassembly_pomdp.observer",
    "disassembly_pomdp.observation_encoder",
]


def _load_observer_module():
    last_error = None
    for module_name in MODULE_CANDIDATES:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Only skip when the candidate module itself is missing.
            # Missing dependencies inside an existing module should remain visible.
            if exc.name == module_name:
                last_error = exc
                continue
            raise

    raise ModuleNotFoundError(
        "Could not import observer module. "
        "Please add the actual module path to MODULE_CANDIDATES."
    ) from last_error


_obs_module = _load_observer_module()

ObserverConfig = _obs_module.ObserverConfig
MeasurementResult = _obs_module.MeasurementResult
SimVisibilityObserver = _obs_module.SimVisibilityObserver
ObservationEncoder = _obs_module.ObservationEncoder
_build_tool_vocab_from_env = _obs_module._build_tool_vocab_from_env


# ---------------------------------------------------------------------
# Lightweight fake assemblies / capabilities
# ---------------------------------------------------------------------


class FakeVisibilityAssembly:
    """Minimal assembly with parts and deterministic visibility behavior."""

    def __init__(self, parts: list[str], visible_parts: set[str]):
        self.parts = {name: FakePart(name) for name in parts}
        self.visible_parts = set(visible_parts)
        self.visibility_calls = []

    def check_visibility(
        self,
        part,
        viewing_directions,
        debug_visualization=False,
    ):
        self.visibility_calls.append(
            {
                "part": part.name,
                "viewing_directions": viewing_directions,
                "debug_visualization": debug_visualization,
            }
        )
        return part.name in self.visible_parts


class ToolListCapability:
    """Fallback capability that exposes list_tools()."""

    def __init__(self, tools):
        self._tools = list(tools)

    def list_tools(self):
        return list(self._tools)


def make_encoder_env(
    *,
    obs_part_names=None,
    detected_parts=None,
    current_tool="H5",
    initial_tool="H5",
    last_accepted=1,
    capabilities=None,
):
    return SimpleNamespace(
        capabilities=(
            capabilities
            if capabilities is not None
            else [ToolListCapability(["H5", "gripper_a"])]
        ),
        _obs_part_names=list(obs_part_names or ["part_1", "part_2", "hidden_part"]),
        _detected_parts=set(detected_parts or set()),
        initial_mounted_tool=initial_tool,
        current_mounted_tool=current_tool,
        _last_accepted=last_accepted,
    )


# ---------------------------------------------------------------------
# SimVisibilityObserver
# ---------------------------------------------------------------------


def test_visibility_observer_detects_visible_parts_only():
    assembly = FakeVisibilityAssembly(
        parts=["part_1", "part_2", "hidden_part"],
        visible_parts={"part_1", "part_2"},
    )

    obs = SimVisibilityObserver(cfg=ObserverConfig(debug=False))
    result = obs.observe(assembly=assembly, stage="reset")

    assert isinstance(result, MeasurementResult)
    assert result.detected_parts == {"part_1", "part_2"}
    assert result.debug is None
    assert len(assembly.visibility_calls) == 3


def test_visibility_observer_debug_contains_counts_and_stage():
    assembly = FakeVisibilityAssembly(
        parts=["part_1", "part_2", "hidden_part"],
        visible_parts={"part_1"},
    )

    obs = SimVisibilityObserver(cfg=ObserverConfig(debug=True))
    obs.reset_episode(assembly=assembly)
    result = obs.observe(assembly=assembly, stage="post")

    assert result.detected_parts == {"part_1"}
    assert result.debug is not None
    assert result.debug["stage"] == "post"
    assert result.debug["episode_id"] == 1
    assert result.debug["parts_total"] == 3
    assert result.debug["visible_cnt"] == 1
    assert result.debug["detected_cnt"] == 1


def test_visibility_observer_reset_episode_increments_episode_id():
    assembly = FakeVisibilityAssembly(parts=["part_1"], visible_parts={"part_1"})

    obs = SimVisibilityObserver(cfg=ObserverConfig(debug=True))
    obs.reset_episode(assembly=assembly)
    first = obs.observe(assembly=assembly, stage="reset")

    obs.reset_episode(assembly=assembly)
    second = obs.observe(assembly=assembly, stage="reset")

    assert first.debug["episode_id"] == 1
    assert second.debug["episode_id"] == 2


def test_visibility_observer_passes_configured_viewing_directions():
    assembly = FakeVisibilityAssembly(parts=["part_1"], visible_parts={"part_1"})
    cfg = ObserverConfig(
        viewing_directions=((1, 0, 0), (0, 0, -1)),
        debug=False,
    )

    obs = SimVisibilityObserver(cfg=cfg)
    obs.observe(assembly=assembly, stage="reset")

    call = assembly.visibility_calls[0]
    dirs = call["viewing_directions"]

    assert len(dirs) == 2
    assert np.allclose(dirs[0], np.array([1.0, 0.0, 0.0]))
    assert np.allclose(dirs[1], np.array([0.0, 0.0, -1.0]))
    assert call["debug_visualization"] is False


# ---------------------------------------------------------------------
# Tool vocabulary
# ---------------------------------------------------------------------


def test_build_tool_vocab_uses_stable_sorted_order():
    env = SimpleNamespace(
        capabilities=[
            ToolListCapability(["z_tool", "a_tool"]),
        ],
        initial_mounted_tool="z_tool",
    )

    vocab = _build_tool_vocab_from_env(env)

    assert vocab.tool_names == ["a_tool", "z_tool"]
    assert vocab.tool_index["a_tool"] == 0
    assert vocab.tool_index["z_tool"] == 1


def test_build_tool_vocab_requires_initial_tool_to_be_available():
    env = SimpleNamespace(
        capabilities=[
            ToolListCapability(["gripper_a", "T6"]),
        ],
        initial_mounted_tool="missing_tool",
    )

    with pytest.raises(ValueError, match="initial_mounted_tool"):
        _build_tool_vocab_from_env(env)


# ---------------------------------------------------------------------
# ObservationEncoder observation space
# ---------------------------------------------------------------------


def test_observation_encoder_builds_expected_observation_space():
    env = make_encoder_env(
        obs_part_names=["part_1", "part_2", "part_3"],
        capabilities=[ToolListCapability(["H5", "gripper_a"])],
    )

    enc = ObservationEncoder()
    space = enc.build_observation_space(env)

    assert isinstance(space, spaces.Dict)
    assert space["parts"].shape == (3,)
    assert space["tool"].shape == (2,)
    assert isinstance(space["accepted"], spaces.Discrete)
    assert space["accepted"].n == 2


def test_observation_encoder_requires_vocab_before_observation():
    env = make_encoder_env()

    enc = ObservationEncoder()

    with pytest.raises(RuntimeError):
        enc.build_observation(env)


def test_observation_encoder_parts_vector_uses_detected_parts_not_assembly_truth():
    env = make_encoder_env(
        obs_part_names=["part_1", "part_2", "hidden_part"],
        detected_parts={"part_2"},
    )

    enc = ObservationEncoder()
    enc.build_observation_space(env)

    obs = enc.build_observation(env)

    assert obs["parts"].dtype == np.float32
    assert np.allclose(obs["parts"], np.array([0.0, 1.0, 0.0], dtype=np.float32))


def test_observation_encoder_raises_if_detected_parts_is_not_set():
    env = make_encoder_env()
    env._detected_parts = None

    enc = ObservationEncoder()
    enc.build_observation_space(env)

    with pytest.raises(RuntimeError):
        enc.build_observation(env)


def test_observation_encoder_tool_one_hot_uses_current_mounted_tool():
    env = make_encoder_env(
        current_tool="gripper_a",
        capabilities=[ToolListCapability(["H5", "gripper_a", "T6"])],
    )

    enc = ObservationEncoder()
    enc.build_observation_space(env)

    obs = enc.build_observation(env)

    expected = np.zeros_like(obs["tool"])
    expected[enc._vocab.tool_index["gripper_a"]] = 1.0

    assert obs["tool"].dtype == np.float32
    assert np.allclose(obs["tool"], expected)


def test_observation_encoder_rejects_unknown_mounted_tool():
    env = make_encoder_env(
        current_tool="unknown_tool",
        initial_tool="H5",
        capabilities=[ToolListCapability(["H5", "gripper_a"])],
    )

    enc = ObservationEncoder()
    enc.build_observation_space(env)

    with pytest.raises(ValueError, match="Mounted tool"):
        enc.build_observation(env)


def test_observation_encoder_accepted_feedback_defaults_to_one():
    env = make_encoder_env(last_accepted=1)

    enc = ObservationEncoder()
    enc.build_observation_space(env)

    obs = enc.build_observation(env)

    assert obs["accepted"] == 1


def test_observation_encoder_accepted_feedback_reflects_last_accepted():
    env = make_encoder_env(last_accepted=0)

    enc = ObservationEncoder()
    enc.build_observation_space(env)

    obs = enc.build_observation(env)

    assert obs["accepted"] == 0


def test_observation_encoder_output_is_contained_in_space():
    env = make_encoder_env(
        obs_part_names=["part_1", "part_2"],
        detected_parts={"part_1"},
        current_tool="H5",
        last_accepted=1,
        capabilities=[ToolListCapability(["H5", "gripper_a"])],
    )

    enc = ObservationEncoder()
    space = enc.build_observation_space(env)
    obs = enc.build_observation(env)

    assert space.contains(obs)
