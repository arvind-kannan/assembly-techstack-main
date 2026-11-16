from __future__ import annotations

import numpy as np

from tests.unit.test_pomdp_observer import (
    FakeVisibilityAssembly,
    ObservationEncoder,
    ObserverConfig,
    SimVisibilityObserver,
    ToolListCapability,
)


def test_visibility_observer_to_observation_encoder_pipeline():
    """
    Minimal integration test:
    visibility observer -> env._detected_parts -> observation encoder.
    """
    assembly = FakeVisibilityAssembly(
        parts=["part_1", "part_2", "hidden_part"],
        visible_parts={"part_1", "part_2"},
    )

    env = type("FakeEnv", (), {})()
    env.capabilities = [ToolListCapability(["H5", "gripper_a"])]
    env._obs_part_names = ["part_1", "part_2", "hidden_part"]
    env.initial_mounted_tool = "H5"
    env.current_mounted_tool = "gripper_a"
    env._last_accepted = 1

    observer = SimVisibilityObserver(cfg=ObserverConfig(debug=True))
    observer.reset_episode(assembly=assembly)
    measurement = observer.observe(assembly=assembly, stage="reset")

    env._detected_parts = measurement.detected_parts

    encoder = ObservationEncoder()
    space = encoder.build_observation_space(env)
    obs = encoder.build_observation(env)

    assert measurement.detected_parts == {"part_1", "part_2"}
    assert np.allclose(obs["parts"], np.array([1.0, 1.0, 0.0], dtype=np.float32))
    assert np.allclose(obs["tool"], np.array([0.0, 1.0], dtype=np.float32))
    assert obs["accepted"] == 1
    assert space.contains(obs)


def test_observer_encoder_pipeline_updates_after_visibility_change():
    """
    The encoded detected-part vector changes when observer measurements change.
    """
    assembly = FakeVisibilityAssembly(
        parts=["part_1", "part_2", "hidden_part"],
        visible_parts={"part_1"},
    )

    env = type("FakeEnv", (), {})()
    env.capabilities = [ToolListCapability(["H5", "gripper_a"])]
    env._obs_part_names = ["part_1", "part_2", "hidden_part"]
    env.initial_mounted_tool = "H5"
    env.current_mounted_tool = "H5"
    env._last_accepted = 1

    observer = SimVisibilityObserver(cfg=ObserverConfig(debug=False))
    encoder = ObservationEncoder()
    encoder.build_observation_space(env)

    first = observer.observe(assembly=assembly, stage="pre")
    env._detected_parts = first.detected_parts
    obs_first = encoder.build_observation(env)

    assembly.visible_parts = {"part_1", "hidden_part"}
    second = observer.observe(assembly=assembly, stage="post")
    env._detected_parts = second.detected_parts
    obs_second = encoder.build_observation(env)

    assert np.allclose(obs_first["parts"], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(obs_second["parts"], np.array([1.0, 0.0, 1.0], dtype=np.float32))
