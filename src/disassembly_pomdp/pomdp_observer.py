"""Observer interfaces and observation encoding for the disassembly POMDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part
from gymnasium import spaces

# The encoder intentionally reads environment-private runtime fields because
# these fields form the current internal observation contract.
# pylint: disable=protected-access,too-few-public-methods

DEFAULT_VIEWING_DIRECTIONS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
)


@dataclass
class ToolVocab:
    """Vocabulary used to convert mounted tool names into one-hot vectors."""

    tool_names: list[str]
    tool_index: dict[str, int]


@dataclass
class MeasurementResult:
    """Observer output returned to the environment after a measurement."""

    detected_parts: set[str]
    debug: dict[str, Any] | None = None


@dataclass
class ObserverConfig:
    """Configuration for simulated visibility-based observation."""

    viewing_directions: Sequence[Sequence[float]] = DEFAULT_VIEWING_DIRECTIONS
    debug: bool = False


@runtime_checkable
class Observer(Protocol):
    """Protocol implemented by all product-state observer backends."""

    def reset_episode(self, *, assembly: Assembly) -> None:
        """Reset observer-side episode state."""

    def observe(self, *, assembly: Assembly, stage: str) -> MeasurementResult:
        """
        Observe the assembly at the requested interaction stage.

        Args:
            assembly: Current hidden environment assembly.
            stage: A simple stage label, e.g. "reset", "pre", or "post".
        """


class SimVisibilityObserver(Observer):
    """
    Simulated observer based on geometry-dependent visibility checks.

    The observer reports the set of part names considered visible by
    ``Assembly.check_visibility``. It acts as a measurement-level proxy and
    does not expose the full ground-truth assembly state.
    """

    def __init__(self, *, cfg: ObserverConfig):
        self.cfg = cfg
        self._episode_id = 0
        self._viewing_directions = [
            np.asarray(direction, dtype=float).reshape(3)
            for direction in self.cfg.viewing_directions
        ]

    def reset_episode(self, *, assembly: Assembly) -> None:
        del assembly
        self._episode_id += 1

    def _is_visible(self, assembly: Assembly, part: Part) -> bool:
        """Return whether a part passes the configured visibility check."""
        return bool(
            assembly.check_visibility(
                part,
                viewing_directions=self._viewing_directions,
                debug_visualization=False,
            )
        )

    def observe(self, *, assembly: Assembly, stage: str) -> MeasurementResult:
        """Return visible part names and optional measurement diagnostics."""
        detected_parts: set[str] = set()
        debug_info: dict[str, Any] | None = None

        total_parts = 0
        visible_count = 0

        for part in assembly.parts.values():
            total_parts += 1
            if self._is_visible(assembly, part):
                visible_count += 1
                detected_parts.add(part.name)

        if self.cfg.debug:
            debug_info = {
                "stage": stage,
                "episode_id": self._episode_id,
                "parts_total": total_parts,
                "visible_cnt": visible_count,
                "detected_cnt": len(detected_parts),
            }

        return MeasurementResult(detected_parts=detected_parts, debug=debug_info)


def _build_tool_vocab_from_env(env: Any) -> ToolVocab:
    """Collect all tool names exposed by the environment capabilities."""
    tool_names: set[str] = set()

    for capability in env.capabilities:
        tool_names.update(capability.list_tools())

    all_tools = sorted(tool_names)

    if not all_tools:
        raise ValueError(
            "No tools were found in the environment capability tool lists."
        )

    initial_tool = getattr(env, "initial_mounted_tool", None)
    if initial_tool not in all_tools:
        raise ValueError(
            f"Configured initial_mounted_tool {initial_tool!r} is not available "
            f"in the environment capability tool lists: {all_tools}"
        )

    return ToolVocab(
        tool_names=all_tools,
        tool_index={name: index for index, name in enumerate(all_tools)},
    )


class ObservationEncoder:
    """
    Encode POMDP observations for Gymnasium.

    Observation fields:
        parts:
            Binary visibility/detection proxy, based on ``env._detected_parts``.
            This is not the full ground-truth assembly state.
        tool:
            One-hot encoding of the currently mounted tool.
        accepted:
            Binary feedback indicating whether the previous action was accepted.
    """

    def __init__(self) -> None:
        self._vocab: ToolVocab | None = None

    def build_observation_space(self, env: Any) -> spaces.Space:
        """Build the Gymnasium observation space and initialize tool vocabulary."""
        self._vocab = _build_tool_vocab_from_env(env)

        return spaces.Dict(
            {
                "parts": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(env._obs_part_names),),
                    dtype=np.float32,
                ),
                "tool": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(self._vocab.tool_names),),
                    dtype=np.float32,
                ),
                "accepted": spaces.Discrete(2),
            }
        )

    def build_observation(self, env: Any) -> dict[str, Any]:
        """Build one observation dictionary from the current environment state."""
        self._require_vocab()

        return {
            "parts": self._parts_detection_obs(env),
            "tool": self._tool_obs(env),
            "accepted": int(getattr(env, "_last_accepted", 1)),
        }

    def _require_vocab(self) -> ToolVocab:
        """Return the initialized vocabulary or raise a clear runtime error."""
        if self._vocab is None:
            raise RuntimeError(
                "ObservationEncoder vocab not initialized. "
                "Call build_observation_space(env) first."
            )
        return self._vocab

    def _parts_detection_obs(self, env: Any) -> np.ndarray:
        """
        Build the binary detected-part vector from ``env._detected_parts``.

        This is the measurement-level observable cue returned by the observer,
        not the full ground-truth assembly state.
        """
        part_names = env._obs_part_names
        detected_parts = getattr(env, "_detected_parts", None)

        if not isinstance(detected_parts, set):
            raise RuntimeError(
                "env._detected_parts must be set before building POMDP observation."
            )

        vector = np.zeros(len(part_names), dtype=np.float32)
        for index, name in enumerate(part_names):
            if name in detected_parts:
                vector[index] = 1.0
        return vector

    def _tool_obs(self, env: Any) -> np.ndarray:
        """Build a one-hot vector for the currently mounted tool."""
        vocab = self._require_vocab()
        vector = np.zeros(len(vocab.tool_names), dtype=np.float32)

        tool_name = getattr(env, "current_mounted_tool", None)

        if tool_name is None:
            raise RuntimeError(
                "env.current_mounted_tool must be set before building "
                "the tool observation."
            )

        if tool_name not in vocab.tool_index:
            raise ValueError(
                f"Mounted tool {tool_name!r} is not present in the "
                f"tool vocabulary: {vocab.tool_names}"
            )

        tool_index = vocab.tool_index[tool_name]
        vector[tool_index] = 1.0
        return vector
