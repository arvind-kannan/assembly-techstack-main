"""Base interfaces and lightweight target references for disassembly capabilities."""

from __future__ import annotations

import contextlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence, TypeAlias

import numpy as np
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot
from gymnasium import spaces

from disassembly_pomdp.config import CellSettings
from motion_planning.removal_planning import REMOVAL_PLAN, removal_plan_of

STAND_IN_ROBOT = "robot_1"
"""The one robot of a capability built without the robots of a cell."""


@contextlib.contextmanager
def suppress_stdout():
    """Temporarily redirect stdout to os.devnull to silence noisy geometry helpers."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def planned_operation(op: Any, plan_output: Any, part_name: str, assembly: Assembly):
    """The operation a plan function (ADR-0004) returned for the shell `op`, translated from the removal plan
    core's judgement of it: the shell's request data merged with the planner's, marked planned, with the unit
    `direction`, the `removal_plan`, and `result_assembly` only if the part is really gone (#234)."""
    planned, result = (
        plan_output
        if isinstance(plan_output, tuple) and len(plan_output) == 2
        else (plan_output, None)
    )
    plan = removal_plan_of(planned, result, part_name, assembly)
    out = planned if planned is not None else op
    merged = {}
    for source in (getattr(op, "data", None), getattr(out, "data", None)):
        if isinstance(source, dict):
            merged.update({k: v for k, v in source.items() if k != "result_assembly"})
    merged["planned"] = True
    merged[REMOVAL_PLAN] = plan
    if plan.direction is not None:
        merged["direction"] = plan.direction
    if plan.result is not None:
        merged["result_assembly"] = plan.result
    out.data = merged
    return out


ConnKey: TypeAlias = tuple[str, str, str | None]
"""Stable screw connection key: ``(screw_part_name, base_part_name, screw_head)``."""


class TargetKind:
    """String constants used to classify action targets before decoding."""

    PART = "part"
    CONNECTION = "connection"
    BUFFER_SLOT = "buffer_slot"
    PART_NEVER_EXISTED = "part_never_existed"


@dataclass(frozen=True, slots=True)
class TargetRef:
    """
    Lightweight reference to the intended target of a local capability action.

    Examples:
        - ``TargetKind.PART``: ``name`` is required.
        - ``TargetKind.CONNECTION``: ``key`` is required.
        - ``TargetKind.BUFFER_SLOT``: ``name`` may refer to a reserved index.
        - ``TargetKind.PART_NEVER_EXISTED``: ``name`` is required.
    """

    kind: str
    name: str | None = None
    key: ConnKey | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_info(self) -> dict[str, Any]:
        """Return a logging-friendly dictionary representation."""
        info: dict[str, Any] = {
            "kind": self.kind,
            "name": self.name,
            "key": self.key,
        }
        if self.meta:
            info["meta"] = dict(self.meta)
        return info


class Capability(ABC):
    """Abstract base class for capability-conditioned disassembly actions.

    Each subclass declares the core ``process`` its steps carry out; subclasses of a
    capability inherit it. The robots for a tool are the core robots of the cell that
    carry the tool and can carry out the process (#230).
    """

    process: ClassVar[Process]

    def __init__(self, name: str, robots: Sequence[Robot] | None = None) -> None:
        if not isinstance(getattr(type(self), "process", None), Process):
            raise TypeError(
                f"{type(self).__name__} declares no core process: "
                "set the class attribute `process` to a `Process`"
            )
        self.name = name
        # None: no cell given, one stand-in robot carries all tools of the capability.
        self._robots: tuple[Robot, ...] | None = (
            None if robots is None else tuple(robots)
        )

        # Last execution-gate record, read by the environment for info/no-op classification.
        self._last_exec_gate: dict[str, Any] | None = None

    @abstractmethod
    def list_tools(self) -> list[str]:
        """Return the tools this capability can act with, in action-index order."""

    @property
    def robots(self) -> tuple[Robot, ...]:
        """The core robots of the cell. Without a cell: one stand-in ``robot_1`` that
        carries all tools of the capability and can carry out its process."""
        if self._robots is not None:
            return self._robots
        return (
            Robot(
                STAND_IN_ROBOT,
                1,
                tools=tuple(self.list_tools()),
                capabilities=(self.process,),
            ),
        )

    def robots_for(self, tool_name: str) -> tuple[str, ...]:
        """The ids of the robots, in cell order, that carry ``tool_name`` and can carry out the process."""
        return tuple(
            r.id for r in self.robots if r.can_carry_out(self.process, tool_name)
        )

    def robot_for(self, tool_name: str) -> str | None:
        """The robot that acts with ``tool_name``: the first in cell order, or None if none can."""
        return next(iter(self.robots_for(tool_name)), None)

    @abstractmethod
    def build_action_space(self, assembly: Assembly) -> spaces.Space:
        """Return the local action space handled by this capability."""

    @abstractmethod
    def decode_action(self, assembly: Assembly, raw_action: Any) -> Any:
        """Decode a raw local action into a capability-specific semantic action."""

    @abstractmethod
    def deterministic_transition(self, assembly: Assembly, action: Any) -> Assembly:
        """Return the success-conditional next assembly for a semantic action."""

    @abstractmethod
    def estimate_action_time(
        self,
        assembly: Assembly,
        action: Any,
        settings: CellSettings,
        tool_before: str | None,
    ) -> float:
        """Return action-start execution time in seconds.

        Tool-change and start-pose travel are handled by the environment.
        """

    def get_target_ref(
        self,
        assembly: Assembly,
        raw_action: np.ndarray | Any,
    ) -> TargetRef | None:
        """Optionally map a raw action to a pre-decode target reference."""
        del assembly, raw_action
        return None

    def is_target_present(self, assembly: Assembly, target: TargetRef) -> bool:
        """Return whether the referenced target currently exists."""
        del assembly, target
        return True

    def predecode_mask_reason(
        self,
        assembly: Assembly,
        raw_action: np.ndarray,
        current_tool_name: str | None = None,
        cell_state: CellState | None = None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Capability-specific pre-decode masking hook.

        Holds and mounted tools are read from ``cell_state`` (none: no holds, the
        mounted tool comes from ``current_tool_name`` or the cached one).

        This method is called by the environment after basic validation
        (range checks, buffer slots, target existence) and before decoding.

        It should filter actions that have near-zero likelihood of changing
        the state, for example:
            - ``wrong_tool``: tool-part incompatibility (C_k)
            - ``geometrically_infeasible``: cheap geometric gate (ψ)
            - ``base_part``: domain-specific non-removable part or fixture

        Returns:
            Tuple ``(masked, reason, meta)``.
        """
        del assembly, raw_action, current_tool_name, cell_state
        return False, None, {}

    def cell_state_transition(
        self,
        cell_state: CellState,
        action: Any,
        next_assembly: Assembly,
    ) -> CellState:
        """Return the cell state after a successful ``action`` that produced ``next_assembly``.

        Holds are cell state, not product state, so they change here and not in
        ``deterministic_transition``. The default leaves the cell state as is.
        """
        del action, next_assembly
        return cell_state

    def set_last_exec_gate(
        self,
        ok: bool | None,
        stage: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Store the last execution-gate record for env-level diagnostics.

        ``ok=None`` means the gate was not evaluated (yet); it is kept as ``None``.
        """
        self._last_exec_gate = {
            "ok": None if ok is None else bool(ok),
            "stage": stage,
            "reason": reason,
            "details": details or {},
        }

    def get_last_exec_gate(self) -> dict[str, Any] | None:
        """Return the last execution-gate record, or ``None`` before the first one."""
        return self._last_exec_gate
