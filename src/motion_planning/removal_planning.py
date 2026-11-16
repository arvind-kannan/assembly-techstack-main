"""The removal plan core: one function per removal process that calls the sequence generator's planner and
returns one `RemovalPlan`, deciding once whether the plan failed and why, the unit removal direction (with the
fallback to the part's degrees of freedom), and whether the result assembly really lost the part.

The gym's capabilities (through `unscrewing_plan_func` / `gripping_and_removing_plan_func`, ADR-0004) and the
MRTAMP plan functions (`mrtamp.plan_functions`, `motion_planning.assembly_plan_functions`) only translate that
result into their own forms (#234).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.connection import Screw_Connection

MAX_GRIP_WIDTH = 1000.0  # mm; no gripper of the cell limits the grip today
REMOVAL_PLAN = "removal_plan"  # key in the planned operation's data: its RemovalPlan
PREFERRED_DOF_AXIS = np.array(
    [0.0, 0.0, 1.0]
)  # fixes the sign of a direction from the DOF (TEMP)
NO_PATH = "the removal planner found no path"
_PLANNER_DETAILS = (  # the planner's diagnostics a planning failure carries
    "candidate_failures",
    "candidate_failures_before_selected",
    "dof_candidate_count",
    "raw_dof_count",
    "invalid_dof_count",
    "gripping_radius",
    "gripping_height",
    "max_grip_width",
)


@dataclass(frozen=True)
class RemovalFailure:
    """Why a removal plan failed. `stage` is "planning" (the planner found no plan; `reason` is its failure
    stage) or "direction" (no usable removal direction; `reason` is "missing_direction" or "bad_direction")."""

    stage: str
    reason: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RemovalPlan:
    """The one result of a removal plan: the planned operation, and either a failure or the unit removal
    direction. `result` is the assembly after the removal, present only if the part is really gone from it."""

    operation: Any
    direction: np.ndarray | None = None
    result: Assembly | None = None
    failure: RemovalFailure | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None


def screw_connection_of(assembly: Assembly, screw_part: str) -> Screw_Connection | None:
    """The screw connection whose screw (`part_name1`) is `screw_part`."""
    return next(
        (
            c
            for c in assembly.connections
            if isinstance(c, Screw_Connection) and c.part_name1 == screw_part
        ),
        None,
    )


def plan_unscrewing(
    assembly: Assembly, connection: Screw_Connection, cost: float = 0.0
) -> RemovalPlan:
    """Plan taking out the screw of `connection`."""
    from crc_disassembly_sequence_generator.planning import (
        HierarchicalDisassemblySequencePlanner as Planner,
    )

    planned, result = Planner.plan_screwing(assembly, connection, cost)
    return removal_plan_of(planned, result, connection.part_name1, assembly)


def plan_removal(assembly: Assembly, part_name: str, cost: float = 0.0) -> RemovalPlan:
    """Plan gripping and removing `part_name`."""
    from crc_disassembly_sequence_generator.planning import (
        HierarchicalDisassemblySequencePlanner as Planner,
    )

    part = assembly.get_part_by_name(part_name)
    if part is None:
        return RemovalPlan(
            None,
            failure=RemovalFailure(
                "planning", "part_missing", f"{part_name} is not in the assembly"
            ),
        )
    planned, result = Planner.plan_gripping_and_removing(
        assembly, part, cost, max_grip_width=MAX_GRIP_WIDTH
    )
    return removal_plan_of(planned, result, part_name, assembly)


def removal_plan_of(
    planned: Any,
    result: Assembly | None,
    part_name: str | None,
    assembly: Assembly | None = None,
) -> RemovalPlan:
    """Judge a removal planner's output: the planned operation and the result assembly it returned for
    removing `part_name`. The direction falls back to the DOF of `part_name` in `assembly` (none without
    `assembly`). An operation that already carries its RemovalPlan is judged once: that plan is returned."""
    data = getattr(planned, "data", None)
    data = data if isinstance(data, dict) else {}
    if isinstance(data.get(REMOVAL_PLAN), RemovalPlan):
        return data[REMOVAL_PLAN]

    if (
        data.get("failure_stage") is not None
        or data.get("failure_reason")
        or float(data.get("cost", 0.0)) < 0.0
    ):
        details = {"planner_failure_reason": data.get("failure_reason")}
        details.update({k: data.get(k) for k in _PLANNER_DETAILS})
        failure = RemovalFailure(
            "planning",
            str(data.get("failure_stage") or "planning_failed"),
            str(data.get("failure_reason") or NO_PATH),
            details,
        )
        return RemovalPlan(planned, failure=failure)

    direction, error = _unit(data.get("direction"))
    if direction is None:
        fallback, fallback_error = None, {"why": "direction_is_none"}
        if assembly is not None and part_name is not None:
            fallback, fallback_error = _unit(_dof_direction(assembly, part_name))
        if fallback is None:
            reason = (
                "bad_direction"
                if data.get("direction") is not None
                else "missing_direction"
            )
            failure = RemovalFailure(
                "direction",
                reason,
                f"no valid removal direction for {part_name}",
                {
                    "why": "no_valid_direction",
                    "op_direction_error": error,
                    "fallback_direction_error": fallback_error,
                },
            )
            return RemovalPlan(planned, failure=failure)
        direction = fallback

    removed = result if part_removed(result, part_name) else None
    return RemovalPlan(planned, direction, removed)


def part_removed(assembly: Assembly | None, part_name: str | None) -> bool:
    """Whether `assembly` is an assembly that no longer holds `part_name`."""
    return (
        isinstance(assembly, Assembly)
        and part_name is not None
        and assembly.get_part_by_name(part_name) is None
    )


def unscrewing_plan_func(assembly: Assembly, op):
    """The core as a capability plan function (ADR-0004) for unscrewing the connection in `op.data`."""
    data = getattr(op, "data", {})
    connection = data.get("connection") if isinstance(data, dict) else None
    if connection is None:
        return op, None
    plan = plan_unscrewing(assembly, connection, float(data.get("cost", 5.0)))
    return _with_removal_plan(op, plan), plan.result


def gripping_and_removing_plan_func(assembly: Assembly, op):
    """The core as a capability plan function (ADR-0004) for removing the part named in `op.data`."""
    data = getattr(op, "data", {})
    part_name = data.get("part_name") if isinstance(data, dict) else None
    if part_name is None or assembly.get_part_by_name(part_name) is None:
        return op, None
    plan = plan_removal(assembly, part_name, float(data.get("cost", 5.0)))
    return _with_removal_plan(op, plan), plan.result


def _with_removal_plan(op, plan: RemovalPlan):
    """The planner's operation (or `op` if it gave none), carrying `plan` in its data."""
    out = plan.operation if plan.operation is not None else op
    if not isinstance(getattr(out, "data", None), dict):
        out.data = {}
    out.data[REMOVAL_PLAN] = plan
    return out


def _unit(vector) -> tuple[np.ndarray | None, dict | None]:
    """(the unit vector of `vector`, None), or (None, why not)."""
    if vector is None:
        return None, {"why": "direction_is_none"}
    try:
        v = np.asarray(vector, dtype=float).reshape(3)
        n = float(np.linalg.norm(v))
    except Exception as e:
        return None, {"why": "direction_not_vector3", "err": repr(e)}
    if (not np.isfinite(n)) or n <= 1e-6:
        return None, {
            "why": "direction_zero_or_nan",
            "norm": n,
            "finite": bool(np.isfinite(n)),
        }
    return v / n, None


def _dof_direction(assembly: Assembly, part_name: str) -> np.ndarray | None:
    """A stable unit direction from the DOF of `part_name`: their mean, or if that is degenerate the DOF best
    aligned with PREFERRED_DOF_AXIS, with its sign fixed towards that axis. None without usable DOF."""
    try:
        part = assembly.get_part_by_name(part_name)
        dof = assembly.get_part_dof(part) if part is not None else None
    except Exception:
        return None
    if dof is None or len(dof) == 0:
        return None
    dof_array = np.asarray(dof, dtype=float)
    if dof_array.ndim != 2 or dof_array.shape[1] != 3:
        return None
    norms = np.linalg.norm(dof_array, axis=1)
    units = dof_array[norms > 1e-6] / norms[norms > 1e-6, None]
    if len(units) == 0:
        return None
    mean = units.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm <= 1e-6:
        direction = units[int(np.argmax(units @ PREFERRED_DOF_AXIS))]
    else:
        direction = mean / mean_norm
    return -direction if float(direction @ PREFERRED_DOF_AXIS) < 0 else direction
