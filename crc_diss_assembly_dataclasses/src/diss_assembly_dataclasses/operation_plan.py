"""The operation plan: operations ordered per robot, linked across robots by sync points."""
from __future__ import annotations

from dataclasses import dataclass, field

from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot
from diss_assembly_dataclasses.operation import Operation


@dataclass(frozen=True)
class SyncPoint:
    """`robot` waits before operation `before` until operation `until` has finished.

    layer: "planner" (task level) or "motion" (added by the cell simulation).
    reason: "order", "handover", "holding" or "paths_too_close".
    """

    robot: str
    before: str
    until: str
    layer: str
    reason: str

    def to_dict(self) -> dict:
        return {"robot": self.robot, "before": self.before, "until": self.until,
                "layer": self.layer, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> SyncPoint:
        return cls(d["robot"], d["before"], d["until"], d["layer"], d["reason"])


@dataclass(frozen=True)
class PlanMetrics:
    """Makespan, each robot's busy time, number of tool changes and of inspections."""

    makespan: float
    busy_time: dict[str, float] = field(default_factory=dict)
    tool_changes: int = 0
    inspections: int = 0

    def to_dict(self) -> dict:
        return {"makespan": self.makespan, "busy_time": dict(self.busy_time),
                "tool_changes": self.tool_changes, "inspections": self.inspections}

    @classmethod
    def from_dict(cls, d: dict) -> PlanMetrics:
        return cls(float(d["makespan"]), {k: float(v) for k, v in d["busy_time"].items()},
                   int(d["tool_changes"]), int(d["inspections"]))


@dataclass(frozen=True)
class OperationPlan:
    """The operations of a run (in a valid execution order), each robot's sequence of
    operation ids, the sync points, the plan metrics and the snapshot the plan starts from."""

    operations: tuple[Operation, ...]
    sequences: dict[str, tuple[str, ...]]
    sync_points: tuple[SyncPoint, ...]
    metrics: PlanMetrics
    start_snapshot: Cell_Assembly_Snapshot

    def operation(self, operation_id: str) -> Operation:
        return next(op for op in self.operations if op.id == operation_id)

    def to_dict(self) -> dict:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "sequences": {r: list(ids) for r, ids in self.sequences.items()},
            "sync_points": [s.to_dict() for s in self.sync_points],
            "metrics": self.metrics.to_dict(),
            "start_snapshot": self.start_snapshot.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> OperationPlan:
        return cls(
            tuple(Operation.from_dict(op) for op in d["operations"]),
            {r: tuple(ids) for r, ids in d["sequences"].items()},
            tuple(SyncPoint.from_dict(s) for s in d["sync_points"]),
            PlanMetrics.from_dict(d["metrics"]),
            Cell_Assembly_Snapshot.from_dict(d["start_snapshot"]),
        )
