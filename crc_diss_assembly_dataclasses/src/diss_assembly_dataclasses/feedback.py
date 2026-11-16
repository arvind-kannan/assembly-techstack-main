"""Feedback of a run: operation outcomes, infeasible operations and anomalies, by reference only."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReasonCode(str, Enum):
    """The closed list of reason codes the cell simulation reports, each with free text."""

    # infeasible operation
    NO_PATH_SHARE = "no_path_share"
    NO_PATH_TRANSIT = "no_path_transit"
    OUT_OF_REACH = "out_of_reach"
    COLLISION = "collision"
    SYNC_DEADLOCK = "sync_deadlock"
    CELL_STATE_MISMATCH = "cell_state_mismatch"
    UNSUPPORTED_PROCESS = "unsupported_process"
    # failed operation
    CONTACT_ABORT = "contact_abort"
    INJECTED = "injected"
    # anomaly
    CONTACT_WARNING = "contact_warning"
    NEAR_LIMIT = "near_limit"
    TIME_OVERRUN = "time_overrun"
    # request rejected
    INVALID_REQUEST = "invalid_request"
    INTERNAL_ERROR = "internal_error"


def _code(value: str | None) -> ReasonCode | None:
    return None if value is None else ReasonCode(value)


@dataclass(frozen=True)
class OperationOutcome:
    """Whether an executed operation reached its result assembly, and the product state observed."""

    operation_id: str
    succeeded: bool
    observed_product_state: tuple[str, ...]
    reason_code: ReasonCode | None = None
    text: str = ""

    def to_dict(self) -> dict:
        return {"kind": "operation_outcome", "operation_id": self.operation_id, "succeeded": self.succeeded,
                "observed_product_state": list(self.observed_product_state),
                "reason_code": None if self.reason_code is None else self.reason_code.value, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> OperationOutcome:
        return cls(d["operation_id"], bool(d["succeeded"]), tuple(d["observed_product_state"]),
                   _code(d["reason_code"]), d["text"])


@dataclass(frozen=True)
class InfeasibleOperation:
    """An operation the cell simulation cannot carry out as planned; the product is unchanged."""

    operation_id: str
    robot: str | None
    reason_code: ReasonCode
    text: str = ""

    def to_dict(self) -> dict:
        return {"kind": "infeasible_operation", "operation_id": self.operation_id, "robot": self.robot,
                "reason_code": self.reason_code.value, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> InfeasibleOperation:
        return cls(d["operation_id"], d["robot"], ReasonCode(d["reason_code"]), d["text"])


@dataclass(frozen=True)
class Anomaly:
    """Something unexpected that is not a failed operation; logged, no replanning."""

    operation_id: str
    robot: str | None
    reason_code: ReasonCode
    text: str = ""

    def to_dict(self) -> dict:
        return {"kind": "anomaly", "operation_id": self.operation_id, "robot": self.robot,
                "reason_code": self.reason_code.value, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> Anomaly:
        return cls(d["operation_id"], d["robot"], ReasonCode(d["reason_code"]), d["text"])


Feedback = OperationOutcome | InfeasibleOperation | Anomaly

_BY_KIND: dict[str, type[OperationOutcome] | type[InfeasibleOperation] | type[Anomaly]] = {
    "operation_outcome": OperationOutcome,
    "infeasible_operation": InfeasibleOperation,
    "anomaly": Anomaly,
}


def feedback_from_dict(d: dict) -> Feedback:
    return _BY_KIND[d["kind"]].from_dict(d)
