"""Step expansion: one step of the sequence graph into the operations that carry it out.

An inserting step becomes gripping at the supply -> inserting -> releasing; gripping and releasing change
only the cell state. v1 plans no holding and no handover. Every other step is one operation. A planner
may use this helper or pass its own with the same signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import SupplySlot

from disassembly_pomdp.graph_steps import Step
from motion_planning.common import SUPPLY_SLOT
from mrtamp.supply import slot_for_part


class MissingSupplySlot(ValueError):
    """The step needs a part from the supply, but the cell has no supply slot for it."""


@dataclass(frozen=True)
class ExpandedOperation:
    """One operation of an expanded step; `reaches_next_state` is False while the product is unchanged."""

    process: Process
    target: str
    duration: float
    reaches_next_state: bool
    direction: tuple[float, ...] | None = None
    parameters: dict = field(default_factory=dict)


def is_gripping_at_supply(op) -> bool:
    return op.process is Process.GRIPPING and SUPPLY_SLOT in op.parameters


def update_holds(op, holds: dict[str, str]) -> None:
    """Gripping at the supply and releasing change only the cell state."""
    if is_gripping_at_supply(op):
        holds[op.robots[0]] = op.target
    elif op.process is Process.RELEASING:
        holds.pop(op.robots[0], None)


def expand_step(
    step: Step, supply_slots: tuple[SupplySlot, ...] = ()
) -> list[ExpandedOperation]:
    """The operations carrying out `step`; their durations add up to the step's `action_time`."""
    target, duration = step.target_part, step.action_time
    if step.process is not Process.INSERTING:
        return [ExpandedOperation(step.process, target, duration, True, step.direction)]
    slot = slot_for_part(supply_slots, target)
    if slot is None:
        raise MissingSupplySlot(f"no supply slot for part {target!r}")
    supply = step.supply_time or 0.0
    if supply > duration:
        raise ValueError(
            f"supply time {supply} exceeds the step time {duration} of {target!r}"
        )
    return [
        ExpandedOperation(
            Process.GRIPPING, target, supply, False, parameters={SUPPLY_SLOT: slot.name}
        ),
        ExpandedOperation(
            Process.INSERTING, target, duration - supply, True, step.direction
        ),
        ExpandedOperation(Process.RELEASING, target, 0.0, True),
    ]
