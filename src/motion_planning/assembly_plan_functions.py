"""Assembly plan functions: one per assembly process, built on the removal planners of the sequence generator.

*Inserting* plans the insert of a held part: free-direction search and grip point on the step's result
assembly (the assembly with the part in), reversed. *Screwing in* plans the screw approach on the result
assembly (screw axis by PCA + cylinder fit in `plan_screwing`), reversed, and starts at the supply slot of
its screw type that serves its robot: it fetches its own screw within its operation. *Gripping at the supply* reads the supply
slot. Poses are tool positions in the frame of the reference part (usually the base part in the fixture)
named in the operation's parameters; supply slots come in the cell frame (`mrtamp.supply.slots_in_cell_frame`)
and are named by slot instead.
Assembly feasibility (press fits, gravity) is not checked here: the removal planners know only the free
directions of the result assembly, and the cell simulation reports what it cannot carry out.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Iterable

from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot, SupplySlot

from motion_planning.common import (
    SUPPLY_SLOT,
    PlanFailure,
    PlanFunction,
    with_share,
)
from motion_planning.removal_planning import (
    plan_removal,
    plan_unscrewing,
    screw_connection_of,
)

if TYPE_CHECKING:
    from disassembly_pomdp.product_model import ProductModel

REFERENCE_PART = "reference_part"  # key in Operation.parameters: the part the operation's poses are relative to


def assembly_plan_functions(
    supply_slots: Iterable[SupplySlot] = (),
    *,
    reference_part: str,
) -> dict[Process, PlanFunction]:
    """Plan functions for inserting, screwing in and gripping at the supply.

    The result assembly comes from the product model they are given (the part a step inserts is not in the
    snapshot before the step); its mesh and contact paths resolve against the product model's folder.
    """
    slots = tuple(supply_slots)

    def result_assembly(
        product_model: ProductModel,
        snapshot: Cell_Assembly_Snapshot,
        operation: Operation,
    ):
        return product_model.assembly_of(
            product_model.snapshot_at(
                operation.expected_product_state, snapshot.cell_state
            )
        )

    def inserting(product_model, snapshot, operation, robot, tool):
        assembly = result_assembly(product_model, snapshot, operation)
        removal = plan_removal(assembly, operation.target)
        return _reversed(
            with_share(operation, removal, robot, tool), robot, reference_part
        )

    def screwing_in(product_model, snapshot, operation, robot, tool):
        assembly = result_assembly(product_model, snapshot, operation)
        connection = screw_connection_of(assembly, operation.target)
        if connection is None:
            raise PlanFailure(
                f"{operation.target} has no screw connection in the result assembly"
            )
        slot = next(
            (
                s
                for s in slots
                if s.screw_type == connection.screw_head and s.serves(robot.id)
            ),
            None,
        )
        if slot is None:
            raise PlanFailure(
                f"no supply slot for screw type {connection.screw_head!r} that serves {robot.id}"
            )
        unscrewing = plan_unscrewing(assembly, connection)
        planned = _reversed(
            with_share(operation, unscrewing, robot, tool), robot, reference_part
        )
        return _fetching(planned, robot, slot)

    def gripping_at_supply(product_model, snapshot, operation, robot, tool):
        name = operation.parameters.get(SUPPLY_SLOT)
        if name is None:
            raise PlanFailure(
                f"{operation.id} is not gripping at the supply; no plan function for other gripping"
            )
        slot = next((s for s in slots if s.name == name), None)
        if slot is None:
            raise PlanFailure(f"no supply slot named {name!r} in the cell")
        share = next(s for s in operation.shares if s.robot == robot.id)
        return _replace_share(
            operation,
            replace(
                share,
                path=(tuple(slot.pose[:3]),),
                skill_parameters={**share.skill_parameters, "supply_slot": slot.name},
            ),
        )

    return {
        Process.INSERTING: inserting,
        Process.SCREWING_IN: screwing_in,
        Process.GRIPPING: gripping_at_supply,
    }


def _reversed(operation: Operation, robot: Robot, reference_part: str) -> Operation:
    """The removal-planned operation as its assembly counterpart: path and direction reversed."""
    share = next(s for s in operation.shares if s.robot == robot.id)
    operation = _replace_share(operation, replace(share, path=share.path[::-1]))
    return replace(
        operation,
        direction=tuple(-x for x in operation.direction),
        parameters={**operation.parameters, REFERENCE_PART: reference_part},
    )


def _fetching(operation: Operation, robot: Robot, slot: SupplySlot) -> Operation:
    """The operation's share starting at the supply slot, where it fetches its screw."""
    share = next(s for s in operation.shares if s.robot == robot.id)
    return _replace_share(
        operation,
        replace(
            share,
            path=(tuple(slot.pose[:3]), *share.path),
            skill_parameters={**share.skill_parameters, "supply_slot": slot.name},
        ),
    )


def _replace_share(operation: Operation, share: Share) -> Operation:
    return replace(
        operation,
        shares=tuple(share if s.robot == share.robot else s for s in operation.shares),
    )
