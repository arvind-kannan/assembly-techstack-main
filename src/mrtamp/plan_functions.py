"""Plan functions of the MRTAMP app: one per process, filling each share's path and path duration.

A plan function takes the product model (the whole product, from the planning request), the cell-assembly
snapshot (the assembly before the operation), the operation, and the robot and tool of the share to fill.
It returns the operation with that share planned (and the direction set) or raises `PlanFailure`;
`fill_shares` turns that into an operation marked with the reason and the robot of the failing share, which
the cell simulation reports as infeasible (ADR-0004, ADR-0007). Any other exception is a bug and propagates.
Poses are tool positions in the assembly frame (translations only, see the known gap in the Motion planning
context).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot
from diss_assembly_dataclasses.operation import Operation
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot

from disassembly_pomdp.product_model import ProductModel
from motion_planning.common import (
    PLAN_FAILURE,
    PLAN_FAILURE_ROBOT,
    PlanFailure,
    PlanFunction,
    with_share,
)
from motion_planning.removal_planning import (
    plan_removal,
    plan_unscrewing,
    screw_connection_of,
)


def fill_shares(
    operation: Operation,
    product_model: ProductModel,
    snapshot: Cell_Assembly_Snapshot,
    robots: Mapping[str, Robot],
    plan_functions: Mapping[Process, PlanFunction],
) -> Operation:
    """Plan every share of `operation` with the plan function of its process; unknown processes stay nominal."""
    plan_function = plan_functions.get(operation.process)
    if plan_function is None:
        return operation
    for share in operation.shares:
        try:
            operation = plan_function(
                product_model, snapshot, operation, robots[share.robot], share.tool
            )
        except PlanFailure as failure:  # other exceptions are bugs and propagate
            return replace(
                operation,
                parameters={
                    **operation.parameters,
                    PLAN_FAILURE: str(failure),
                    PLAN_FAILURE_ROBOT: share.robot,
                },
            )
    return operation


def removal_plan_functions() -> dict[Process, PlanFunction]:
    """Adapters around the sequence generator's removal planners; mesh and contact paths resolve against the
    folder of the product model they are given."""

    def gripping_and_removing(product_model, snapshot, operation, robot, tool):
        plan = plan_removal(product_model.assembly_of(snapshot), operation.target)
        return with_share(operation, plan, robot, tool)

    def unscrewing(product_model, snapshot, operation, robot, tool):
        assembly = product_model.assembly_of(snapshot)
        connection = screw_connection_of(assembly, operation.target)
        if connection is None:
            raise PlanFailure(
                f"{operation.target} has no screw connection in the assembly"
            )
        return with_share(operation, plan_unscrewing(assembly, connection), robot, tool)

    return {
        Process.GRIPPING_AND_REMOVING: gripping_and_removing,
        Process.UNSCREWING: unscrewing,
    }
