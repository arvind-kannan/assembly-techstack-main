"""What every plan function of the MRTAMP app shares: the contract type, the failure marker, and the
helpers that turn a cell-assembly snapshot into an `Assembly` and the removal plan core's result into a share.

A plan function takes the product model, the cell-assembly snapshot, the operation, and the robot and tool
of the share to fill; it returns the operation with that share planned (and the direction set) or raises `PlanFailure`,
which the caller turns into an operation the cell simulation reports as infeasible (ADR-0004, ADR-0007).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState
from diss_assembly_dataclasses.connection import (
    Connection,
    Fixed_Connection,
    Geometric_Connection,
    Safety_Ring_Connection,
    Screw_Connection,
)
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.part import Part
from diss_assembly_dataclasses.resource import Robot

from motion_planning.removal_planning import NO_PATH, RemovalPlan

if TYPE_CHECKING:  # the product-model type builds on the helpers below
    from disassembly_pomdp.product_model import ProductModel

PlanFunction = Callable[
    ["ProductModel", Cell_Assembly_Snapshot, Operation, Robot, "str | None"], Operation
]
PLAN_FAILURE = "plan_failure"  # key in Operation.parameters: the plan function's reason for failing
PLAN_FAILURE_ROBOT = (
    "plan_failure_robot"  # key in Operation.parameters: the robot whose share failed
)
SUPPLY_SLOT = "supply_slot"  # key in Operation.parameters: marks a gripping operation as gripping at the supply, naming the slot
PRODUCT_MODEL_UNIT = 0.001  # metres per unit of the product models and their planners (mm); share paths are in metres

_CONNECTIONS = {
    "generic": Connection,
    "geometric": Geometric_Connection,
    "screw": Screw_Connection,
    "safety_ring": Safety_Ring_Connection,
    "fixed": Fixed_Connection,
}


class PlanFailure(ValueError):
    """A plan function found no plan for its share (no path, no supply slot, ...). Any other exception
    from a plan function is a bug and propagates."""


def with_share(
    operation: Operation, plan: RemovalPlan, robot: Robot, tool: str | None
) -> Operation:
    """`operation` with the share of `robot` filled from the removal plan core's `plan`: the planner's path
    converted from the product model's millimetres to metres (as the cell simulation reads poses) and the plan's
    unit direction. Raises `PlanFailure` with the core's reason if the plan failed or has no path."""
    if not plan.ok:
        raise PlanFailure(plan.failure.message)
    data = plan.operation.data
    path = np.asarray(data.get("path", ()), dtype=float)
    if path.ndim != 2 or len(path) == 0:
        raise PlanFailure(NO_PATH)
    path = (
        path[:, :3] * PRODUCT_MODEL_UNIT
    )  # positions only (the known gap: no orientations)
    share = Share(
        robot.id,
        tool,
        tuple(tuple(map(float, p)) for p in path),
        float(data["path_duration"]),
    )
    shares = tuple(share if s.robot == robot.id else s for s in operation.shares)
    return replace(
        operation, shares=shares, direction=tuple(map(float, plan.direction))
    )


def assembly_xml_with(assembly_xml: str, product_state: tuple[str, ...]) -> str:
    """The assembly XML reduced to the parts of `product_state` and the connections between them."""
    root = ET.fromstring(assembly_xml)
    keep = set(product_state)
    for el in list(root):
        names = {
            el.attrib.get("name"),
            el.attrib.get("part_name1"),
            el.attrib.get("part_name2"),
        } - {None}
        if not names <= keep:
            root.remove(el)
    return ET.tostring(root, encoding="unicode")


def snapshot_at(
    model_xml: str, product_state: tuple[str, ...], cell_state: CellState
) -> Cell_Assembly_Snapshot:
    """The cell-assembly snapshot at `product_state`: `model_xml` reduced to its parts, with `cell_state`."""
    return Cell_Assembly_Snapshot(
        assembly_xml_with(model_xml, product_state), cell_state
    )


def assembly_of(snapshot: Cell_Assembly_Snapshot, folder: Path) -> Assembly:
    """The snapshot's assembly; mesh and contact paths in its XML resolve against `folder`."""
    root = ET.fromstring(snapshot.assembly_xml)
    base_dir = os.path.abspath(folder)
    assembly = Assembly(root.attrib["name"])
    for element in root.findall("part"):
        assembly.add_part(Part.from_xml_element(element, base_dir))
    for element in root.findall("connection"):
        connection_type = _CONNECTIONS[element.attrib["type"]]
        if (
            connection_type is Geometric_Connection
        ):  # the only connection that reads files
            connection = connection_type.from_xml_element(element, base_dir)
        else:
            connection = connection_type.from_xml_element(element)
        assembly.add_connection(connection)
    return assembly
