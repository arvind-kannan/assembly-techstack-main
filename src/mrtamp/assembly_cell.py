"""The cell of an assembly run: a disassembly cell whose first robot can also assemble."""

from __future__ import annotations

from dataclasses import replace

from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell

ASSEMBLY_PROCESSES = (
    Process.GRIPPING,
    Process.INSERTING,
    Process.RELEASING,
    Process.SCREWING_IN,
)


def with_assembly_capabilities(cell: Cell) -> Cell:
    """`cell` with its first robot also able to grip at the supply, insert, release and screw in."""
    first, *others = cell.robots
    capabilities = tuple(dict.fromkeys(first.capabilities + ASSEMBLY_PROCESSES))
    return replace(cell, robots=(replace(first, capabilities=capabilities), *others))
