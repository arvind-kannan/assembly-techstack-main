"""Completing the sequence graph by reversal (ADR-0005): every removing step gets an inverse inserting
step, every unscrewing step an inverse screwing-in step, milling steps stay one-way.

Reversal does not check assembly feasibility (press fits, gravity, gripping at the supply); the cell
simulation reports such problems as infeasible operations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import networkx as nx
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_steps import Step

_INVERSE_PROCESS = {
    Process.GRIPPING: Process.INSERTING,
    Process.REMOVING: Process.INSERTING,
    Process.GRIPPING_AND_REMOVING: Process.INSERTING,
    Process.UNSCREWING: Process.SCREWING_IN,
}


@dataclass(frozen=True)
class AssemblyConstants:
    """Times added to the reversed path duration of an inverse step, in seconds.

    Placeholders until they are measured on the cell (#90); the defaults are guesses.
    """

    screw_in_time: float = 5.0
    supply_travel_time: float = 10.0
    insert_approach_time: float = 3.0
    insert_feed_time: float = 5.0


def complete_by_reversal(
    graph: nx.MultiDiGraph, constants: AssemblyConstants = AssemblyConstants()
) -> nx.MultiDiGraph:
    """A copy of `graph` with an inverse step (marked `inverse`) added for every reversible step."""
    complete = copy.deepcopy(graph)
    for u, v, attrs in graph.edges(data=True):
        step = Step.from_edge(attrs)
        process = _INVERSE_PROCESS.get(step.process)
        if process is None:
            continue  # milling is one-way
        if process is Process.SCREWING_IN:
            action_time, supply_time = step.action_time + constants.screw_in_time, None
        else:
            action_time = (
                step.action_time
                + constants.supply_travel_time
                + constants.insert_approach_time
                + constants.insert_feed_time
            )
            supply_time = constants.supply_travel_time
        inverse = Step(
            process=process,
            target_part=step.target_part,
            tool=step.tool,
            action_time=action_time,
            inverse=True,
            direction=None
            if step.direction is None
            else tuple(-x for x in step.direction),
            screw_head=step.screw_head,
            supply_time=supply_time,
            label=f"inverse:{step.target_part} [{step.tool}]",
        )
        complete.add_edge(v, u, **inverse.to_attrs())
    return complete
