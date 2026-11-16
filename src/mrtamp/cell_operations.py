"""Operations that change only the cell state, at model level: loading, unloading and reconfiguring move a pallet or
a tooling; a machine operation (rework milling) needs a machine that offers it and a pallet on it, and changes
nothing the model sees. None of them changes the product state.

`apply_cell_operation` is the seam: the model-level stand-in steps a plan's cell state through it. An infeasible
operation is reported with a reason code of the closed list: `unsupported_process` when the robot or its tool
cannot do it (no capability, the tool grips none of the item's interfaces, the station offers no such process),
`cell_state_mismatch` when the cell state does not fit (item not where the operation expects it, coupling place
that does not accept the item or is occupied, no pallet in the machine).
"""

from __future__ import annotations

from dataclasses import replace

from diss_assembly_dataclasses.cell_state import CellState, Location
from diss_assembly_dataclasses.feedback import InfeasibleOperation, ReasonCode
from diss_assembly_dataclasses.operation import (
    MachineOperationParameters,
    Operation,
    PalletTransferParameters,
    ReconfiguringParameters,
    parameters_of,
)
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell, Interface, StationKind

from disassembly_pomdp.travel_time_table import TravelTimeTable

CELL_OPERATIONS = frozenset(
    {Process.LOADING, Process.UNLOADING, Process.RECONFIGURING, Process.REWORK_MILLING}
)
# key in the travel-time table: the rework-milling skill's default cycle time, for a machine operation without one.
# Interim, like the inspection time: skill defaults sit in the travel-time table until #235 (ADR-0012).
REWORK_MILLING_CYCLE_TIME = "rework_milling_cycle_time"


def cycle_time(op: Operation, travel_times: TravelTimeTable) -> float | None:
    """The estimated cycle time of machine operation `op`: its own, else the skill default from the travel-time
    table, else None (the operation keeps its share durations)."""
    params = parameters_of(op)
    if params.cycle_time is not None:
        return params.cycle_time
    if REWORK_MILLING_CYCLE_TIME in travel_times:
        return travel_times.get(REWORK_MILLING_CYCLE_TIME)
    return None


def with_mounted_tools(op: Operation, cell_state: CellState) -> CellState:
    """`cell_state` with the tool of each share of `op` mounted on its robot."""
    return replace(
        cell_state,
        mounted_tools={
            **cell_state.mounted_tools,
            **{s.robot: s.tool for s in op.shares if s.tool is not None},
        },
    )


def apply_cell_operation(
    op: Operation, cell_state: CellState, cell: Cell
) -> CellState | InfeasibleOperation:
    """The cell state after `op` (one of `CELL_OPERATIONS`), or why it cannot be carried out from `cell_state`.

    The robot of a loading, unloading or reconfiguring is its first share's; its tool is the share's tool, else the
    one it has mounted. The share's tool is mounted afterwards, so mounting a gripper needs no operation of its own.
    """
    robot = op.shares[0].robot
    mounted = with_mounted_tools(op, cell_state).mounted_tools
    if op.process is Process.REWORK_MILLING:
        problem = _machine_operation(op, parameters_of(op), cell_state, cell)
        after = cell_state
    else:
        problem = _capability(op, robot, cell)
        if problem is None:
            problem, after = _MOVES[op.process](
                parameters_of(op), robot, mounted.get(robot), cell_state, cell
            )
    if problem is not None:
        code, text = problem
        return InfeasibleOperation(op.id, robot, code, text)
    return replace(after, mounted_tools=mounted)


_Problem = tuple[ReasonCode, str] | None


def _mismatch(text: str) -> _Problem:
    return ReasonCode.CELL_STATE_MISMATCH, text


def _unsupported(text: str) -> _Problem:
    return ReasonCode.UNSUPPORTED_PROCESS, text


def _capability(op: Operation, robot: str, cell: Cell) -> _Problem:
    r = next(r for r in cell.robots if r.id == robot)
    if op.process not in r.capabilities:
        return _unsupported(f"{robot} cannot carry out {op.process.value}")
    return None


def _interfaces(item: str, cell: Cell) -> tuple[Interface, ...] | None:
    found = next((i for i in (*cell.pallets, *cell.tooling) if i.name == item), None)
    return None if found is None else found.interfaces


def _grips(robot: str, tool: str | None, item: str, cell: Cell) -> _Problem:
    """Does the robot's tool grip `item` through one of the item's interfaces?"""
    grips = next((t.grips for t in cell.tools if t.name == tool), ())
    if not set(grips) & set(_interfaces(item, cell)):
        return _unsupported(
            f"{robot}'s tool {tool} grips none of the interfaces of {item}"
        )
    return None


def _place_problem(
    item: str, location: Location, cell_state: CellState, cell: Cell
) -> _Problem:
    """Can `item` be set down at `location`? A coupling place must exist, accept one of its interfaces and be
    free; a tool or a robot must not have another item already."""
    if location.station is not None:
        station = cell.station(location.station)
        place = (
            None
            if station is None
            else next(
                (
                    c
                    for c in station.coupling_places
                    if c.name == location.coupling_place
                ),
                None,
            )
        )
        if place is None:
            return _mismatch(
                f"{location.station} has no coupling place {location.coupling_place}"
            )
        if place.accepts not in _interfaces(item, cell):
            return _mismatch(
                f"coupling place {place.name} of {station.name} accepts {place.accepts.value}, "
                f"which {item} does not carry"
            )
    occupant = cell_state.occupant(location)
    if occupant is not None and occupant != item:
        return _mismatch(f"{_describe(location)} is occupied by {occupant}")
    return None


def _describe(location: Location) -> str:
    if location.station is not None:
        return f"coupling place {location.coupling_place} of {location.station}"
    return location.robot or location.tool


def _item_problem(
    item: str, kind: str, where: Location, cell_state: CellState, cell: Cell
) -> _Problem:
    names = {"pallet": cell.pallets, "tooling": cell.tooling}[kind]
    if item not in {i.name for i in names}:
        return _mismatch(f"the cell has no {kind} {item}")
    if cell_state.locations.get(item) != where:
        return _mismatch(f"{item} is not at {_describe(where)}")
    return None


def _loading(
    params: PalletTransferParameters,
    robot: str,
    tool: str | None,
    cell_state: CellState,
    cell: Cell,
) -> tuple[_Problem, CellState]:
    problem = (
        _item_problem(
            params.pallet, "pallet", Location.held_by(robot), cell_state, cell
        )
        or _grips(robot, tool, params.pallet, cell)
        or _place_problem(params.pallet, params.location, cell_state, cell)
    )
    return problem, cell_state.with_location(params.pallet, params.location)


def _unloading(
    params: PalletTransferParameters,
    robot: str,
    tool: str | None,
    cell_state: CellState,
    cell: Cell,
) -> tuple[_Problem, CellState]:
    hand = Location.held_by(robot)
    problem = (
        _item_problem(params.pallet, "pallet", params.location, cell_state, cell)
        or _grips(robot, tool, params.pallet, cell)
        or _hand_free(robot, cell_state)
    )
    return problem, cell_state.with_location(params.pallet, hand)


def _hand_free(robot: str, cell_state: CellState) -> _Problem:
    carried = cell_state.occupant(Location.held_by(robot)) or cell_state.holds.get(
        robot
    )
    return None if carried is None else _mismatch(f"{robot} already carries {carried}")


def _reconfiguring(
    params: ReconfiguringParameters,
    robot: str,
    tool: str | None,
    cell_state: CellState,
    cell: Cell,
) -> tuple[_Problem, CellState]:
    problem = (
        _item_problem(params.tooling, "tooling", params.source, cell_state, cell)
        or _grips(robot, tool, params.tooling, cell)
        or _place_problem(params.tooling, params.destination, cell_state, cell)
    )
    return problem, cell_state.with_location(params.tooling, params.destination)


_MOVES = {
    Process.LOADING: _loading,
    Process.UNLOADING: _unloading,
    Process.RECONFIGURING: _reconfiguring,
}


def _machine_operation(
    op: Operation, params: MachineOperationParameters, cell_state: CellState, cell: Cell
) -> _Problem:
    station = cell.station(params.station)
    if station is None or station.kind is not StationKind.MACHINE:
        return _unsupported(f"{params.station} is not a machine of the cell")
    if op.process not in station.processes:
        return _unsupported(f"{station.name} does not offer {op.process.value}")
    pallets = {p.name for p in cell.pallets}
    if not any(
        cell_state.occupant(Location.at(station.name, c.name)) in pallets
        for c in station.coupling_places
    ):
        return _mismatch(f"no pallet on a coupling place of {station.name}")
    return None
