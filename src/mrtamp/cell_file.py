"""The cell file: the cell (robots, tools, stations with their supply slots, pallets, tooling), the
tools mounted and where each pallet and tooling is at the start, the descriptions and the notes, as JSON.
The cell's times are in its travel-time table, a file of its own next to the cell file (`disassembly_pomdp.travel_time_table`, ADR-0012), which is read and
written with it.

The fingerprint of both goes into the provenance of a sequence graph export, so a run can warn when the cell
changed since the export. `descriptions` links an element of the cell (a robot, a station, ...) by name to
the URDF file that draws it; the core cell holds no geometry files. `notes` says, by element name, where a value
comes from when it is not measured or taught: an estimated or a placeholder pose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from diss_assembly_dataclasses.cell_state import Location
from diss_assembly_dataclasses.resource import Cell, Station

from disassembly_pomdp.travel_time_table import (
    TravelTimeTable,
    load_travel_time_table,
    save_travel_time_table,
)


@dataclass(frozen=True)
class CellFile:
    cell: Cell
    mounted_tools: dict[str, str | None] = field(
        default_factory=dict
    )  # robot id -> tool at the start
    descriptions: dict[str, str] = field(
        default_factory=dict
    )  # element name -> URDF file
    simulation_only: bool = (
        False  # True: runs that target the real cell are refused with this file
    )
    locations: dict[str, Location] = field(
        default_factory=dict
    )  # pallet or tooling name -> location at the start
    notes: dict[str, str] = field(
        default_factory=dict
    )  # element name -> where its values come from (estimated, placeholder)
    travel_times: TravelTimeTable = field(
        kw_only=True
    )  # from the file next to the cell file; required, so no cell runs with zero times by accident


def save_cell_file(cell_file: CellFile, path: str | Path) -> None:
    data = {
        "cell": cell_file.cell.to_dict(),
        "mounted_tools": dict(cell_file.mounted_tools),
        "descriptions": dict(cell_file.descriptions),
        "simulation_only": cell_file.simulation_only,
    }
    # Left out when empty, so a cell file without pallets, tooling or notes keeps its JSON.
    if cell_file.locations:
        data["locations"] = {
            item: at.to_dict() for item, at in cell_file.locations.items()
        }
    if cell_file.notes:
        data["notes"] = dict(cell_file.notes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    save_travel_time_table(cell_file.travel_times, path)


def load_cell_file(path: str | Path) -> CellFile:
    """The cell file at `path` with the travel-time table next to it. A cell file that still holds the cell's
    times (`timing`, before AT 174) is refused: its times go into the table file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "timing" in data["cell"]:
        raise ValueError(
            f"{Path(path).name}: the cell holds no times (ADR-0012); move cell.timing into the travel-time "
            f'table next to it as {{"times": ...}}'
        )
    return CellFile(
        Cell.from_dict(data["cell"]),
        dict(data.get("mounted_tools", {})),
        dict(data.get("descriptions", {})),
        bool(data.get("simulation_only", False)),
        {
            item: Location.from_dict(at)
            for item, at in data.get("locations", {}).items()
        },
        dict(data.get("notes", {})),
        travel_times=load_travel_time_table(path),
    )


def with_station(cell: Cell, station: Station) -> Cell:
    """`cell` with `station` in place of its station of the same name, or added when it has none."""
    if cell.station(station.name) is None:
        return replace(cell, stations=(*cell.stations, station))
    return replace(
        cell,
        stations=tuple(station if s.name == station.name else s for s in cell.stations),
    )
