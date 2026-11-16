"""The travel-time table of a cell: its times, in a file of their own next to the cell file (ADR-0012).

Interim until #235 reshapes it: the table holds today's flat timing keys unchanged (the fields of
`CellSettings` plus `inspection_time`, feed rates and offsets included), and the cell simulation reads its
tool-change and inspection times from it. A cell file `cell.json` has its table in `cell.travel_times.json`;
the cell's fingerprint covers both files (`graph_export.cell_fingerprint`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SUFFIX = ".travel_times.json"


@dataclass(frozen=True)
class TravelTimeTable:
    """Times and feed rates by key (seconds, mm/min), as in today's `CellSettings`."""

    times: dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.times.get(key, default))

    def __contains__(self, key: str) -> bool:
        return key in self.times

    def to_dict(self) -> dict:
        return {"times": dict(self.times)}

    @classmethod
    def from_dict(cls, d: dict) -> TravelTimeTable:
        return cls({k: float(v) for k, v in d["times"].items()})


def table_path(cell_file: str | Path) -> Path:
    """Where the travel-time table of the cell file `cell_file` is: next to it, `<name>.travel_times.json`."""
    cell_file = Path(cell_file)
    return cell_file.with_name(cell_file.stem + SUFFIX)


def load_travel_time_table(cell_file: str | Path) -> TravelTimeTable:
    path = table_path(cell_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"no travel-time table {path.name} next to the cell file {Path(cell_file).name} (ADR-0012)"
        )
    return TravelTimeTable.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_travel_time_table(table: TravelTimeTable, cell_file: str | Path) -> None:
    path = table_path(cell_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table.to_dict(), indent=2), encoding="utf-8")
