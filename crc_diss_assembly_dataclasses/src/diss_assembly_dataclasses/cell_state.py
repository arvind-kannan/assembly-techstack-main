"""Cell state and the cell-assembly snapshot. Holds here do not count toward the product state."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace

from diss_assembly_dataclasses.resource import Pose, Station, _pose


@dataclass(frozen=True)
class Location:
    """Where a pallet or a tooling is: on a coupling place of a station (a process module's or the clamping
    desk's seat included), held by a robot, or in a tool (the jaws of a gripper). Exactly one of the three."""

    station: str | None = None
    coupling_place: str | None = None
    robot: str | None = None
    tool: str | None = None

    def __post_init__(self):
        kinds = [self.station is not None or self.coupling_place is not None, self.robot is not None,
                 self.tool is not None]
        if sum(kinds) != 1 or (kinds[0] and (self.station is None or self.coupling_place is None)):
            raise ValueError(
                "a location is exactly one of: a station with its coupling place, a robot, a tool; "
                f"got {self.to_dict()}"
            )

    @classmethod
    def at(cls, station: str, coupling_place: str) -> Location:
        return cls(station=station, coupling_place=coupling_place)

    @classmethod
    def held_by(cls, robot: str) -> Location:
        return cls(robot=robot)

    @classmethod
    def in_tool(cls, tool: str) -> Location:
        return cls(tool=tool)

    def to_dict(self) -> dict:
        return {k: v for k, v in (("station", self.station), ("coupling_place", self.coupling_place),
                                  ("robot", self.robot), ("tool", self.tool)) if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> Location:
        return cls(d.get("station"), d.get("coupling_place"), d.get("robot"), d.get("tool"))


@dataclass(frozen=True)
class CellState:
    """Which part each robot holds and which tool each robot has mounted, at one moment; where each pallet and
    each tooling is (`locations`, by name); and a measured pose per station that overrides the nominal pose of
    the cell file (`station_poses`, cell frame).

    A held part may be outside the assembly; a handover is two robots holding the same part. Pallets and tooling
    are not parts: a robot carrying a pallet has it in `locations`, not in `holds`.
    """

    holds: dict[str, str] = field(default_factory=dict)  # robot id -> held part name
    mounted_tools: dict[str, str | None] = field(default_factory=dict)  # robot id -> tool name
    locations: dict[str, Location] = field(default_factory=dict)  # pallet or tooling name -> location
    station_poses: dict[str, Pose] = field(default_factory=dict)  # station name -> measured pose

    def __post_init__(self):
        object.__setattr__(
            self, "station_poses", {s: _pose(p, f"station {s} (measured)") for s, p in self.station_poses.items()}
        )

    def occupant(self, location: Location) -> str | None:
        """The pallet or tooling at `location`, or None."""
        return next((item for item, at in self.locations.items() if at == location), None)

    def with_location(self, item: str, location: Location) -> CellState:
        """This cell state with `item` moved to `location`."""
        return replace(self, locations={**self.locations, item: location})

    def station_pose(self, station: Station) -> Pose:
        """The measured pose of `station` if there is one, else its nominal pose from the cell."""
        return self.station_poses.get(station.name, station.pose)

    def to_dict(self) -> dict:
        d = {"holds": dict(self.holds), "mounted_tools": dict(self.mounted_tools)}
        # Left out when empty, so a cell state without pallets, tooling or measured poses keeps its dict.
        if self.locations:
            d["locations"] = {item: at.to_dict() for item, at in self.locations.items()}
        if self.station_poses:
            d["station_poses"] = {s: list(p) for s, p in self.station_poses.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CellState:
        return cls(
            dict(d["holds"]),
            dict(d["mounted_tools"]),
            {item: Location.from_dict(at) for item, at in d.get("locations", {}).items()},
            dict(d.get("station_poses", {})),
        )


@dataclass(frozen=True)
class Cell_Assembly_Snapshot:
    """The assembly together with the cell state at one moment, real or planned.

    The assembly travels as XML text (`Assembly.save_as_xml` format); parts are referenced by name.
    """

    assembly_xml: str
    cell_state: CellState

    @property
    def product_state_key(self) -> tuple[str, ...]:
        """The product state, as the sorted names of the parts present."""
        root = ET.fromstring(self.assembly_xml)
        return tuple(sorted(p.attrib["name"] for p in root.findall("part")))

    def to_dict(self) -> dict:
        return {"assembly_xml": self.assembly_xml, "cell_state": self.cell_state.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> Cell_Assembly_Snapshot:
        return cls(d["assembly_xml"], CellState.from_dict(d["cell_state"]))
