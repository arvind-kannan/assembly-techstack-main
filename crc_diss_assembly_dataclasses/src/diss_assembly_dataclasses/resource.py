"""Resources of the cell: the cell itself, its robots, tools, stations, pallets and tooling. Plain data.

The one geometry here is a pose: `(x, y, z, rx, ry, rz)` in metres and radians, the rotation as fixed-axis
angles about x, then y, then z (roll, pitch, yaw). A station's pose is in the cell frame (Robot 1's base at
the origin); the poses of its coupling places and supply slots are relative to the station.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from diss_assembly_dataclasses.process import Process

Pose = tuple[float, float, float, float, float, float]


def _pose(values, what: str) -> Pose:
    pose = tuple(float(v) for v in values)
    if len(pose) != 6:
        raise ValueError(f"the pose of {what} needs six numbers (x, y, z, rx, ry, rz), got {len(pose)}")
    return pose


class Interface(str, Enum):
    """The closed list of mechanical couplings through which an item is gripped or seated."""

    SWS = "sws"
    FINGER_INTERFACE = "finger_interface"
    NSR_A = "nsr_a"
    ZERO_POINT_CLAMPING = "zero_point_clamping"


class StationKind(str, Enum):
    TOOL_STORAGE = "tool_storage"
    SUPPLY = "supply"
    DROPOFF = "dropoff"
    FIXTURE = "fixture"
    MACHINE = "machine"  # own control, runs a cycle the cell starts
    PROCESS_MODULE = "process_module"  # operated by a robot


@dataclass(frozen=True)
class Tool:
    """An end effector; for unscrewing and screwing in its name equals the screw head.
    `grips` lists the interfaces it grips items through (empty for a tool that grips none)."""

    name: str
    process: Process
    grips: tuple[Interface, ...] = ()

    def to_dict(self) -> dict:
        return {"name": self.name, "process": self.process.value, "grips": [i.value for i in self.grips]}

    @classmethod
    def from_dict(cls, d: dict) -> Tool:
        return cls(d["name"], Process(d["process"]), tuple(Interface(i) for i in d.get("grips", ())))


@dataclass(frozen=True)
class Robot:
    """A physical arm of the cell, the tools it can mount and the processes it can carry out."""

    id: str
    nc_channel: int
    tools: tuple[str, ...] = ()
    capabilities: tuple[Process, ...] = ()

    def can_carry_out(self, process: Process, tool: str | None) -> bool:
        return process in self.capabilities and (tool is None or tool in self.tools)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "nc_channel": self.nc_channel,
            "tools": list(self.tools),
            "capabilities": [p.value for p in self.capabilities],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Robot:
        return cls(d["id"], int(d["nc_channel"]), tuple(d["tools"]), tuple(Process(p) for p in d["capabilities"]))


@dataclass(frozen=True)
class CouplingPlace:
    """A place of a station where an item is set down and coupled through the one interface it accepts."""

    name: str
    pose: Pose  # relative to the station
    accepts: Interface

    def __post_init__(self):
        object.__setattr__(self, "pose", _pose(self.pose, f"coupling place {self.name}"))

    def to_dict(self) -> dict:
        return {"name": self.name, "pose": list(self.pose), "accepts": self.accepts.value}

    @classmethod
    def from_dict(cls, d: dict) -> CouplingPlace:
        return cls(d["name"], d["pose"], Interface(d["accepts"]))


@dataclass(frozen=True)
class SupplySlot:
    """A place of the supply station that serves one part (by name) or one screw type (by screw head).

    `robots` limits the slot to these robots, such as a screw nest plate per robot; empty serves every robot.
    """

    name: str
    pose: Pose  # relative to the supply station
    part: str | None = None
    screw_type: str | None = None
    robots: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "pose", _pose(self.pose, f"supply slot {self.name}"))
        object.__setattr__(self, "robots", tuple(self.robots))

    def serves(self, robot: str) -> bool:
        return not self.robots or robot in self.robots

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pose": list(self.pose),
            "part": self.part,
            "screw_type": self.screw_type,
            "robots": list(self.robots),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SupplySlot:
        return cls(d["name"], d["pose"], d.get("part"), d.get("screw_type"), tuple(d.get("robots", ())))


_OFFER_PROCESSES = (StationKind.MACHINE, StationKind.PROCESS_MODULE)


@dataclass(frozen=True)
class Station:
    """A named place of the cell: its kind, its pose in the cell frame, the processes it offers (machines and
    process modules only), its coupling places and, for the supply only, its supply slots."""

    name: str
    kind: StationKind
    pose: Pose
    processes: tuple[Process, ...] = ()
    coupling_places: tuple[CouplingPlace, ...] = ()
    slots: tuple[SupplySlot, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "pose", _pose(self.pose, f"station {self.name}"))
        if self.slots and self.kind is not StationKind.SUPPLY:
            raise ValueError(f"station {self.name} is a {self.kind.value}; only the supply has supply slots")
        if self.processes and self.kind not in _OFFER_PROCESSES:
            raise ValueError(
                f"station {self.name} is a {self.kind.value}; only machines and process modules offer processes"
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "pose": list(self.pose),
            "processes": [p.value for p in self.processes],
            "coupling_places": [c.to_dict() for c in self.coupling_places],
            "slots": [s.to_dict() for s in self.slots],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Station:
        return cls(
            d["name"],
            StationKind(d["kind"]),
            d["pose"],
            tuple(Process(p) for p in d.get("processes", ())),
            tuple(CouplingPlace.from_dict(c) for c in d.get("coupling_places", ())),
            tuple(SupplySlot.from_dict(s) for s in d.get("slots", ())),
        )


@dataclass(frozen=True)
class Pallet:
    """A fixture with a clamp that a robot carries between stations; `interfaces` are those it carries."""

    name: str
    interfaces: tuple[Interface, ...]

    def to_dict(self) -> dict:
        return {"name": self.name, "interfaces": [i.value for i in self.interfaces]}

    @classmethod
    def from_dict(cls, d: dict) -> Pallet:
        return cls(d["name"], tuple(Interface(i) for i in d["interfaces"]))


@dataclass(frozen=True)
class Tooling:
    """An exchangeable insert of a gripper, the clamping desk or a process module (jaws, press tools);
    `interfaces` are those it carries. Not a tool: a tool is mounted on a robot."""

    name: str
    interfaces: tuple[Interface, ...]

    def to_dict(self) -> dict:
        return {"name": self.name, "interfaces": [i.value for i in self.interfaces]}

    @classmethod
    def from_dict(cls, d: dict) -> Tooling:
        return cls(d["name"], tuple(Interface(i) for i in d["interfaces"]))


@dataclass(frozen=True)
class Cell:
    """The robot workcell as set up: robots, tools, stations, pallets and tooling. It holds no times: those
    are in the cell's travel-time table, next to the cell file (ADR-0012)."""

    robots: tuple[Robot, ...]
    tools: tuple[Tool, ...] = ()
    stations: tuple[Station, ...] = ()
    pallets: tuple[Pallet, ...] = ()
    tooling: tuple[Tooling, ...] = ()

    def station(self, name: str) -> Station | None:
        return next((s for s in self.stations if s.name == name), None)

    def to_dict(self) -> dict:
        return {
            "robots": [r.to_dict() for r in self.robots],
            "tools": [t.to_dict() for t in self.tools],
            "stations": [s.to_dict() for s in self.stations],
            "pallets": [p.to_dict() for p in self.pallets],
            "tooling": [t.to_dict() for t in self.tooling],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Cell:
        return cls(
            tuple(Robot.from_dict(r) for r in d["robots"]),
            tuple(Tool.from_dict(t) for t in d["tools"]),
            tuple(Station.from_dict(s) for s in d["stations"]),
            tuple(Pallet.from_dict(p) for p in d.get("pallets", ())),
            tuple(Tooling.from_dict(t) for t in d.get("tooling", ())),
        )
