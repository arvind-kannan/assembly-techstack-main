"""The generic operation: one action on an assembly with the robots it needs, one share per robot."""
from __future__ import annotations

from dataclasses import dataclass, field

from diss_assembly_dataclasses.cell_state import Location
from diss_assembly_dataclasses.process import Process

Pose = tuple[float, ...]


@dataclass(frozen=True)
class Share:
    """One robot's part of an operation: tool, path (tool poses relative to the parts),
    path duration and skill parameters."""

    robot: str
    tool: str | None
    path: tuple[Pose, ...] = ()
    path_duration: float = 0.0
    skill_parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "robot": self.robot,
            "tool": self.tool,
            "path": [list(p) for p in self.path],
            "path_duration": self.path_duration,
            "skill_parameters": dict(self.skill_parameters),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Share:
        return cls(
            d["robot"],
            d["tool"],
            tuple(tuple(float(x) for x in p) for p in d["path"]),
            float(d["path_duration"]),
            dict(d["skill_parameters"]),
        )


@dataclass(frozen=True)
class StepRef:
    """The sequence-graph step an operation carries out (all or part of): the product state
    before the step (by product state key), its target part and its process."""

    before: tuple[str, ...]
    target: str
    process: Process

    def to_dict(self) -> dict:
        return {
            "before": list(self.before),
            "target": self.target,
            "process": self.process.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StepRef:
        return cls(tuple(d["before"]), d["target"], Process(d["process"]))


@dataclass(frozen=True)
class Operation:
    """Process, target, one share per robot, direction and the product state it is expected
    to reach (by product state key). An inspection is an operation over all robots.
    An operation that carries out a step of the sequence graph names it (`step`), so its
    feedback leads back to the step; an inspection names none."""

    id: str
    process: Process
    target: str | None
    shares: tuple[Share, ...]
    direction: tuple[float, ...] | None = None
    expected_product_state: tuple[str, ...] = ()
    parameters: dict = field(default_factory=dict)
    step: StepRef | None = None

    @property
    def robots(self) -> tuple[str, ...]:
        return tuple(s.robot for s in self.shares)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "process": self.process.value,
            "target": self.target,
            "shares": [s.to_dict() for s in self.shares],
            "direction": None if self.direction is None else list(self.direction),
            "expected_product_state": list(self.expected_product_state),
            "parameters": dict(self.parameters),
        }
        # Left out when unset, so operations without a step keep their dict.
        if self.step is not None:
            d["step"] = self.step.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Operation:
        step = d.get("step")
        return cls(
            d["id"],
            Process(d["process"]),
            d["target"],
            tuple(Share.from_dict(s) for s in d["shares"]),
            None if d["direction"] is None else tuple(float(x) for x in d["direction"]),
            tuple(d["expected_product_state"]),
            dict(d["parameters"]),
            None if step is None else StepRef.from_dict(step),
        )


@dataclass(frozen=True)
class MachineOperationParameters:
    """A machine operation (rework milling): the machine station carries out its cycle on the part on the pallet
    on one of its coupling places. Without a `cycle_time` (seconds) the estimate is the skill's default."""

    station: str
    cycle_time: float | None = None

    def to_dict(self) -> dict:
        return {"station": self.station, "cycle_time": self.cycle_time}

    @classmethod
    def from_dict(cls, d: dict) -> MachineOperationParameters:
        cycle_time = d.get("cycle_time")
        return cls(d["station"], None if cycle_time is None else float(cycle_time))


@dataclass(frozen=True)
class PalletTransferParameters:
    """Loading (a robot sets the pallet onto the coupling place of the station) or unloading (takes it from
    there)."""

    pallet: str
    station: str
    coupling_place: str

    @property
    def location(self) -> Location:
        return Location.at(self.station, self.coupling_place)

    def to_dict(self) -> dict:
        return {"pallet": self.pallet, "station": self.station, "coupling_place": self.coupling_place}

    @classmethod
    def from_dict(cls, d: dict) -> PalletTransferParameters:
        return cls(d["pallet"], d["station"], d["coupling_place"])


@dataclass(frozen=True)
class ReconfiguringParameters:
    """Reconfiguring: a robot moves the tooling from one location to another (see `Location`: a coupling place,
    a process module's seat included, a robot, or a tool such as a gripper)."""

    tooling: str
    source: Location
    destination: Location

    def to_dict(self) -> dict:
        return {"tooling": self.tooling, "source": self.source.to_dict(), "destination": self.destination.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> ReconfiguringParameters:
        return cls(d["tooling"], Location.from_dict(d["source"]), Location.from_dict(d["destination"]))


ProcessParameters = MachineOperationParameters | PalletTransferParameters | ReconfiguringParameters

PARAMETER_TYPES: dict[Process, type[ProcessParameters]] = {
    Process.REWORK_MILLING: MachineOperationParameters,
    Process.LOADING: PalletTransferParameters,
    Process.UNLOADING: PalletTransferParameters,
    Process.RECONFIGURING: ReconfiguringParameters,
}


def parameters_of(op: Operation) -> ProcessParameters | None:
    """The typed parameters of `op` (from `op.parameters`) when its process has a parameter type, else None."""
    kind = PARAMETER_TYPES.get(op.process)
    return None if kind is None else kind.from_dict(op.parameters)
