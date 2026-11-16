"""`planner_worker`: the cell simulation's motion planner for planner workers (techstack #212, ADR-0011 of the
techstack).

It plans nothing itself. For each operation it builds the scene description from the cell-assembly snapshot
(obstacles, held parts, each robot's start joint configuration), sends it to the planner worker named by its
address, and hands the worker's NC commands to the cell simulation (wire format: `planner_worker.py`). It asks the
worker as well whether two robots' segments conflict, since only the worker has the geometry.

Never a silent fallback (ADR-0008 of the techstack): a worker that cannot be reached or does not answer in time
makes the operation infeasible (`no_path_share`), and the simulation stops there and produces no NC. A worker that
answers with an error, or with an answer the client cannot read, is a bug, not an infeasible operation: it raises,
and the sim server reports `internal_error`.

Options (`configure_planner`): `address` (`"host:port"`; required unless the planner was registered with one),
`timeout` (s, for connecting and for each answer), `obstacles` (the cell's static obstacles, a list of dicts, passed
on unchanged) and `reference_poses` (part name -> `x, y, z, rx, ry, rz`, where an operation's reference part is).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import planner_worker
from cell_simulation import MotionFailure, MotionPlanner, ReasonCode, Segment, ShareMotion
from sim_server_protocol import InvalidRequest

NAME = "planner_worker"
REFERENCE_PART = "reference_part"  # operation parameter: the part its poses are relative to
DEFAULTS = {"address": None, "timeout": 120.0, "obstacles": [], "reference_poses": {}}


@dataclass
class WorkerMotion(ShareMotion):
    """A motion the worker planned, with the data it wants back to check it for conflicts."""

    conflict_data: object = field(default=None)


def register(service, name: str = NAME, address: str | None = None) -> None:
    """Register a planner worker client as `name` on a `CellSimulationService`; `address` (`"host:port"`) is the
    default for its `address` option."""
    service.register_planner(name, lambda options: PlannerWorkerClient(name, {"address": address, **options}))


class PlannerWorkerClient(MotionPlanner):
    def __init__(self, name: str, options: dict):
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise InvalidRequest(f"unknown {name} options {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        self.name = name
        self.options = {**DEFAULTS, **options}
        self.address = _check_options(name, self.options)

    # -- the MotionPlanner interface -------------------------------------------------------------------------

    def plan_operation(self, robots, operation, starts, snapshot):
        moving = next(s["robot"] for s in operation["shares"] if s["path"])
        try:
            answer = self._ask("plan_operation", scene=self.scene(robots, operation, starts, snapshot))
        except OSError as error:
            return MotionFailure(ReasonCode.NO_PATH_SHARE,
                                 f"planner worker {self.name} at {self._where()} did not answer ({error}); "
                                 f"the simulation never falls back to direct motion", robot=moving)
        infeasible = answer.get("infeasible")
        if infeasible is not None:
            return self._failure(infeasible)
        motions = answer.get("motions")
        if not isinstance(motions, dict):
            raise self._bad("plan_operation", "neither motions nor infeasible")
        return {robot: (self._motion(pair, "transit", robot), self._motion(pair, "share", robot))
                for robot, pair in motions.items()}

    def conflicts(self, first: Segment, second: Segment) -> bool:
        """Asked of the worker; a segment it did not plan (a dwell) does not move, so it is not tested."""
        if not isinstance(first.motion, WorkerMotion) or not isinstance(second.motion, WorkerMotion):
            return False
        try:
            answer = self._ask("conflicts", first=_segment(first), second=_segment(second))
        except OSError as error:  # a conflict test cannot answer "infeasible"; never guess one way or the other
            raise RuntimeError(f"planner worker {self.name} at {self._where()} did not answer a conflict test "
                               f"({error})") from error
        if not isinstance(answer.get("conflict"), bool):
            raise self._bad("conflicts", "no conflict flag")
        return answer["conflict"]

    # -- the scene description -------------------------------------------------------------------------------

    def scene(self, robots: dict, operation: dict, starts: dict, snapshot: dict) -> dict:
        """What the worker gets for `operation` (see `planner_worker`)."""
        part = operation["parameters"].get(REFERENCE_PART)
        pose = None
        if part is not None:
            pose = self.options["reference_poses"].get(part)
            if pose is None:
                raise InvalidRequest(f"{operation['id']} is relative to {part!r}, which {self.name} has no reference "
                                     f"pose for; send it in the reference_poses option of configure_planner")
        holds, mounted = snapshot["holds"], snapshot["mounted_tools"]
        scene_robots = {}
        for robot, start in starts.items():
            entry = {"start_pose": _plain(start["pose"]), "start_configuration": _plain(start["configuration"]),
                     "mounted_tool": mounted.get(robot), "holds": holds.get(robot), "in_operation": robot in robots}
            if robot in robots:
                entry.update({key: robots[robot][key] for key in ("nc_channel", "tools", "capabilities")})
            scene_robots[robot] = entry
        return {
            "operation": _plain_operation(operation),
            "robots": scene_robots,
            "product": {"assembly_xml": snapshot["assembly_xml"], "product_model_xml": snapshot["product_model_xml"],
                        "reference_part": part, "reference_pose": None if pose is None else list(pose)},
            "held_parts": dict(holds),
            "obstacles": list(self.options["obstacles"]),
        }

    # -- the worker's answers --------------------------------------------------------------------------------

    def _ask(self, request: str, **fields) -> dict:
        """The worker's answer to `request`; raises `OSError` when it does not answer, `RuntimeError` when it fails."""
        answer = planner_worker.call(self.address, {"request": request, "protocol": planner_worker.PROTOCOL, **fields},
                                     self.options["timeout"])
        if not isinstance(answer, dict):
            raise self._bad(request, f"a {type(answer).__name__}, not a dict")
        if answer.get("ok") is not True:
            raise RuntimeError(f"planner worker {self.name} at {self._where()} failed on {request}: "
                               f"{answer.get('error')}: {answer.get('message')}")
        return answer

    def _failure(self, infeasible) -> MotionFailure:
        if not isinstance(infeasible, dict) or infeasible.get("during") not in ("transit", "share") \
                or not isinstance(infeasible.get("robot"), str):
            raise self._bad("plan_operation", "an infeasible operation without its robot and during")
        try:
            reason = ReasonCode(infeasible.get("reason_code"))
        except ValueError:
            raise self._bad("plan_operation", f"the unknown reason code {infeasible.get('reason_code')!r}") from None
        return MotionFailure(reason, str(infeasible.get("text", "")), robot=infeasible["robot"],
                             during=infeasible["during"])

    def _motion(self, pair, which: str, robot: str) -> WorkerMotion:
        motion = pair.get(which) if isinstance(pair, dict) else None
        if not isinstance(motion, dict):
            raise self._bad("plan_operation", f"no {which} motion for {robot}")
        duration = motion.get("duration")
        if not _finite(duration) or duration < 0:
            raise self._bad("plan_operation", f"a {which} duration of {robot} that is not a number of at least 0")
        commands = motion.get("commands")
        if not isinstance(commands, list) or not all(isinstance(c, dict) for c in commands):
            raise self._bad("plan_operation", f"{which} commands of {robot} that are not a list of dicts")
        end_pose, end_configuration = motion.get("end_pose"), motion.get("end_configuration")
        return WorkerMotion(float(duration), commands, None if end_pose is None else tuple(end_pose),
                            None if end_configuration is None else tuple(end_configuration),
                            motion.get("conflict_data"))

    def _bad(self, request: str, what: str) -> RuntimeError:
        return RuntimeError(f"planner worker {self.name} at {self._where()} answered {request} with {what}")

    def _where(self) -> str:
        return f"{self.address[0]}:{self.address[1]}"


def _segment(segment: Segment) -> dict:
    return {"robot": segment.robot, "id": segment.id, "kind": segment.kind, "start": segment.start,
            "end": segment.end, "motion": segment.motion.conflict_data}


def _finite(value) -> bool:
    """A finite real number (numpy scalars included, booleans excluded)."""
    return not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(value)


def _plain(values):
    return None if values is None else [float(v) for v in values]


def _plain_operation(operation: dict) -> dict:
    """The operation with its numbers as Python floats (the techstack may send numpy scalars, which a worker's
    venv could not unpickle without numpy)."""
    shares = [{**share, "path": [_plain(pose) for pose in share["path"]], "path_duration": float(share["path_duration"])}
              for share in operation["shares"]]
    return {**operation, "shares": shares}


def _check_options(name: str, options: dict) -> tuple[str, int]:
    """Refuse bad options at `configure_planner`, not later in a `simulate`; returns the worker's address."""
    problems = []
    address = options["address"]
    host, _, port = address.rpartition(":") if isinstance(address, str) else ("", "", "")
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        problems.append(f"address must be \"host:port\", got {address!r}")
    timeout = options["timeout"]
    if not _finite(timeout) or timeout <= 0:
        problems.append("timeout must be a positive number")
    if not isinstance(options["obstacles"], list) or not all(isinstance(o, dict) for o in options["obstacles"]):
        problems.append("obstacles must be a list of dicts")
    poses = options["reference_poses"]
    if not isinstance(poses, dict) or not all(
            isinstance(k, str) and isinstance(v, (list, tuple)) and len(v) == 6 and all(_finite(x) for x in v)
            for k, v in poses.items()):
        problems.append("reference_poses maps part names to x, y, z, rx, ry, rz")
    if problems:
        raise InvalidRequest(f"{name} options: " + "; ".join(problems))
    return host, int(port)
