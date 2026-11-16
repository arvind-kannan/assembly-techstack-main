"""Cell simulation for operation plans: `simulate`, `check_feasibility` and `configure_planner` (techstack #110).

Kept free of pybullet so it can be tested on its own, like `sim_server_protocol`. DisassemblySim only
registers its motion planners here and forwards three request types to `handle`.

What it does with an operation plan (ADR-0008 of the techstack, "the cell simulation sits before the cell"):
  * plans all robot motion through the configured `MotionPlanner`: each share's path and the transit
    before it. It never falls back: no planner configured, or a planner that finds nothing, is an
    infeasible operation with a reason code from the closed list, and the simulation stops there;
  * executes virtually: *dry* (default) takes the plan as planned, *physics* (opt-in) needs a registered
    physics executor and is refused (`invalid_request`) when there is none;
  * computes cell time from the generated motion (planner durations, dwells, tool change constants of the
    cell's travel-time table, which the request carries next to the cell), never wall clock;
  * adds motion-level sync points where the planner says two robots' segments conflict, and turns every
    sync point, task-level or motion-level, into a `WAITM` mark in both NC programs;
  * reports the observed snapshot at each inspection reached, the operation outcomes and anomalies, and for
    a cell-targeted run both NC programs up to and including the first inspection, while rehearsing all of
    the plan (a late infeasibility is found before the cell moves).

Requests and reports are plain dicts (the techstack's `to_dict` forms), so they pickle safely over the wire.
"""
from __future__ import annotations

import math
import numbers
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable

from sim_server_protocol import InvalidRequest

SUPPLY_SLOT = "supply_slot"  # operation parameter: gripping at the supply, naming the slot
PLAN_FAILURE = "plan_failure"  # operation parameter: a plan function failed, with its reason
PLAN_FAILURE_ROBOT = "plan_failure_robot"  # operation parameter: the robot whose share failed (absent on older marks)
WAITM_MARKERS = 9  # WAITM marker numbers cycle 1..9; the SYNC ordinal stays universal
INSPECTING, RELEASING, INSERTING, SCREWING_IN, GRIPPING = "inspecting", "releasing", "inserting", "screwing_in", "gripping"


class ReasonCode(str, Enum):
    """The closed list of reason codes (same strings as the techstack's `ReasonCode`)."""

    NO_PATH_SHARE = "no_path_share"
    NO_PATH_TRANSIT = "no_path_transit"
    OUT_OF_REACH = "out_of_reach"
    COLLISION = "collision"
    SYNC_DEADLOCK = "sync_deadlock"
    CELL_STATE_MISMATCH = "cell_state_mismatch"
    UNSUPPORTED_PROCESS = "unsupported_process"
    CONTACT_ABORT = "contact_abort"
    INJECTED = "injected"
    CONTACT_WARNING = "contact_warning"
    NEAR_LIMIT = "near_limit"
    TIME_OVERRUN = "time_overrun"
    INVALID_REQUEST = "invalid_request"
    INTERNAL_ERROR = "internal_error"


INFEASIBLE_CODES = frozenset({
    ReasonCode.NO_PATH_SHARE, ReasonCode.NO_PATH_TRANSIT, ReasonCode.OUT_OF_REACH, ReasonCode.COLLISION,
    ReasonCode.SYNC_DEADLOCK, ReasonCode.CELL_STATE_MISMATCH, ReasonCode.UNSUPPORTED_PROCESS,
})


# --- motion planners ------------------------------------------------------------------------------------------

@dataclass
class ShareMotion:
    """What a planner generated for one share or transit: its cell time and the NC command dicts
    (the format `write_g_code` takes) that carry it out. `end_pose` is where the robot is afterwards,
    `end_configuration` its joint configuration there (None: the planner does not say)."""

    duration: float
    commands: list[dict] = field(default_factory=list)
    end_pose: tuple | None = None
    end_configuration: tuple | None = None


@dataclass
class MotionFailure:
    """A planner found no motion; `reason` is from `INFEASIBLE_CODES`. From `plan_operation` it also names the
    robot and whether its `transit` or its `share` failed."""

    reason: ReasonCode
    text: str
    robot: str | None = None
    during: str = "share"


@dataclass
class Segment:
    """A planned stretch of one robot's motion in the schedule, for robot-vs-robot checks."""

    robot: str
    id: str  # the operation id, or `<operation id>:transit`
    kind: str  # "operation" | "transit"
    motion: ShareMotion
    start: float = 0.0
    end: float = 0.0


class MotionPlanner:
    """What a motion planner (cuMotion, an OMPL RRT, ...) provides to the cell simulation.

    `robot` and `operation` are the request's dicts. A planner that cannot plan returns a `MotionFailure`
    (never direct motion): the simulation reports it as an infeasible operation.
    """

    name = "planner"

    def plan_operation(self, robots: dict[str, dict], operation: dict, starts: dict[str, dict],
                       snapshot: dict) -> dict[str, tuple[ShareMotion, ShareMotion]] | MotionFailure:
        """Plan one operation, all of its shares at once (ADR-0010 of the techstack): for every share with a path,
        `{robot id: (transit motion, share motion)}`, the transit leading to where the share's path starts.

        `robots` are the cell's robot dicts of the operation's shares. `starts` gives each robot of the cell its
        start `{"pose": ..., "configuration": ...}`, both None while it is at home. `snapshot` is the cell-assembly
        snapshot before the operation: `assembly_xml` (the parts present), `product_state`, `holds`,
        `mounted_tools`, and `product_model_xml` (every part of the product, held ones included).

        Default (single-robot planners): each share on its own, by `plan_transit` and `plan_share`."""
        motions = {}
        for share in operation["shares"]:
            if not share["path"]:
                continue
            robot = robots[share["robot"]]
            transit = self.plan_transit(robot, starts[robot["id"]]["pose"], self.path_start(operation, share))
            if isinstance(transit, MotionFailure):
                return replace(transit, robot=robot["id"], during="transit")
            motion = self.plan_share(robot, operation, share)
            if isinstance(motion, MotionFailure):
                return replace(motion, robot=robot["id"], during="share")
            motions[robot["id"]] = (transit, motion)
        return motions

    def plan_share(self, robot: dict, operation: dict, share: dict) -> ShareMotion | MotionFailure:
        raise NotImplementedError

    def path_start(self, operation: dict, share: dict) -> tuple:
        """Where the share's path starts, in the frame transits are planned in: the goal of the transit before
        it. Default: its first pose as given (a planner that places poses relative to a part overrides it)."""
        return tuple(share["path"][0])

    def plan_transit(self, robot: dict, start_pose: tuple | None, goal_pose: tuple) -> ShareMotion | MotionFailure:
        """From `start_pose` (None: the robot's home) to `goal_pose`."""
        raise NotImplementedError

    def conflicts(self, first: Segment, second: Segment) -> bool:
        """Whether two segments of different robots cannot run at the same time (paths too close)."""
        raise NotImplementedError


# --- NC rendering ---------------------------------------------------------------------------------------------

def render_nc(commands: list[dict], channel: int) -> str:
    """The NC text of a command list: the subset of `write_g_code`'s format that planners' commands use
    (cartesian G/X/Y/Z/A/B/C words in metres and radians, joint `RA` words, F, comments, WAITM). The cell
    adapter in DisassemblySim may replace it with the full `write_g_code` export."""
    import math
    lines, mode = [f"; MRTAMP program for NC channel {channel}"], None
    for command in commands:
        if "comment" in command:
            lines.append(f"; ---- {command['comment']}")
        elif "WAITM" in command:
            lines.append(f"; ==== SYNC {command['seq']}: {command.get('why', '')}")
            lines.append(f"WAITM({command['WAITM']},1,2) ; sync {command.get('why', '')}")
        elif "toolchange" in command:
            lines.append(f"; TOOLCHANGE {command.get('from')} -> {command['toolchange']} (cell macro)")
        else:
            joint = any(k.startswith("RA") for k in command)
            if joint and mode != "joint":
                lines.append("TRAFOOF")
                mode = "joint"
            elif not joint and mode != "cart":
                lines.append("TRAORI")
                mode = "cart"
            words = []
            for key in ("G", "X", "Y", "Z", "A", "B", "C", "R", "RA1", "RA2", "RA3", "RA4", "RA5", "RA6", "F"):
                value = command.get(key)
                if value is None:
                    continue
                if key == "G":
                    words.append("G1" if value == 0 else f"G{value}")
                elif key in ("X", "Y", "Z"):
                    words.append(f"{key}{value * 1000:.3f}")
                elif key in ("A", "B", "C", "R"):
                    words.append(f"{key}{math.degrees(value):.3f}")
                elif key.startswith("RA"):
                    words.append(f"{key}={math.degrees(value):.3f}")
                else:
                    words.append(f"F{value:.1f}")
            lines.append(" ".join(words))
    lines.append("M30")
    return "\n".join(lines) + "\n"


# --- the service ----------------------------------------------------------------------------------------------

class CellSimulationService:
    """The state the server keeps between requests: the registered planners and the configured one."""

    def __init__(self, render: Callable[[list[dict], int], str] = render_nc, physics_executor=None):
        self._factories: dict[str, Callable[[dict], MotionPlanner]] = {}
        self.planner: MotionPlanner | None = None
        self.render = render
        self.physics_executor = physics_executor  # callable(plan_dict, segments) -> {operation id: outcome dict}

    def register_planner(self, name: str, factory: Callable[[dict], MotionPlanner]) -> None:
        self._factories[name] = factory

    @property
    def planner_names(self) -> list[str]:
        return sorted(self._factories)

    def configure_planner(self, name: str, options: dict | None = None) -> None:
        """Set the planner once (the techstack does not pass it per request). An unknown or failing planner
        leaves the previous setting in place."""
        if name not in self._factories:
            raise InvalidRequest(f"unknown planner {name!r}; registered: {self.planner_names}")
        self.planner = self._factories[name](dict(options or {}))

    def handle(self, request: dict) -> dict:
        kind = request["request"]
        if kind == "configure_planner":
            self.configure_planner(_text(request, "planner"), _optional(request, "options", dict))
            return {"ok": True, "planner": self.planner.name}
        if kind == "simulate":
            return self.simulate(request)
        if kind == "check_feasibility":
            return self.check_feasibility(request)
        raise InvalidRequest(f"unknown request {kind!r}")

    # -- check_feasibility -----------------------------------------------------------------------------------

    def check_feasibility(self, request: dict) -> dict:
        """Can the concurrent group (at most one operation per robot) be carried out from the snapshot?
        Plans each operation's motion and checks the robots' segments against each other; executes nothing,
        produces no NC and leaves no state behind."""
        group = _list(request, "concurrent_group")
        if not group:
            raise InvalidRequest("a concurrent group is a non-empty list of operations")
        cell, snapshot = _cell(request), _snapshot(request, "snapshot")
        operations = [_operation(op) for op in group]
        if len({op["id"] for op in operations}) != len(operations):
            raise InvalidRequest("operation ids in a concurrent group must be unique")
        robots = [r for op in operations for r in _robots_of(op)]
        if len(robots) != len(set(robots)):
            raise InvalidRequest("a concurrent group has at most one operation per robot")
        state, holds, mounted = set(snapshot["product_state"]), dict(snapshot["holds"]), dict(snapshot["mounted_tools"])
        model_xml = _optional(request, "product_model_xml", str) or snapshot["assembly_xml"]
        _xml_state(model_xml, "product_model_xml")
        durations, segments, handled = {}, [], {}
        for op in operations:
            problem = _check_model(op, state, holds, cell)
            if problem is None and op["target"] is not None and handled.setdefault(op["target"], op["id"]) != op["id"]:
                problem = (ReasonCode.CELL_STATE_MISMATCH, op["shares"][0]["robot"],
                           f"{op['target']} is also handled by {handled[op['target']]} at the same time")
            if problem is not None:
                return _feasibility(False, op["id"], *problem, durations)
            motions = self._motions(op, cell, {}, _planner_snapshot(snapshot["assembly_xml"], model_xml, state, holds,
                                                                    mounted))
            if isinstance(motions, _Infeasible):
                return _feasibility(False, op["id"], motions.reason, motions.robot, motions.text, durations)
            clocks = {share["robot"]: _tool_change_time(cell, mounted.get(share["robot"]), share["tool"])
                      for share in op["shares"]}
            for segment in motions:  # the group starts together: each robot's segments follow its tool change
                segment.start = clocks[segment.robot]
                segment.end = clocks[segment.robot] = segment.start + segment.motion.duration
            for segment in motions:
                for other in segments:
                    if other.robot != segment.robot and _overlap(other, segment) and self._conflict(other, segment):
                        return _feasibility(False, op["id"], ReasonCode.COLLISION, segment.robot,
                                            f"{segment.id} of {segment.robot} conflicts with {other.id} of {other.robot}",
                                            durations)
                segments.append(segment)
            durations[op["id"]] = max(_duration_of(op, motions, cell, mounted, share["robot"]) for share in op["shares"])
        return _feasibility(True, None, None, None, "", durations)

    # -- simulate --------------------------------------------------------------------------------------------

    def simulate(self, request: dict) -> dict:
        plan, cell, snapshot = _plan(request), _cell(request), _snapshot(request, "start_snapshot")
        target, mode = _text(request, "target"), _text(request, "execution_mode")
        if target not in ("simulation", "cell"):
            raise InvalidRequest(f"unknown target {target!r}")
        if mode not in ("dry", "physics"):
            raise InvalidRequest(f"unknown execution mode {mode!r}")
        if mode == "physics" and self.physics_executor is None:
            raise InvalidRequest("physics execution is not available in this simulation")
        model_xml = _optional(request, "product_model_xml", str) or snapshot["assembly_xml"]
        _xml_state(model_xml, "product_model_xml")

        run = _Run(self, plan, cell, snapshot, model_xml)
        infeasible = run.execute()
        if mode == "physics" and infeasible is None:
            run.apply_physics(self.physics_executor)
        return run.report(infeasible, target)

    # -- planning one operation ------------------------------------------------------------------------------

    def _conflict(self, first: Segment, second: Segment) -> bool:
        """Without a planner there is no motion, so nothing to conflict (paths need a planner anyway)."""
        return self.planner is not None and bool(self.planner.conflicts(first, second))

    def _motions(self, op: dict, cell: dict, starts: dict, snapshot: dict):
        """The transit and share motion of every share of `op`, as segments in order per robot, or the
        `_Infeasible` that stops the simulation. `starts` maps a robot that has moved to its start
        `{"pose", "configuration"}`; `snapshot` is the one `MotionPlanner.plan_operation` gets."""
        robots = {r["id"]: r for r in cell["robots"]}
        moving = [share["robot"] for share in op["shares"] if share["path"]]
        motions = {}
        if moving:
            if self.planner is None:
                return _Infeasible(ReasonCode.NO_PATH_SHARE, moving[0],
                                   f"no motion planner configured for {op['id']}; send configure_planner "
                                   f"(registered: {self.planner_names}); the simulation never falls back to direct motion")
            home = {"pose": None, "configuration": None}
            motions = self.planner.plan_operation({r: robots[r] for r in _robots_of(op)}, op,
                                                  {r: dict(starts.get(r, home)) for r in robots}, snapshot)
            if isinstance(motions, MotionFailure):
                where = f"transit to {op['id']}" if motions.during == "transit" else op["id"]
                return _Infeasible(_infeasible(motions.reason), motions.robot, f"{where}: {motions.text}")
            missing = [r for r in moving if not isinstance(motions.get(r), tuple) or len(motions[r]) != 2]
            if missing:
                raise RuntimeError(f"planner {self.planner.name} gave no transit and share motion of {op['id']} "
                                   f"for {', '.join(missing)}")
        segments: list[Segment] = []
        for share in op["shares"]:
            robot = share["robot"]
            if not share["path"]:
                segments.append(Segment(robot, op["id"], "operation", ShareMotion(
                    _dwell(op, share, cell), [{"comment": _describe(op)}])))
                continue
            transit, motion = motions[robot]
            segments.append(Segment(robot, f"{op['id']}:transit", "transit", transit))
            segments.append(Segment(robot, op["id"], "operation", motion))
        return segments


@dataclass
class _Infeasible:
    reason: ReasonCode
    robot: str | None
    text: str


def _infeasible(code) -> ReasonCode:
    """A planner's reason as one of the infeasible codes; anything else is a bug in the planner."""
    code = ReasonCode(code)
    if code not in INFEASIBLE_CODES:
        raise RuntimeError(f"planner reported {code.value!r}, which is not an infeasible-operation reason")
    return code


class _Run:
    """One `simulate`: walks the plan in order, scheduling segments per robot."""

    def __init__(self, service: CellSimulationService, plan: dict, cell: dict, snapshot: dict, model_xml: str):
        self.service, self.plan, self.cell, self.model_xml = service, plan, cell, model_xml
        self.robots = [r["id"] for r in cell["robots"]]
        self.channel = {r["id"]: r["nc_channel"] for r in cell["robots"]}
        self.start_state = tuple(snapshot["product_state"])
        self.state, self.holds = set(snapshot["product_state"]), dict(snapshot["holds"])
        self.mounted = dict(snapshot["mounted_tools"])
        self.clock = {r: 0.0 for r in self.robots}
        self.busy = {r: 0.0 for r in self.robots}
        self.starts: dict[str, dict] = {}  # robot -> {"pose", "configuration"} where its last motion ended
        self.commands = {r: [] for r in self.robots}
        self.cut: dict[str, dict] = {}  # robot -> the command that ends its NC program (first inspection)
        self.ends: dict[str, float] = {}  # operation or transit id -> end time
        self.last_command: dict[tuple, dict] = {}  # (segment id, robot) -> its last command, to insert WAITM after it
        self.segments: list[Segment] = []
        self.timeline: list[dict] = []
        self.sync_added: list[dict] = []
        self.sync_ordinal = 0
        self.tool_changes = 0
        self.feasible: list[str] = []
        self.outcomes: list[dict] = []
        self.observed: dict[str, dict] = {}
        self.inspections = 0
        self.holds_after: dict[str, dict] = {}
        self.mark_robots: dict[int, list[str]] = {}  # WAITM ordinal -> the two robots it joins

    def execute(self) -> dict | None:
        waits: dict[tuple, list[dict]] = {}
        for sync in self.plan["sync_points"]:
            waits.setdefault((sync["robot"], sync["before"]), []).append(sync)
        for op in self.plan["operations"]:
            problem = _check_model(op, self.state, self.holds, self.cell)
            if problem is not None:
                return {"operation_id": op["id"], "robot": problem[1], "reason_code": problem[0].value, "text": problem[2]}
            snapshot = _planner_snapshot(_assembly_xml_with(self.model_xml, tuple(sorted(self.state))), self.model_xml,
                                         self.state, self.holds, self.mounted)
            motions = self.service._motions(op, self.cell, self.starts, snapshot)
            if isinstance(motions, _Infeasible):
                return {"operation_id": op["id"], "robot": motions.robot, "reason_code": motions.reason.value,
                        "text": motions.text}
            self._schedule(op, motions, waits)
            self.feasible.append(op["id"])
            if op["process"] == INSPECTING:
                self.inspections += 1
                self.observed[op["id"]] = self._snapshot()
            else:
                self.state = set(op["expected_product_state"])
                _update_holds(op, self.holds)
                self.outcomes.append({"kind": "operation_outcome", "operation_id": op["id"], "succeeded": True,
                                      "observed_product_state": sorted(op["expected_product_state"]),
                                      "reason_code": None, "text": ""})
            self.holds_after[op["id"]] = dict(self.holds)
        return None

    def apply_physics(self, executor) -> None:
        """Physics mode: the executor reports which operations failed on contact; the product state stays
        where the first failure left it until the next inspection, which observes it."""
        results = executor(self.plan, self.segments)
        outcomes, state, failed = [], set(self.start_state), False
        for op in self.plan["operations"]:
            if op["id"] not in self.feasible:
                break
            if op["process"] == INSPECTING:
                self.observed[op["id"]] = self._snapshot(state, self.holds_after.get(op["id"], {}))
                if failed:
                    break
                continue
            if failed:
                continue
            result = results.get(op["id"], {"succeeded": True})
            if result["succeeded"]:
                state = set(op["expected_product_state"])
                outcomes.append(_outcome(op, True, state))
            else:
                failed = True
                outcomes.append(_outcome(op, False, state, ReasonCode.CONTACT_ABORT, result.get("text", "")))
        self.outcomes = outcomes

    # -- scheduling ------------------------------------------------------------------------------------------

    def _schedule(self, op: dict, motions: list[Segment], waits: dict) -> None:
        by_robot: dict[str, list[Segment]] = {}
        for segment in motions:
            by_robot.setdefault(segment.robot, []).append(segment)
        ready = {}
        for share in op["shares"]:
            robot = share["robot"]
            t = self.clock[robot]
            tool = share["tool"]
            if tool is not None and tool != self.mounted.get(robot):
                dt = _tool_change_time(self.cell, self.mounted.get(robot), tool)
                self.timeline.append({"kind": "tool_change", "robot": robot, "tool": tool, "start": t, "end": t + dt})
                self.commands[robot].append({"toolchange": tool, "from": self.mounted.get(robot)})
                self.busy[robot] += dt
                self.mounted[robot] = tool
                self.tool_changes += 1
                t += dt
            for sync in waits.get((robot, op["id"]), ()):
                t = max(t, self._task_level_wait(sync, robot, op))
            ready[robot] = t
        start_of_op = max(ready.values())
        for share in op["shares"]:
            robot = share["robot"]
            t = start_of_op
            for segment in by_robot[robot]:
                t = self._place(segment, t, op)
            self.clock[robot] = t
            if share["path"]:
                last = self.segments[-1].motion
                self.starts[robot] = {"pose": last.end_pose or tuple(share["path"][-1]),
                                      "configuration": last.end_configuration}
        self.ends[op["id"]] = max(self.clock[s["robot"]] for s in op["shares"])
        if op["process"] == INSPECTING and not self.cut:
            for robot in self.robots:  # every robot's program ends here, whether it took part or not
                marker = {"comment": f"{op['id']}: end of segment (inspection)"}
                self.commands[robot].append(marker)
                self.cut[robot] = marker

    def _place(self, segment: Segment, t: float, op: dict) -> float:
        """Put a segment at time `t` or later: a conflict with another robot's overlapping segment makes
        the robot wait (a motion-level sync point) until that segment has ended."""
        planner = self.service.planner
        while True:
            segment.start, segment.end = t, t + segment.motion.duration
            blocker = next((o for o in self.segments if o.robot != segment.robot and _overlap(o, segment)
                            and planner is not None and planner.conflicts(o, segment)), None)
            if blocker is None:
                break
            self._sync(segment.robot, segment.id, blocker, "motion", "paths_too_close")
            t = blocker.end
        for command in segment.motion.commands:
            self.commands[segment.robot].append(command)
        if segment.motion.commands:
            self.last_command[(segment.id, segment.robot)] = segment.motion.commands[-1]
        self.timeline.append({"kind": segment.kind, "operation_id": segment.id, "robot": segment.robot,
                              "start": segment.start, "end": segment.end})
        self.busy[segment.robot] += segment.motion.duration
        self.segments.append(segment)
        self.ends[segment.id] = segment.end
        return segment.end

    def _task_level_wait(self, sync: dict, robot: str, op: dict) -> float:
        blocker = next((s for s in self.segments if s.id == sync["until"] and s.robot != robot), None)
        if blocker is not None:
            self._mark(robot, op["id"], blocker, sync["reason"])
        return self.ends[sync["until"]]

    def _sync(self, robot: str, before: str, blocker: Segment, layer: str, reason: str) -> None:
        self.sync_added.append({"robot": robot, "before": before, "until": blocker.id, "layer": layer, "reason": reason})
        self._mark(robot, before, blocker, reason)

    def _mark(self, robot: str, before: str, blocker: Segment, reason: str) -> None:
        """A `WAITM` mark in both programs: after the segment waited for, and where `robot` waits."""
        self.sync_ordinal += 1
        why = f"{robot} waits before {before} for {blocker.id} ({reason})"

        def marker():
            return {"WAITM": 1 + (self.sync_ordinal - 1) % WAITM_MARKERS, "why": why, "seq": self.sync_ordinal}

        commands = self.commands[blocker.robot]
        after = self.last_command.get((blocker.id, blocker.robot))
        at = next((i for i, c in enumerate(commands) if c is after), len(commands) - 1)
        commands.insert(at + 1, marker())
        self.commands[robot].append(marker())
        self.mark_robots[self.sync_ordinal] = [robot, blocker.robot]

    # -- report ----------------------------------------------------------------------------------------------

    def _snapshot(self, state=None, holds=None) -> dict:
        state = self.state if state is None else state
        return {
            "assembly_xml": _assembly_xml_with(self.model_xml, tuple(sorted(state))),
            "cell_state": {"holds": dict(self.holds if holds is None else holds), "mounted_tools": dict(self.mounted)},
        }

    def report(self, infeasible: dict | None, target: str) -> dict:
        return {
            "ok": True,
            "feasible_operations": list(self.feasible),
            "infeasible": None if infeasible is None else {"kind": "infeasible_operation", **infeasible},
            "metrics": {"makespan": max(self.clock.values(), default=0.0), "busy_time": dict(self.busy),
                        "tool_changes": self.tool_changes, "inspections": self.inspections},
            "timeline": self.timeline,
            "sync_points_added": self.sync_added,
            "observed_snapshots": self.observed,
            "outcomes": self.outcomes,
            "anomalies": [],
            "nc_programs": self._nc_programs() if target == "cell" and infeasible is None else None,
            "rejection": None,
        }

    def _nc_programs(self) -> dict:
        """Both programs, each ended at the first inspection; a WAITM mark whose partner lies beyond the end
        of the other program is dropped (kept as a comment), so every mark in the programs is paired."""
        cut = {}
        for robot in self.robots:
            commands, marker = self.commands[robot], self.cut.get(robot)
            cut[robot] = commands if marker is None else commands[:next(i for i, c in enumerate(commands) if c is marker) + 1]
        present = {robot: {c["seq"] for c in commands if "WAITM" in c} for robot, commands in cut.items()}
        programs = {}
        for robot, commands in cut.items():
            paired = [c if "WAITM" not in c or all(c["seq"] in present[r] for r in self.robots if r in self._partners(c))
                      else {"comment": f"sync {c['seq']} falls beyond the end of this segment: {c['why']}"}
                      for c in commands]
            channel = self.channel[robot]
            programs[str(channel)] = {"file_name": f"MRTAMP_CH{channel}.MPF", "text": self.service.render(paired, channel)}
        return programs

    def _partners(self, mark: dict) -> list[str]:
        """The robots a WAITM mark joins (the one waiting and the one waited for)."""
        return self.mark_robots[mark["seq"]]


# --- model-level checks (what the model can say without motion) ----------------------------------------------

def _check_model(op: dict, state: set, holds: dict, cell: dict):
    """`(reason, robot, text)` for an operation the model already rules out, or None."""
    robots = {r["id"]: r for r in cell["robots"]}
    shares = op["shares"]
    unknown = next((s["robot"] for s in shares if s["robot"] not in robots), None)
    if unknown is not None:
        return ReasonCode.UNSUPPORTED_PROCESS, unknown, f"{unknown} is not a robot of the cell"
    if op["process"] == INSPECTING:
        if not any(INSPECTING in robots[s["robot"]]["capabilities"] for s in shares):
            return ReasonCode.UNSUPPORTED_PROCESS, None, "no robot of the inspection can inspect"
        return None
    if PLAN_FAILURE in op["parameters"]:
        robot = op["parameters"].get(PLAN_FAILURE_ROBOT, shares[0]["robot"])
        return ReasonCode.NO_PATH_SHARE, robot, str(op["parameters"][PLAN_FAILURE])
    for share in shares:
        robot = robots.get(share["robot"])
        if robot is None or op["process"] not in robot["capabilities"] or (
                share["tool"] is not None and share["tool"] not in robot["tools"]):
            return ReasonCode.UNSUPPORTED_PROCESS, share["robot"], f"{share['robot']} cannot carry out {op['process']} with {share['tool']}"
    robot, target = shares[0]["robot"], op["target"]
    mismatch = None
    if _is_gripping_at_supply(op) or op["process"] == SCREWING_IN:
        if target in state:
            mismatch = f"{target} is already in the assembly"
    elif op["process"] in (INSERTING, RELEASING):
        if holds.get(robot) != target:
            mismatch = f"{robot} does not hold {target}"
        elif op["process"] == INSERTING and target in state:
            mismatch = f"{target} is already in the assembly"
    elif target is not None and target not in state:
        mismatch = f"{target} is not in the assembly"
    return None if mismatch is None else (ReasonCode.CELL_STATE_MISMATCH, robot, mismatch)


def _planner_snapshot(assembly_xml: str, product_model_xml: str, state, holds: dict, mounted: dict) -> dict:
    """The cell-assembly snapshot a motion planner gets (a copy: the planner cannot change the run's)."""
    return {"assembly_xml": assembly_xml, "product_model_xml": product_model_xml, "product_state": sorted(state),
            "holds": dict(holds), "mounted_tools": dict(mounted)}


def _is_gripping_at_supply(op: dict) -> bool:
    return op["process"] == GRIPPING and SUPPLY_SLOT in op["parameters"]


def _update_holds(op: dict, holds: dict) -> None:
    if _is_gripping_at_supply(op):
        holds[op["shares"][0]["robot"]] = op["target"]
    elif op["process"] == RELEASING:
        holds.pop(op["shares"][0]["robot"], None)


def _assembly_xml_with(assembly_xml: str, product_state: tuple) -> str:
    """The assembly XML reduced to the parts of `product_state` and the connections between them."""
    root = ET.fromstring(assembly_xml)
    keep = set(product_state)
    for element in list(root):
        names = {element.attrib.get("name"), element.attrib.get("part_name1"), element.attrib.get("part_name2")} - {None}
        if not names <= keep:
            root.remove(element)
    return ET.tostring(root, encoding="unicode")


# --- times ---------------------------------------------------------------------------------------------------

def _tool_change_time(cell: dict, before: str | None, after: str | None) -> float:
    if after is None or after == before:
        return 0.0
    times = cell["times"]
    return float(times.get("toolchange_time_assembly_to_storage", 0.0)) \
        + float(times.get("toolchange_time_at_toolstorage", 0.0)) \
        + float(times.get("toolchange_time_storage_to_assembly", 0.0))


def _dwell(op: dict, share: dict, cell: dict) -> float:
    """The time of a share with no path: the cell's dwell for the process, else the share's declared duration.
    The inspection's dwell is `inspection_time`, the key the techstack's planners write."""
    key = "inspection_time" if op["process"] == INSPECTING else f"{op['process']}_time"
    return float(cell["times"].get(key, share["path_duration"]))


def _duration_of(op: dict, motions: list[Segment], cell: dict, mounted: dict, robot: str) -> float:
    share = next(s for s in op["shares"] if s["robot"] == robot)
    change = _tool_change_time(cell, mounted.get(robot), share["tool"]) if share["tool"] is not None else 0.0
    return change + sum(s.motion.duration for s in motions if s.robot == robot)


def _overlap(a: Segment, b: Segment) -> bool:
    return a.start < b.end and b.start < a.end


def _describe(op: dict) -> str:
    return f"{op['id']}: {op['process']} {op['target'] or ''}".rstrip()


def _outcome(op: dict, succeeded: bool, state: set, code: ReasonCode | None = None, text: str = "") -> dict:
    return {"kind": "operation_outcome", "operation_id": op["id"], "succeeded": succeeded,
            "observed_product_state": sorted(state), "reason_code": None if code is None else code.value, "text": text}


def _feasibility(feasible: bool, operation_id, reason, robot, text: str, durations: dict) -> dict:
    return {
        "feasible": feasible,
        "infeasible": None if feasible else {"kind": "infeasible_operation", "operation_id": operation_id,
                                              "robot": robot, "reason_code": reason.value, "text": text},
        "durations": dict(durations),
        "rejection": None,
    }


# --- strict request parsing (plain dicts in, InvalidRequest out) ---------------------------------------------

def _text(d: dict, key: str) -> str:
    value = d.get(key)
    if not isinstance(value, str):
        raise InvalidRequest(f"{key} must be a string")
    return value


def _list(d: dict, key: str) -> list:
    value = d.get(key)
    if not isinstance(value, list):
        raise InvalidRequest(f"{key} must be a list")
    return value


def _optional(d: dict, key: str, kind: type):
    value = d.get(key)
    if value is not None and not isinstance(value, kind):
        raise InvalidRequest(f"{key} must be a {kind.__name__}")
    return value


def _dict(d: dict, key: str) -> dict:
    value = d.get(key)
    if not isinstance(value, dict):
        raise InvalidRequest(f"{key} must be a dict")
    return value


def _number(value, what: str, minimum: float | None = None) -> float:
    """A finite real number (numpy scalars included, booleans excluded)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise InvalidRequest(f"{what} must be a finite number, got {value!r}")
    if minimum is not None and value < minimum:
        raise InvalidRequest(f"{what} must be at least {minimum}, got {value!r}")
    return float(value)


def _names(value, what: str) -> list:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise InvalidRequest(f"{what} must be a list of strings")
    return value


def _xml_state(xml: str, what: str) -> tuple:
    try:
        parts = ET.fromstring(xml).findall("part")
    except ET.ParseError as error:
        raise InvalidRequest(f"{what} is not XML: {error}") from error
    if not all(isinstance(p.attrib.get("name"), str) for p in parts):
        raise InvalidRequest(f"{what} has a part without a name")
    return tuple(sorted(p.attrib["name"] for p in parts))


def _snapshot(d: dict, key: str) -> dict:
    snap = _dict(d, key)
    cell_state = _dict(snap, "cell_state")
    holds, mounted = _dict(cell_state, "holds"), _dict(cell_state, "mounted_tools")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in holds.items()):
        raise InvalidRequest(f"{key}.cell_state.holds maps robot ids to part names")
    if not all(isinstance(k, str) and (v is None or isinstance(v, str)) for k, v in mounted.items()):
        raise InvalidRequest(f"{key}.cell_state.mounted_tools maps robot ids to tool names or None")
    xml = _text(snap, "assembly_xml")
    return {"assembly_xml": xml, "product_state": _xml_state(xml, f"{key}.assembly_xml"),
            "holds": dict(holds), "mounted_tools": dict(mounted)}


def _cell(d: dict) -> dict:
    """The validated robots of the request's cell, with the times of its travel-time table (interim until techstack #235)."""
    cell = _dict(d, "cell")
    robots = _list(cell, "robots")
    if not robots:
        raise InvalidRequest("cell.robots must not be empty")
    for robot in robots:
        if not isinstance(robot, dict) or not isinstance(robot.get("id"), str) or isinstance(robot.get("nc_channel"), bool) \
                or not isinstance(robot.get("nc_channel"), numbers.Integral):
            raise InvalidRequest("every robot needs a string id and an integer nc_channel")
        _names(robot.get("capabilities"), f"robot {robot['id']} capabilities")
        _names(robot.get("tools"), f"robot {robot['id']} tools")
    ids = [r["id"] for r in robots]
    if len(ids) != len(set(ids)):
        raise InvalidRequest("robot ids must be unique")
    times = _dict(_dict(d, "travel_times"), "times")
    for name, value in times.items():
        _number(value, f"travel_times.times[{name!r}]", 0.0)
    return {"robots": [{**r, "nc_channel": int(r["nc_channel"])} for r in robots], "times": dict(times)}


def _operation(op) -> dict:
    """A validated copy of an operation dict (the caller's data is never changed)."""
    if not isinstance(op, dict):
        raise InvalidRequest("an operation is a dict")
    name = _text(op, "id")
    _text(op, "process")
    target = _optional(op, "target", str)
    shares = _list(op, "shares")
    if not shares:
        raise InvalidRequest(f"operation {name} has no shares")
    checked = []
    for share in shares:
        if not isinstance(share, dict) or not isinstance(share.get("robot"), str):
            raise InvalidRequest(f"operation {name} has a malformed share")
        tool = share.get("tool")
        if tool is not None and not isinstance(tool, str):
            raise InvalidRequest(f"operation {name}: a share's tool is a string or None")
        path = share.get("path")
        if not isinstance(path, list):
            raise InvalidRequest(f"operation {name}: a share's path is a list of poses")
        for pose in path:
            if not isinstance(pose, (list, tuple)) or not pose or len(pose) != len(path[0]):
                raise InvalidRequest(f"operation {name}: the poses of a path are non-empty lists of one length")
            for value in pose:
                _number(value, f"operation {name}: a pose value")
        checked.append({**share, "tool": tool, "path": [list(p) for p in path],
                        "path_duration": _number(share.get("path_duration"), f"operation {name}: path_duration", 0.0)})
    robots = [s["robot"] for s in checked]
    if len(robots) != len(set(robots)):
        raise InvalidRequest(f"operation {name} has two shares for one robot")
    _names(op.get("expected_product_state"), f"operation {name} expected_product_state")
    if not isinstance(op.get("parameters"), dict):
        raise InvalidRequest(f"operation {name} needs a parameters dict")
    return {**op, "target": target, "shares": checked, "parameters": dict(op["parameters"])}


def _robots_of(op: dict) -> list[str]:
    return [s["robot"] for s in op["shares"]]


def _plan(request: dict) -> dict:
    plan = _dict(request, "operation_plan")
    operations = [_operation(op) for op in _list(plan, "operations")]
    ids = [op["id"] for op in operations]
    if len(ids) != len(set(ids)):
        raise InvalidRequest("operation ids must be unique")
    sync_points = _list(plan, "sync_points")
    for sync in sync_points:
        if not isinstance(sync, dict) or not all(isinstance(sync.get(k), str) for k in ("robot", "before", "until", "layer", "reason")):
            raise InvalidRequest("a sync point needs robot, before, until, layer and reason")
        if sync["before"] not in ids or sync["until"] not in ids or ids.index(sync["until"]) >= ids.index(sync["before"]):
            raise InvalidRequest(f"sync point of {sync['robot']} before {sync['before']} waits for {sync['until']}, "
                                 "which is not an earlier operation")
    return {"operations": operations, "sync_points": sync_points}
