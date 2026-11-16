"""B1 baseline MRTAMP planner: a greedy two-robot scheduler, the yardstick for theses.

A free robot takes the ready step with the shortest estimated time (nominal time plus tool change time)
that keeps the goal reachable and commutes with the step the other robot is carrying out. With
`priority="critical_path"` it first prefers the step with the longest chain of steps still waiting behind it in
the precedence relation (`disassembly_pomdp.graph_precedence`), computed once per request for the goal of the
run settings; the relation only ranks the out-edges of the current product state, it chooses no path. A step that does
not commute never overlaps the other robot's step: if no robot can start it concurrently, the robot waits
for the other's step through a task-level sync point (*order*). When a cell simulation is connected,
every concurrent group is sent to its `check_feasibility` first; a group it refuses is not run
concurrently. Without one, nominal times decide. Inspections are placed by the shared helper
(`mrtamp.inspections`), step expansion and the step catalogue are the shared ones too; the chosen steps go to
the shared operation-plan builder (`mrtamp.operation_plan_builder`), which keeps the robots' clocks. The robots are
tried in the order of their clocks, so with more than two robots it is a list scheduler. Stateless:
everything comes from the planning request. Ports the idea of the legacy planner search graph, not its code.

Interchangeable screws (#264): B1 reads the sequence graph through `mrtamp.screw_groups.GroupView`, so the screws of
a group in `collapsed_screw_groups` are order-free: any remaining screw of the group is a next step, and same-group
screws commute. A group screw goes to the robot that would finish it first, also when that robot must wait for a
running step. Two same-group screws run at the same time only when they are at least `min_screw_distance` apart
(by `part_positions`); otherwise the robot waits for the other's screw. A group screw without a position (none,
or one that is not a finite number) is an error: B1 logs it once at plan start, naming the screws, and never pairs such a screw. A waiting group candidate
is also sent to the cell simulation with the steps it would still overlap once its blockers are done. On a graph
without groups none of this applies, and B1 plans as before.
"""

from __future__ import annotations

import functools
import itertools
import logging
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Literal, Mapping, Sequence

import networkx as nx
from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot

from disassembly_pomdp.graph_precedence import PrecedenceError, extract_precedence
from disassembly_pomdp.graph_steps import Step
from mrtamp.inspections import commute
from mrtamp.operation_plan_builder import CommittedStep, OperationPlanBuilder
from mrtamp.plan_functions import PlanFunction
from mrtamp.planning import (
    GoalUnreachable,
    Planner,
    PlanningRequest,
    PlanResult,
    unreachable_reason,
)
from mrtamp.screw_groups import MIN_SCREW_DISTANCE, GroupView, apart, has_position
from mrtamp.step_catalogue import StepCatalogue
from mrtamp.step_expansion import expand_step
from mrtamp.step_operations import RobotPlanning, plan_step

logger = logging.getLogger(__name__)


@dataclass
class _Candidate:
    robot: Robot
    after: tuple[str, ...]
    step: Step
    estimate: float  # nominal time with penalty cost, plus tool change time
    concurrent: list[
        CommittedStep
    ]  # steps of the other robots still running when this one would start
    blockers: list[CommittedStep] = field(
        default_factory=list
    )  # those it does not commute with, or the cell simulation refused
    critical_path: float = 0.0  # stays 0 unless B1 ranks by critical path
    in_group: bool = False  # a screw of a group of interchangeable screws

    @property
    def sort_key(self):
        return (
            bool(self.blockers),
            -self.critical_path,
            self.estimate,
            self.step.target_part,
            self.step.process.value,
            self.after,
        )


Priority = Literal["shortest", "critical_path"]


def plan(
    request: PlanningRequest,
    expand=expand_step,
    plan_functions: Mapping[Process, PlanFunction] | None = None,
    priority: Priority = "shortest",
    part_positions: Mapping[str, Sequence[float]] | None = None,
    min_screw_distance: float = MIN_SCREW_DISTANCE,
) -> PlanResult:
    """`expand(step, supply_slots)` turns a step into operations; pass your own to research other expansions.
    Shares get paths from `plan_functions` (nominal times for a process without one). `priority` is the rule a
    free robot ranks its candidate steps by: "shortest" step first, or longest "critical_path" first.
    `part_positions` (metres, see `mrtamp.screw_groups.part_positions`) and `min_screw_distance` decide which
    same-group screws may run at the same time."""
    if priority not in ("shortest", "critical_path"):
        raise ValueError(
            f"unknown priority {priority!r} (expected 'shortest' or 'critical_path')"
        )
    settings = request.settings
    graph_catalogue = StepCatalogue(request, expand)
    state = request.start_snapshot.product_state_key
    view = GroupView(request.sequence_graph)
    _log_screws_without_position(view, part_positions or {})
    good = _States(view, _states_reaching_goal(request, graph_catalogue))
    view_catalogue = graph_catalogue.through(view)
    if state not in good:
        return GoalUnreachable(unreachable_reason(request))
    critical_path = _critical_path(request, view) if priority == "critical_path" else {}
    screws = _ScrewRule(view, part_positions or {}, min_screw_distance)

    builder = OperationPlanBuilder(request, expand, plan_functions, graph=view)
    while not settings.goal.reached(state):
        chosen = _choose(
            request,
            expand,
            view_catalogue,
            good,
            state,
            builder.committed,
            builder.clock,
            builder.mounted_tools,
            critical_path,
            screws,
        )
        if chosen is None:
            return GoalUnreachable(unreachable_reason(request))
        wait_for = (
            max(chosen.blockers, key=lambda s: s.end) if chosen.blockers else None
        )
        builder.commit(chosen.robot, state, chosen.after, chosen.step, wait_for)
        state = chosen.after
    return builder.finish()


def _states_reaching_goal(
    request: PlanningRequest, catalogue: StepCatalogue
) -> set[tuple[str, ...]]:
    """The product states from which some robots' admissible steps lead to the goal."""
    settings, graph = request.settings, request.sequence_graph
    predecessors = defaultdict(set)
    for u in graph.nodes:
        for robot in request.cell.robots:
            for s in catalogue.steps(u, robot):
                predecessors[s.after].add(u)
    good = {n for n in graph.nodes if settings.goal.reached(n)}
    queue = deque(good)
    while queue:
        for u in predecessors[queue.popleft()]:
            if u not in good:
                good.add(u)
                queue.append(u)
    return good


class _States:
    """The product states from which the goal is reached, read through the group view: a state off a group's
    chain counts as its chain state."""

    def __init__(self, view: GroupView, good: set[tuple[str, ...]]):
        self.view, self.good = view, good

    def __contains__(self, state) -> bool:
        return self.view.chain_state(tuple(state)) in self.good


def _log_screws_without_position(
    view: GroupView, positions: Mapping[str, Sequence[float]]
) -> None:
    """A group screw without a position is a broken product model or a caller that passes no positions."""
    missing = sorted(
        part
        for g in view.groups
        for part in g
        if part not in positions or not has_position(positions[part])
    )
    if missing:
        logger.error(
            "group screws without a position, they never run at the same time as another screw of their group: %s",
            ", ".join(missing),
        )


@dataclass(frozen=True)
class _ScrewRule:
    """When two screws of the same group may run at the same time: when they are at least `min_distance` apart."""

    view: GroupView
    positions: Mapping[str, Sequence[float]]
    min_distance: float

    def too_close(self, running: CommittedStep, step: Step) -> bool:
        group = self.view.group_of(step.target_part)
        other = running.planned.step.target_part
        return (
            group is not None
            and other in group
            and not apart(other, step.target_part, self.positions, self.min_distance)
        )


def _choose(
    request,
    expand,
    catalogue,
    good,
    state,
    running,
    clock,
    mounted,
    critical_path,
    screws,
) -> _Candidate | None:
    """The next step and robot: the first robot, by clock, with a candidate that can start concurrently takes its
    best (longest critical path if ranked by it, then shortest); if none can, the candidate that would finish first
    waits for the steps it does not commute with. When that first best is a group screw, the group screw that would
    finish first wins instead, among the best of every robot, also one that must wait."""
    cell = request.cell
    order = sorted(cell.robots, key=lambda r: (clock[r.id], cell.robots.index(r)))
    waiting, free_screws = [], []
    for robot in order:
        candidates = _candidates(
            catalogue,
            good,
            state,
            robot,
            running,
            clock,
            mounted,
            critical_path,
            screws,
        )
        best = _best_of(request, expand, state, candidates, clock, mounted, screws)
        if best is None:
            continue
        if best.blockers:
            waiting.append(best)
        elif not free_screws and not best.in_group:
            return best
        elif best.in_group:
            free_screws.append(best)
    if not free_screws:
        return (
            min(waiting, key=lambda c: _finish(c, clock, request)) if waiting else None
        )
    contenders = [*free_screws, *(c for c in waiting if c.in_group)]
    return min(
        contenders,
        key=lambda c: (
            _finish(c, clock, request),
            bool(c.blockers),
            order.index(c.robot),
        ),
    )


def _best_of(
    request, expand, state, candidates, clock, mounted, screws
) -> _Candidate | None:
    """A robot's best candidate: the first that can start concurrently and that the cell simulation accepts with
    the steps still running, else one that waits for its blockers (with groups: the one that would finish first)."""
    while candidates:
        best = candidates[0]
        if best.blockers:
            if not screws.view.groups:
                return best
            return _waiting_best(request, expand, state, candidates, clock, mounted)
        if best.concurrent and not _feasible_together(
            request, expand, state, best, mounted, clock[best.robot.id]
        ):
            best.blockers = list(
                best.concurrent
            )  # the cell simulation refused: do not run them together
            candidates.sort(key=lambda c: c.sort_key)
            continue
        return best
    return None


def _waiting_best(request, expand, state, candidates, clock, mounted) -> _Candidate:
    """The waiting candidate that would finish first. The running steps it would still overlap once its blockers
    are done go to the cell simulation with it; when it refuses, they become blockers too."""
    while True:
        best = min(candidates, key=lambda c: (_finish(c, clock, request), c.sort_key))
        start = _finish(best, clock, request) - best.estimate
        others = replace(
            best, concurrent=[s for s in best.concurrent if s not in best.blockers]
        )
        overlap = _overlapping(others, start)
        if not overlap or _feasible_together(
            request, expand, state, others, mounted, start
        ):
            return best
        best.blockers = [*best.blockers, *overlap]


def _overlapping(candidate: _Candidate, start: float) -> list[CommittedStep]:
    """The candidate's concurrent steps that would run while it does, if it starts at `start`."""
    return [
        s
        for s in candidate.concurrent
        if s.end > start and s.start < start + candidate.estimate
    ]


def _candidates(
    catalogue,
    good,
    state,
    robot,
    running,
    clock,
    mounted,
    critical_path,
    screws,
) -> list[_Candidate]:
    graph = screws.view
    concurrent = [s for s in running if s.robot != robot.id and s.end > clock[robot.id]]
    found = []
    for admissible in catalogue.steps(state, robot, mounted.get(robot.id)):
        after, step = admissible.after, admissible.step
        if after not in good:
            continue
        effect = set(state) ^ set(after)
        blockers = [
            s
            for s in concurrent
            if not commute(graph, state, s.planned.step, s.effect, step, effect)
            or screws.too_close(s, step)
        ]
        found.append(
            _Candidate(
                robot,
                after,
                step,
                admissible.cost,
                concurrent,
                blockers,
                critical_path.get(step.target_part, 0.0),
                graph.group_of(step.target_part) is not None,
            )
        )
    return sorted(found, key=lambda c: c.sort_key)


def _critical_path(request: PlanningRequest, view: GroupView) -> dict[str, float]:
    """Each part to be removed (in an assembly run: added) on the way to the goal, with its critical path: its
    shortest step time plus the longest chain of steps the precedence relation puts after it.

    The goal state is the cheapest one reachable from the start along the steps the goal allows. The
    relation is read in the disassembly direction, from the full product down to the lower of start and goal
    state; removed parts keep that order, inserted parts take it reversed, and a pair of one removed and one
    inserted part is not ordered. The relation holds on every path from the full product, so when replanning from a
    later start state it may order fewer pairs than one for that start alone. Empty, so all steps rank equal (shortest
    first), if the relation cannot be had; a PrecedenceError is then reported as a warning.

    With interchangeable screws the relation is read on the graph from the start's chain state; the order the chain
    puts between two screws of a group is not forced, and each screw of a group stands for any other."""
    graph, settings = request.sequence_graph, request.settings
    start = request.start_snapshot.product_state_key
    chain_start = view.chain_state(start)
    run_steps = graph.edge_subgraph(
        (u, v, k)
        for u, v, k, attrs in graph.edges(keys=True, data=True)
        if settings.goal.allows_step(Step.from_edge(attrs).inverse, set(u) ^ set(v))
    )
    if chain_start not in run_steps:
        return {}
    cost = nx.single_source_dijkstra_path_length(
        run_steps,
        chain_start,
        weight=lambda u, v, steps: min(
            Step.from_edge(s).action_time for s in steps.values()
        ),
    )
    ends = [n for n in cost if settings.goal.reached(n)]
    if not ends:
        return {}
    end = min(ends, key=cost.get)
    with_parts = graph.copy()
    for node, attrs in with_parts.nodes(data=True):
        attrs.setdefault("parts_present", list(node))
    try:
        _, precedence = extract_precedence(
            with_parts, min((chain_start, end), key=len)
        )  # the bottom of the run's steps
    except PrecedenceError as error:
        warnings.warn(
            f"B1 ranks shortest step first: no precedence relation for the critical path ({error})"
        )
        return {}
    chain_parts = set(chain_start) ^ set(end)
    chain_removed = set(chain_start) - set(end)
    followers = defaultdict(set)
    for p, q in precedence:
        if (
            p in chain_parts
            and q in chain_parts
            and (p in chain_removed) == (q in chain_removed)
            and (view.group_of(p) is None or q not in view.group_of(p))
        ):  # a mixed pair is not ordered, nor a pair of the same group
            first, then = (
                (p, q) if p in chain_removed else (q, p)
            )  # inserted parts are ordered in reverse
            followers[first].add(then)
    parts = set(start) ^ set(end)
    followers = _interchangeable(followers, view, parts)
    step_time = {
        part: min(
            s.action_time
            for s in (Step.from_edge(a) for _, _, a in run_steps.edges(data=True))
            if s.target_part == part
        )
        for part in parts
    }

    @functools.cache
    def chain_time(part: str) -> float:
        return step_time[part] + max(
            (chain_time(q) for q in followers[part]), default=0.0
        )

    return {part: chain_time(part) for part in parts}


def _interchangeable(followers, view: GroupView, parts: set[str]):
    """The followers relation over the run's `parts`, with each screw of a group standing for every screw of the
    group among them: they take each other's followers and follow what any of them follows."""
    if not view.groups:
        return followers

    def stands_for(part):
        group = view.group_of(part)
        return (group & parts) if group is not None else {part} & parts

    widened = defaultdict(set)
    for p, qs in followers.items():
        for q in qs:
            for first in stands_for(p):
                widened[first] |= stands_for(q)
    return widened


def _finish(
    candidate: _Candidate, clock: dict[str, float], request: PlanningRequest
) -> float:
    """When a waiting candidate would end: after its robot is free and its blockers are done."""
    start = max([clock[candidate.robot.id], *(s.end for s in candidate.blockers)])
    return start + candidate.estimate


def _feasible_together(
    request: PlanningRequest,
    expand,
    state,
    candidate: _Candidate,
    mounted: dict,
    start: float,
) -> bool:
    """Ask the connected cell simulation whether the candidate's first operation, started at `start`, can run with
    the operations of the steps it would overlap; without one, nominal times decide and the answer is yes.

    A concurrent group holds at most one step per robot, so the overlapped steps go in turns: the first of each
    robot, then the second, and so on; every group must be accepted. Each step stands for its first operation, as
    a nominal operation (no plan function, so a plan failure of a running step does not refuse the group), and a
    group starts from the state before its earliest step. First operations (gripping at the supply, removing,
    unscrewing, ...) hold nothing yet."""
    simulation = request.cell_simulation
    overlap = _overlapping(candidate, start)
    if simulation is None or not overlap:
        return True
    by_robot = defaultdict(list)
    for s in overlap:
        by_robot[s.robot].append(s)
    return all(
        _feasible_group(
            request,
            expand,
            state,
            candidate,
            mounted,
            [s for s in overlap if s in turn],
        )
        for turn in itertools.zip_longest(*by_robot.values())
    )


def _feasible_group(
    request: PlanningRequest, expand, state, candidate: _Candidate, mounted, steps
) -> bool:
    """One feasibility query: the candidate's first operation with the first operations of `steps`."""
    robots = {r.id: r for r in request.cell.robots}

    def first_operation(number, robot, before, after, step):
        # no plan functions here, so the cell state of the snapshot is never read; each operation of the group
        # gets its own id, as the cell simulation requires
        return plan_step(
            RobotPlanning(request, robot, expand),
            before,
            after,
            step,
            10**6 + number,
            {},
            cell_state=request.start_snapshot.cell_state,
        ).operations[0]

    others = [
        first_operation(
            number, robots[s.robot], s.planned.before, s.planned.after, s.planned.step
        )
        for number, s in enumerate(steps)
    ]
    mine = first_operation(
        len(others), candidate.robot, state, candidate.after, candidate.step
    )
    snapshot = request.product_model.snapshot_at(
        tuple(steps[0].planned.before), CellState(mounted_tools=dict(mounted))
    )
    return request.cell_simulation.check_feasibility(
        (*others, mine),
        snapshot,
        request.cell,
        product_model=request.product_model,
        travel_times=request.travel_times,
    ).feasible


def make_b1_planner(
    plan_functions: Mapping[Process, PlanFunction],
    priority: Priority = "shortest",
    part_positions: Mapping[str, Sequence[float]] | None = None,
    min_screw_distance: float = MIN_SCREW_DISTANCE,
) -> Planner:
    """B1 with its shares planned by `plan_functions`, ranking candidate steps by `priority`; same-group screws
    run at the same time only when `part_positions` puts them at least `min_screw_distance` apart."""
    return functools.partial(
        plan,
        plan_functions=plan_functions,
        priority=priority,
        part_positions=part_positions,
        min_screw_distance=min_screw_distance,
    )
