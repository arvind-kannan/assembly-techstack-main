"""B1, the greedy two-robot scheduler (#108), at the planner seam."""

import logging
import math
from dataclasses import replace
from itertools import combinations

import networkx as nx
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState
from diss_assembly_dataclasses.feedback import InfeasibleOperation, ReasonCode
from diss_assembly_dataclasses.operation_plan import OperationPlan
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_reversal import AssemblyConstants, complete_by_reversal
from disassembly_pomdp.travel_time_table import TravelTimeTable
from motion_planning.common import assembly_xml_with
from mrtamp.assembly_cell import with_assembly_capabilities
from mrtamp.b1_planner import plan as b1_plan
from mrtamp.cell_simulation import FeasibilityAnswer
from mrtamp.nominal_schedule import nominal_schedule, operation_spans
from mrtamp.planning import Goal, GoalUnreachable, PlanningRequest, RunSettings
from mrtamp.run_controller import run
from mrtamp.stand_in import ModelLevelStandIn
from mrtamp.step_expansion import expand_step
from mrtamp.v1_planner import plan as v1_plan
from tests.mrtamp_scenarios import (
    SUPPLY_SLOTS,
    TWO_SCREWS_XML,
    product_model,
    step,
    travel_times,
    two_robot_cell,
    two_screws_graph,
    two_screws_snapshot_both_on_t6,
)


def request(screw_2_time=5.0, cell_simulation=None, **overrides):
    kwargs = dict(
        sequence_graph=two_screws_graph(screw_2_time),
        product_model=product_model(TWO_SCREWS_XML),
        cell=two_robot_cell(),
        start_snapshot=two_screws_snapshot_both_on_t6(),
        settings=RunSettings(goal=Goal(absent=("lid",))),
        cell_simulation=cell_simulation,
        travel_times=travel_times(),
    )
    kwargs.update(overrides)
    return PlanningRequest(**kwargs)


def spans_of(plan, req):
    """Start and end of every operation (by id), from the nominal schedule of the plan with its sync points."""
    return operation_spans(plan, req.cell, travel_times=req.travel_times)


def op_on(plan, target):
    return next(op for op in plan.operations if op.target == target)


class SpySimulation:
    """Answers feasibility queries (feasible, or infeasible when `refuse` says so) and records the groups."""

    def __init__(self, refuse=lambda group: False):
        self.groups, self.refuse = [], refuse

    def check_feasibility(self, group, snapshot, cell, *, product_model, travel_times):
        # what the cell simulation rejects as an invalid request
        assert len({op.id for op in group}) == len(group), "repeated operation ids"
        robots = [r for op in group for r in op.robots]
        assert len(set(robots)) == len(robots), "two operations of one robot"
        self.groups.append(tuple(group))
        if self.refuse(group):
            blamed = group[-1]
            return FeasibilityAnswer(
                False,
                InfeasibleOperation(
                    blamed.id, blamed.robots[0], ReasonCode.COLLISION, "scripted"
                ),
            )
        return FeasibilityAnswer(
            True, None, {op.id: op.shares[0].path_duration for op in group}
        )


def test_independent_steps_run_concurrently_on_the_two_robots_and_the_makespan_beats_v1():
    req = request()

    b1, v1 = b1_plan(req), v1_plan(req)

    assert isinstance(b1, OperationPlan)
    one, two = op_on(b1, "screw_1"), op_on(b1, "screw_2")
    assert {one.robots[0], two.robots[0]} == {"robot_1", "robot_2"}
    spans = spans_of(b1, req)
    assert spans[one.id][0] == spans[two.id][0] == 0.0  # both unscrew at once
    assert b1.metrics.makespan < v1.metrics.makespan
    assert b1.metrics.busy_time["robot_2"] > 0


def test_b1_plan_metrics_are_the_nominal_schedule_of_the_plan_with_its_sync_points():
    req = request(screw_2_time=7.0)
    plan = b1_plan(req)

    _, metrics, _ = nominal_schedule(
        plan.operations,
        req.cell,
        req.start_snapshot.cell_state.mounted_tools,
        plan.sync_points,
        travel_times=req.travel_times,
    )
    assert plan.metrics == metrics
    assert plan.sequences["robot_1"] and plan.sequences["robot_2"]


def test_a_step_that_depends_on_the_other_robots_step_waits_for_it_through_an_order_sync_point():
    # Robot 2 cannot remove the lid and is still unscrewing when Robot 1 is free for it; no risky process, so no
    # inspection between the steps stops the robots either
    req = request(
        screw_2_time=100.0,
        cell=two_robot_cell(robot_2_removes=False),
        settings=RunSettings(goal=Goal(absent=("lid",)), risky_processes=()),
    )
    plan = b1_plan(req)
    lid, screw_2 = op_on(plan, "lid"), op_on(plan, "screw_2")

    assert screw_2.robots[0] != lid.robots[0]
    [sync] = [s for s in plan.sync_points if s.before == lid.id]
    assert (sync.robot, sync.until, sync.layer, sync.reason) == (
        lid.robots[0],
        screw_2.id,
        "planner",
        "order",
    )
    spans = spans_of(plan, req)
    assert (
        spans[lid.id][0] == spans[screw_2.id][1] == 100.0
    )  # the wait is what delays the lid


def test_a_step_that_does_not_commute_never_overlaps_the_other_robots_step():
    req = request(screw_2_time=7.0)
    plan = b1_plan(req)
    spans = spans_of(plan, req)

    lid = op_on(plan, "lid")
    assert all(
        spans[op_on(plan, screw).id][1] <= spans[lid.id][0]
        for screw in ("screw_1", "screw_2")
    )


def test_each_b1_plan_function_sees_the_mounted_tools_as_they_are_before_its_step():
    seen = []

    def plan_function(product_model, snapshot, operation, robot, tool):
        seen.append((operation.target, snapshot.cell_state.mounted_tools[robot.id]))
        return operation

    start = Cell_Assembly_Snapshot(
        TWO_SCREWS_XML, CellState(mounted_tools={"robot_1": "H5", "robot_2": "H5"})
    )
    b1_plan(
        request(start_snapshot=start),
        plan_functions={
            Process.UNSCREWING: plan_function,
            Process.GRIPPING_AND_REMOVING: plan_function,
        },
    )

    # both robots start with H5; whichever removes the lid has unscrewed a screw with T6 first
    assert sorted(seen) == [("lid", "T6"), ("screw_1", "H5"), ("screw_2", "H5")]


def test_without_a_cell_simulation_b1_plans_with_nominal_times():
    assert isinstance(b1_plan(request(cell_simulation=None)), OperationPlan)


def test_with_a_cell_simulation_b1_sends_a_feasibility_query_per_concurrent_group():
    simulation = SpySimulation()

    plan = b1_plan(request(cell_simulation=simulation))

    assert isinstance(plan, OperationPlan)
    [group] = simulation.groups  # only the two unscrewings run at the same time
    assert {op.target for op in group} == {"screw_1", "screw_2"}
    assert len({op.robots[0] for op in group}) == len(
        group
    )  # at most one operation per robot


def test_each_operation_of_a_feasibility_query_has_its_own_id():
    # the cell simulation rejects a concurrent group whose operation ids repeat (invalid request)
    simulation = SpySimulation()

    b1_plan(request(cell_simulation=simulation))

    [group] = simulation.groups
    assert len({op.id for op in group}) == len(group)


def test_an_infeasible_concurrent_group_is_not_run_concurrently():
    req = request(cell_simulation=SpySimulation(refuse=lambda group: True))
    plan = b1_plan(req)

    spans = sorted(
        span
        for op_id, span in spans_of(plan, req).items()
        if not op_id.startswith("inspection")
    )
    assert all(
        a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:])
    )  # nothing overlaps


def test_the_run_controller_runs_b1_through_the_stand_in_to_the_goal():
    record = run(
        sequence_graph=two_screws_graph(7.0),
        product_model=product_model(TWO_SCREWS_XML),
        cell=two_robot_cell(),
        start_snapshot=two_screws_snapshot_both_on_t6(),
        settings=RunSettings(goal=Goal(absent=("lid",))),
        planner=b1_plan,
        cell_simulation=ModelLevelStandIn(),
        travel_times=travel_times(),
    )

    assert record.status == "goal_reached"
    plan, report = record.operation_plans[-1], record.simulation_reports[-1]
    assert (
        report.infeasible is None and report.metrics == plan.metrics
    )  # the stand-in honours the sync points


def test_b1_is_stateless_and_reports_goal_unreachable_instead_of_dropping_a_target_part():
    req = request()

    assert b1_plan(req) == b1_plan(req)
    assert isinstance(
        b1_plan(replace(req, settings=RunSettings(goal=Goal(absent=("lid", "base"))))),
        GoalUnreachable,
    )


def test_feasibility_queries_use_the_planners_own_step_expansion():
    def doubled(step, supply_slots=()):
        return [
            replace(o, duration=o.duration * 2) for o in expand_step(step, supply_slots)
        ]

    simulation = SpySimulation()

    plan = b1_plan(request(cell_simulation=simulation), expand=doubled)

    [group] = simulation.groups
    assert [op.shares[0].path_duration for op in group] == [
        10.0,
        10.0,
    ]  # the unscrewings, doubled from 5 s
    assert all(
        op.shares[0].path_duration == 10.0
        for op in plan.operations
        if op.target in ("screw_1", "screw_2")
    )


# Critical-path priority (#144): in the chain graphs, c must come off before d; the other parts are independent.
CHAIN_TIMES = {"a": 1.0, "b": 1.0, "c": 2.0, "d": 10.0}


def chain_xml(parts):
    lines = "\n".join(
        f'\t<part name="{p}" mesh_path="{p}.stl" density="8000.0" material="Generic" />'
        for p in parts
    )
    return f'<assembly name="chain">\n{lines}\n</assembly>'


def chain_graph(times):
    """Every order of removing the parts in `times` from the base by unscrewing, except that d comes off only after c."""
    g = nx.MultiDiGraph()
    parts = tuple(times)
    for n in range(len(parts) + 1):
        for removed in combinations(parts, n):
            state = tuple(sorted({"base", *parts} - set(removed)))
            for part in set(state) - {"base"}:
                if part == "d" and "c" in state:
                    continue
                after = tuple(sorted(set(state) - {part}))
                g.add_edge(
                    state, after, **step(Process.UNSCREWING, "T6", part, times[part])
                )
    return g


def chain_request(times=CHAIN_TIMES, start=None, **overrides):
    parts = ("base", *times)
    kwargs = dict(
        sequence_graph=chain_graph(times),
        product_model=product_model(chain_xml(parts)),
        cell=two_robot_cell(),
        start_snapshot=Cell_Assembly_Snapshot(
            chain_xml(start or parts),
            CellState(mounted_tools={"robot_1": "T6", "robot_2": "T6"}),
        ),
        settings=RunSettings(goal=Goal(absent=tuple(times)), risky_processes=()),
        travel_times=travel_times(),
    )
    kwargs.update(overrides)
    return PlanningRequest(**kwargs)


def test_critical_path_priority_starts_the_long_chain_first_and_beats_shortest_step_first_with_two_robots():
    req = chain_request()  # a, b 1 s; c 2 s, then d 10 s

    shortest, critical = (
        b1_plan(req, priority="shortest"),
        b1_plan(req, priority="critical_path"),
    )

    assert isinstance(shortest, OperationPlan) and isinstance(critical, OperationPlan)
    assert (
        spans_of(critical, req)[op_on(critical, "c").id][0] == 0.0
    )  # c starts at once, d is free sooner
    assert critical.metrics.makespan < shortest.metrics.makespan
    assert b1_plan(req) == shortest  # today's behaviour stays the default


def test_critical_path_priority_reads_the_precedence_relation_the_other_way_round_in_an_assembly_run():
    # Screwed in, d (5 s) must come before c (10 s); e and f (6 s each) are independent. The final inspection adds
    # 10 s. Critical path: d, e | c, f -> 25 s; shortest first: d, e | f, c -> 26 s; e and f first -> 31 s.
    times = {"c": 10.0, "d": 5.0, "e": 6.0, "f": 6.0}
    cell = two_robot_cell()
    cell = replace(
        cell,
        robots=tuple(
            replace(r, capabilities=r.capabilities + (Process.SCREWING_IN,))
            for r in cell.robots
        ),
    )
    req = chain_request(
        times,
        start=("base",),
        cell=cell,
        sequence_graph=complete_by_reversal(
            chain_graph(times), AssemblyConstants(screw_in_time=0.0)
        ),
        settings=RunSettings(goal=Goal(present=("base", *times)), risky_processes=()),
    )

    shortest, critical = (
        b1_plan(req, priority="shortest"),
        b1_plan(req, priority="critical_path"),
    )

    assert isinstance(critical, OperationPlan)
    assert spans_of(critical, req)[op_on(critical, "d").id][0] == 0.0
    assert (critical.metrics.makespan, shortest.metrics.makespan) == (25.0, 26.0)


# Interchangeable screws (#264): the export holds a group of order-free screws as one chain, by name, and lists the
# group in `collapsed_screw_groups`; B1 reads the screws of a group as order-free.
GROUP = ("screw_a", "screw_b", "screw_c", "screw_d")
# a and b sit 5 cm apart, as do c and d; the two pairs are 30 cm apart (metres, in the product model's frame)
POSITIONS = {
    "screw_a": (0.0, 0.0, 0.0),
    "screw_b": (0.05, 0.0, 0.0),
    "screw_c": (0.3, 0.0, 0.0),
    "screw_d": (0.35, 0.0, 0.0),
}


def group_xml(state=None, cover=False):
    """The lid with its four screws (and a cover on top), reduced to the parts of `state` if given."""
    parts = ("base", "lid", *GROUP, *(("cover",) if cover else ()))
    lines = "\n".join(
        f'\t<part name="{p}" mesh_path="{p}.stl" density="8000.0" material="Generic" />'
        for p in parts
    )
    screws = "\n".join(
        f'\t<connection type="screw" part_name1="{s}" part_name2="lid" screw_head="T6" />'
        for s in GROUP
    )
    xml = f'<assembly name="lid_with_a_screw_group">\n{lines}\n{screws}\n</assembly>'
    return xml if state is None else assembly_xml_with(xml, state)


def screw_group_graph(times=None, cover=False, cover_tool="H5"):
    """The collapsed export: (the cover comes off,) the screws come out in one chain by name, then the lid."""
    times = {s: 5.0 for s in GROUP} | (times or {})
    g = nx.MultiDiGraph()
    state = tuple(sorted(("base", "lid", *GROUP, *(("cover",) if cover else ()))))
    chain = [
        *((Process.GRIPPING_AND_REMOVING, cover_tool, "cover", 8.0),) * cover,
        *((Process.UNSCREWING, "T6", s, times[s]) for s in GROUP),
        (Process.GRIPPING_AND_REMOVING, "H5", "lid", 8.0),
    ]
    for process, tool, part, time in chain:
        after = tuple(sorted(set(state) - {part}))
        g.add_edge(state, after, **step(process, tool, part, time))
        state = after
    g.graph["collapsed_screw_groups"] = [list(GROUP)]
    return g


def group_request(
    times=None, cover=False, mounted=None, start=None, cover_tool="H5", **overrides
):
    kwargs = dict(
        sequence_graph=screw_group_graph(times, cover, cover_tool),
        product_model=product_model(group_xml(cover=cover)),
        cell=two_robot_cell(),
        start_snapshot=Cell_Assembly_Snapshot(
            group_xml(start, cover),
            CellState(mounted_tools=mounted or {"robot_1": "T6", "robot_2": "T6"}),
        ),
        settings=RunSettings(goal=Goal(absent=("lid",))),
        travel_times=travel_times(),
    )
    kwargs.update(overrides)
    return PlanningRequest(**kwargs)


def screw_ops(plan):
    return [op for op in plan.operations if op.target in GROUP]


def overlapping(spans, ops):
    return [
        (a, b)
        for a, b in combinations(ops, 2)
        if spans[a.id][0] < spans[b.id][1] and spans[b.id][0] < spans[a.id][1]
    ]


def test_b1_plans_group_screws_out_of_chain_order_so_that_both_robots_unscrew():
    req = group_request()

    plan = b1_plan(req, part_positions=POSITIONS)

    assert isinstance(plan, OperationPlan)
    assert req.settings.goal.reached(plan.operations[-1].expected_product_state)
    by_robot = {
        r: [op.target for op in screw_ops(plan) if op.robots[0] == r]
        for r in ("robot_1", "robot_2")
    }
    assert all(len(screws) == 2 for screws in by_robot.values())
    assert any(
        op.expected_product_state not in req.sequence_graph for op in screw_ops(plan)
    )  # some screws came out of chain order
    spans = spans_of(plan, req)
    assert [spans[op.id][0] for op in screw_ops(plan)[:2]] == [0.0, 0.0]
    assert plan.metrics.makespan < b1_plan(req, part_positions={}).metrics.makespan


def group_assembly_request(**overrides):
    """As on the battery: Robot 1 inserts the lid; both robots may screw in, Robot 2 waits for the lid. Tool changes
    take no time here, so Robot 1 does not lose the screws to its change back from the lid's gripper."""
    cell = with_assembly_capabilities(two_robot_cell())
    second = replace(
        cell.robots[1],
        capabilities=cell.robots[1].capabilities + (Process.SCREWING_IN,),
    )
    no_tool_changes = TravelTimeTable(
        {k: 0.0 if "toolchange" in k else v for k, v in travel_times().times.items()}
    )
    kwargs = dict(
        start=("base",),
        sequence_graph=complete_by_reversal(
            screw_group_graph(), AssemblyConstants(screw_in_time=0.0)
        ),
        cell=replace(cell, robots=(cell.robots[0], second)),
        settings=RunSettings(goal=Goal(present=("base", "lid", *GROUP))),
        supply_slots=SUPPLY_SLOTS,
        travel_times=no_tool_changes,
    )
    return group_request(**(kwargs | overrides))


def test_in_an_assembly_run_both_robots_screw_in_once_the_lid_is_in():
    req = group_assembly_request()

    plan = b1_plan(req, part_positions=POSITIONS)

    assert isinstance(plan, OperationPlan)
    assert req.settings.goal.reached(plan.operations[-1].expected_product_state)
    screwers = {op.robots[0] for op in screw_ops(plan)}
    assert screwers == {"robot_1", "robot_2"}
    lid_in = max(
        end
        for op_id, (_, end) in spans_of(plan, req).items()
        if op_id in {op.id for op in plan.operations if op.target == "lid"}
    )
    assert all(spans_of(plan, req)[op.id][0] >= lid_in for op in screw_ops(plan))


def test_two_screws_of_a_group_run_at_the_same_time_only_when_they_are_the_set_distance_apart():
    req = group_request()

    for positions, distance in ((POSITIONS, 0.2), (POSITIONS, 0.01), ({}, 0.2)):
        plan = b1_plan(req, part_positions=positions, min_screw_distance=distance)
        pairs = overlapping(spans_of(plan, req), screw_ops(plan))
        for a, b in pairs:
            assert math.dist(positions[a.target], positions[b.target]) >= distance
        if distance == 0.01:
            assert len(pairs) >= 2  # the close screws pair up too
        if not positions:
            assert not pairs  # screws without a position never pair


def b1_errors(caplog):
    return [
        r
        for r in caplog.records
        if r.name == "mrtamp.b1_planner" and r.levelno >= logging.ERROR
    ]


def test_a_group_screw_without_a_position_is_logged_as_one_error_at_plan_start(
    caplog,
):
    req = group_request()
    known = {k: v for k, v in POSITIONS.items() if k not in ("screw_b", "screw_c")}

    with caplog.at_level(logging.ERROR, logger="mrtamp.b1_planner"):
        plan = b1_plan(req, part_positions=known)

    assert isinstance(plan, OperationPlan)
    (error,) = b1_errors(caplog)
    assert "screw_b" in error.getMessage() and "screw_c" in error.getMessage()
    assert not any(name in error.getMessage() for name in ("screw_a", "screw_d"))
    for a, b in overlapping(spans_of(plan, req), screw_ops(plan)):
        assert a.target in known and b.target in known


def test_no_error_is_logged_when_every_group_screw_has_a_position(caplog):
    with caplog.at_level(logging.ERROR, logger="mrtamp.b1_planner"):
        b1_plan(group_request(), part_positions=POSITIONS)

    assert not b1_errors(caplog)


def test_a_group_screw_goes_to_the_robot_that_finishes_it_first_also_when_it_has_to_wait():
    # Robot 2 only unscrews and must wait for Robot 1's cover; Robot 1 holds H5 and would first change to T6
    req = group_request(
        cover=True,
        mounted={"robot_1": "H5", "robot_2": "T6"},
        cell=two_robot_cell(robot_2_removes=False),
        settings=RunSettings(goal=Goal(absent=("lid",)), risky_processes=()),
    )

    plan = b1_plan(req, part_positions=POSITIONS)

    cover, first = op_on(plan, "cover"), screw_ops(plan)[0]
    assert (cover.robots[0], first.robots[0]) == ("robot_1", "robot_2")
    [sync] = [s for s in plan.sync_points if s.before == first.id]
    assert (sync.robot, sync.until, sync.reason) == ("robot_2", cover.id, "order")
    spans = spans_of(plan, req)
    assert spans[first.id][0] == spans[cover.id][1]


class PairSpy(SpySimulation):
    """Also records the steps (target, robot) of each concurrent group it accepted."""

    def __init__(self, refuse=lambda group: False):
        super().__init__(refuse)
        self.accepted = []

    def check_feasibility(self, group, snapshot, cell, *, product_model, travel_times):
        answer = super().check_feasibility(
            group,
            snapshot,
            cell,
            product_model=product_model,
            travel_times=travel_times,
        )
        if answer.feasible:
            self.accepted.append({(op.target, op.robots[0]) for op in group})
        return answer


def test_screws_run_together_only_in_concurrent_groups_the_cell_simulation_accepted():
    def refuse_a_with_c(group):
        return {op.target for op in group} >= {"screw_a", "screw_c"}

    scenarios = (
        dict(cover=False),
        dict(cover=True, mounted={"robot_1": "H5", "robot_2": "T6"}),
        # Robot 1 takes off the cover with T6 and then screw_a; Robot 2 waits for the cover with screw_c, which
        # then overlaps screw_a although only the cover blocks it
        dict(cover=True, cover_tool="T6", cell=two_robot_cell(robot_2_removes=False)),
    )
    for refuse in (lambda group: True, refuse_a_with_c):
        for scenario in scenarios:
            simulation = PairSpy(refuse)
            req = group_request(cell_simulation=simulation, **scenario)
            plan = b1_plan(req, part_positions=POSITIONS)
            ops = [op for op in plan.operations if op.target is not None]
            for a, b in overlapping(spans_of(plan, req), ops):
                pair = {(a.target, a.robots[0]), (b.target, b.robots[0])}
                assert any(pair <= group for group in simulation.accepted)


def test_no_inspection_falls_between_two_screws_of_a_group_because_they_are_out_of_chain_order():
    plan = b1_plan(group_request(), part_positions=POSITIONS)

    kinds = [
        "inspection" if op.process is Process.INSPECTING else op.target
        for op in plan.operations
    ]
    last_screw = max(i for i, k in enumerate(kinds) if k in GROUP)
    assert "inspection" not in kinds[:last_screw]
    assert kinds[last_screw + 1 :] == ["inspection", "lid", "inspection"]


OFF_CHAIN = ("base", "lid", "screw_b", "screw_d")  # a and c are out, b and d still in


def test_b1_replans_from_a_product_state_off_the_chain_through_the_run_controller():
    assert OFF_CHAIN not in screw_group_graph()
    requests = []

    class FirstInspectionSeesTwoScrewsLeft(ModelLevelStandIn):
        def simulate(self, request):
            report = super().simulate(request)
            if len(requests) > 1:
                return report
            first = next(
                op
                for op in request.operation_plan.operations
                if op.process is Process.INSPECTING
            )
            seen = report.observed_snapshots[first.id]
            return replace(
                report,
                observed_snapshots={
                    first.id: Cell_Assembly_Snapshot(
                        group_xml(OFF_CHAIN), seen.cell_state
                    )
                },
            )

    def planner(request):
        requests.append(request)
        return b1_plan(request, part_positions=POSITIONS)

    record = run(
        sequence_graph=screw_group_graph(),
        product_model=product_model(group_xml()),
        cell=two_robot_cell(),
        start_snapshot=group_request().start_snapshot,
        settings=RunSettings(goal=Goal(absent=("lid",))),
        planner=planner,
        cell_simulation=FirstInspectionSeesTwoScrewsLeft(),
        travel_times=travel_times(),
    )

    assert len(requests) == 2
    assert requests[1].start_snapshot.product_state_key == OFF_CHAIN
    assert record.status == "goal_reached"
    assert {op.target for op in screw_ops(record.operation_plans[-1])} == {
        "screw_b",
        "screw_d",
    }


def test_critical_path_priority_does_not_read_the_chain_of_a_group_as_forced_order():
    # screw_d is the slow one and last in the chain; read as forced order, screw_a would have the longest chain
    req = group_request(times={"screw_d": 9.0})

    plan = b1_plan(req, part_positions=POSITIONS, priority="critical_path")

    assert spans_of(plan, req)[op_on(plan, "screw_d").id][0] == 0.0


def test_b1_plans_from_a_start_off_the_chain_with_the_critical_path_of_the_remaining_screws():
    req = group_request(times={"screw_d": 9.0}, start=OFF_CHAIN)

    plan = b1_plan(req, part_positions=POSITIONS, priority="critical_path")

    assert isinstance(plan, OperationPlan)
    assert {op.target for op in screw_ops(plan)} == {"screw_b", "screw_d"}
    assert spans_of(plan, req)[op_on(plan, "screw_d").id][0] == 0.0


def test_on_a_graph_without_screw_groups_the_screw_settings_change_nothing(caplog):
    req = request(screw_2_time=7.0, cell_simulation=SpySimulation())

    with caplog.at_level(logging.ERROR, logger="mrtamp.b1_planner"):
        assert b1_plan(req) == b1_plan(
            req, part_positions={"screw_1": (0, 0, 0)}, min_screw_distance=5.0
        )
    assert not b1_errors(caplog)


def test_a_group_screw_whose_position_is_not_a_finite_number_is_logged_and_never_paired(
    caplog,
):
    req = group_request()
    nan = float("nan")
    positions = {
        **POSITIONS,
        "screw_b": (nan, nan, nan),
        "screw_c": (math.inf, 0.0, 0.0),
    }

    with caplog.at_level(logging.ERROR, logger="mrtamp.b1_planner"):
        plan = b1_plan(req, part_positions=positions)

    (error,) = b1_errors(caplog)
    assert "screw_b" in error.getMessage() and "screw_c" in error.getMessage()
    assert not any(name in error.getMessage() for name in ("screw_a", "screw_d"))
    for a, b in overlapping(spans_of(plan, req), screw_ops(plan)):
        assert not {a.target, b.target} & {"screw_b", "screw_c"}


def test_the_feasibility_query_of_an_assembly_run_gets_the_parts_inserted_so_far():
    class SnapshotSpy(SpySimulation):
        def __init__(self):
            super().__init__()
            self.snapshots = []

        def check_feasibility(
            self, group, snapshot, cell, *, product_model, travel_times
        ):
            self.snapshots.append(snapshot)
            return super().check_feasibility(
                group,
                snapshot,
                cell,
                product_model=product_model,
                travel_times=travel_times,
            )

    simulation = SnapshotSpy()

    plan = b1_plan(
        group_assembly_request(cell_simulation=simulation), part_positions=POSITIONS
    )

    assert isinstance(plan, OperationPlan)
    assert simulation.groups  # screws in the group run together once the lid is in
    for group, snapshot in zip(simulation.groups, simulation.snapshots):
        parts = set(snapshot.product_state_key)
        assert "lid" in parts  # inserted earlier in the run
        assert all(op.step.before == snapshot.product_state_key for op in group[:1])
