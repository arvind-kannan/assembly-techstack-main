"""Whole runs at the run controller seam: v1 planner + model-level stand-in on a small sequence graph."""

import json
from dataclasses import replace

import pytest
from diss_assembly_dataclasses.feedback import (
    Anomaly,
    InfeasibleOperation,
    OperationOutcome,
    ReasonCode,
)
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.operation_plan import (
    OperationPlan,
    PlanMetrics,
    SyncPoint,
)
from diss_assembly_dataclasses.process import Process

from motion_planning.common import assembly_xml_with
from mrtamp.penalties import StepKey, penalties
from mrtamp.planning import Goal, PlanningRequest
from mrtamp.run_controller import RunSettings, _rest_of, run
from mrtamp.stand_in import ModelLevelStandIn
from mrtamp.v1_planner import plan as v1_plan
from tests.mrtamp_scenarios import (
    lid_screwed_on_base_graph,
    product_model,
    start_snapshot,
    transformer_cell,
    travel_times,
)
from tests.sim_loopback import loopback_simulation, requires_sim


def run_to_remove_the_lid(**overrides):
    kwargs = dict(
        sequence_graph=lid_screwed_on_base_graph(),
        product_model=product_model(),
        cell=transformer_cell(),
        start_snapshot=start_snapshot(),
        settings=RunSettings(goal=Goal(absent=("lid",))),
        planner=v1_plan,
        cell_simulation=ModelLevelStandIn(),
        travel_times=travel_times(),
    )
    kwargs.update(overrides)
    return run(**kwargs)


def test_disassembly_run_reaches_the_goal_on_the_cheapest_path_with_one_robot():
    record = run_to_remove_the_lid()

    assert record.status == "goal_reached"
    plan = record.operation_plans[-1]
    assert [(op.process, op.target) for op in plan.operations] == [
        (Process.UNSCREWING, "screw"),
        (Process.INSPECTING, None),  # the removal depends on the uninspected unscrewing
        (Process.GRIPPING_AND_REMOVING, "lid"),
        (Process.INSPECTING, None),
    ]
    assert {
        op.robots for op in plan.operations if op.process is not Process.INSPECTING
    } == {("robot_1",)}
    assert plan.operations[-1].robots == ("robot_1", "robot_2")
    assert record.final_snapshot.product_state_key == ("base",)


def test_plan_metrics_count_nominal_times_tool_changes_and_the_inspections():
    record = run_to_remove_the_lid()

    # H5 -> T6 (80 s), unscrew 5 s, inspection 10 s, T6 -> H5 (80 s), remove lid 8 s, inspection 10 s
    metrics = record.operation_plans[-1].metrics
    assert metrics.makespan == 193.0
    assert metrics.busy_time == {"robot_1": 193.0, "robot_2": 20.0}
    assert metrics.tool_changes == 2
    assert metrics.inspections == 2
    assert record.simulation_reports[-1].metrics == metrics


def test_run_record_is_json_with_the_graph_settings_plans_and_reports(tmp_path):
    record = run_to_remove_the_lid()
    path = tmp_path / "run_record.json"

    record.save_json(path)
    d = json.loads(path.read_text(encoding="utf-8"))

    assert d["status"] == "goal_reached"
    assert d["settings"]["goal"] == {"present": [], "absent": ["lid"]}
    assert len(d["sequence_graph"]["edges"]) == 3
    [entry] = d["entries"]
    assert (
        OperationPlan.from_dict(entry["operation_plan"]) == record.operation_plans[-1]
    )
    assert entry["simulation_report"]["ok"] is True
    assert entry["simulation_report"]["nc_programs"] is None
    assert entry["decision"] == "goal_reached"


def test_goal_unreachable_stops_the_run_with_a_reason_instead_of_dropping_target_parts():
    record = run_to_remove_the_lid(
        settings=RunSettings(goal=Goal(absent=("lid", "base")))
    )

    assert record.status == "goal_unreachable"
    assert record.operation_plans == []
    [entry] = record.to_dict()["entries"]
    assert "base" in entry["goal_unreachable"]
    assert entry["simulation_report"] is None


# --- replanning -------------------------------------------------------------------------------


class ScriptedSimulation:
    """The stand-in with a script: call n applies fault n (a function of request and report) or none."""

    produces_nc_programs = False

    def __init__(self, *faults):
        self.faults = list(faults)
        self.requests = []

    def simulate(self, request):
        self.requests.append(request)
        report = ModelLevelStandIn().simulate(request)
        fault = self.faults.pop(0) if self.faults else None
        return fault(request, report) if fault else report


def infeasible_at(target):
    def fault(request, report):
        ops = request.operation_plan.operations
        i = next(i for i, op in enumerate(ops) if op.target == target)
        reason = InfeasibleOperation(
            ops[i].id, ops[i].robots[0], ReasonCode.COLLISION, "scripted"
        )
        return replace(
            report,
            feasible_operations=tuple(op.id for op in ops[:i]),
            infeasible=reason,
            observed_snapshots={},
            outcomes=(),
        )

    return fault


def fails_at(target):
    """The operation on `target` does not take effect: the inspection sees the state before it."""

    def fault(request, report):
        ops = request.operation_plan.operations
        i = next(i for i, op in enumerate(ops) if op.target == target)
        before = (
            ops[i - 1].expected_product_state
            if i
            else request.start_snapshot.product_state_key
        )
        inspection = ops[-1]
        seen = report.observed_snapshots[inspection.id]
        wrong = replace(
            seen,
            assembly_xml=assembly_xml_with(request.start_snapshot.assembly_xml, before),
        )
        return replace(
            report,
            observed_snapshots={**report.observed_snapshots, inspection.id: wrong},
        )

    return fault


def with_anomaly(request, report):
    return replace(
        report,
        anomalies=(
            Anomaly(
                request.operation_plan.operations[0].id,
                "robot_1",
                ReasonCode.NEAR_LIMIT,
                "scripted",
            ),
        ),
    )


def operations_of(plan):
    return [(op.process, op.target) for op in plan.operations]


def test_infeasible_operation_bans_that_step_and_replans_from_the_same_snapshot():
    simulation = ScriptedSimulation(infeasible_at("screw"))

    record = run_to_remove_the_lid(cell_simulation=simulation)

    first, second = record.operation_plans
    assert (Process.UNSCREWING, "screw") in operations_of(first)
    assert operations_of(second) == [
        (Process.MILLING, "lid"),
        (Process.INSPECTING, None),
    ]
    assert second.start_snapshot == start_snapshot()
    assert [e.decision for e in record.entries] == [
        "replan_infeasible_operation",
        "goal_reached",
    ]
    [infeasible] = record.entries[1].feedback
    assert (
        isinstance(infeasible, InfeasibleOperation)
        and infeasible.reason_code is ReasonCode.COLLISION
    )
    assert record.status == "goal_reached"


def test_unexpected_inspection_replans_from_the_inspected_snapshot():
    simulation = ScriptedSimulation(fails_at("lid"))

    record = run_to_remove_the_lid(cell_simulation=simulation)

    inspected = record.entries[0].inspection
    assert inspected.product_state_key == ("base", "lid")
    second = record.operation_plans[1]
    assert second.start_snapshot == inspected
    assert operations_of(second) == [
        (Process.GRIPPING_AND_REMOVING, "lid"),
        (Process.INSPECTING, None),
    ]
    assert [e.decision for e in record.entries] == [
        "replan_failed_operation",
        "goal_reached",
    ]
    outcomes = record.entries[1].feedback
    first_plan = record.operation_plans[0]
    unscrew, lid = (
        next(op for op in first_plan.operations if op.target == t)
        for t in ("screw", "lid")
    )
    assert [(o.operation_id, o.succeeded) for o in outcomes] == [
        (unscrew.id, True),
        (lid.id, False),
    ]
    assert outcomes[1].observed_product_state == ("base", "lid")


def test_failed_operation_is_retried_up_to_the_retry_limit_and_then_banned():
    record = run_to_remove_the_lid(
        cell_simulation=ScriptedSimulation(fails_at("lid"), fails_at("lid"))
    )

    assert record.status == "goal_unreachable"
    assert len(record.operation_plans) == 2  # the first attempt and one retry
    assert [e.decision for e in record.entries] == [
        "replan_failed_operation",
        "replan_failed_operation",
        "goal_unreachable",
    ]


def test_a_higher_retry_limit_allows_more_attempts():
    simulation = ScriptedSimulation(fails_at("lid"), fails_at("lid"))

    record = run_to_remove_the_lid(
        cell_simulation=simulation,
        settings=RunSettings(goal=Goal(absent=("lid",)), retry_limit=2),
    )

    assert record.status == "goal_reached"
    assert len(record.operation_plans) == 3


def test_retry_limit_zero_bans_a_failed_operation_at_once():
    record = run_to_remove_the_lid(
        cell_simulation=ScriptedSimulation(fails_at("lid")),
        settings=RunSettings(goal=Goal(absent=("lid",)), retry_limit=0),
    )

    assert record.status == "goal_unreachable"
    assert len(record.operation_plans) == 1


def test_anomaly_is_logged_without_replanning():
    record = run_to_remove_the_lid(cell_simulation=ScriptedSimulation(with_anomaly))

    assert record.status == "goal_reached"
    [entry] = record.entries
    assert [
        a.reason_code for a in entry.produced_feedback if isinstance(a, Anomaly)
    ] == [ReasonCode.NEAR_LIMIT]
    assert record.to_dict()["entries"][0]["produced_feedback"][-1]["kind"] == "anomaly"


def test_run_record_lists_plans_reports_feedback_and_decisions_in_order(tmp_path):
    record = run_to_remove_the_lid(
        cell_simulation=ScriptedSimulation(infeasible_at("screw"), fails_at("lid"))
    )

    assert [e.decision for e in record.entries] == [
        "replan_infeasible_operation",
        "replan_failed_operation",
        "goal_reached",
    ]
    assert len(record.operation_plans) == len(record.simulation_reports) == 3
    first, second, third = (e.feedback for e in record.entries)
    assert first == () and second == record.entries[0].produced_feedback
    assert third == second + record.entries[1].produced_feedback
    record.save_json(tmp_path / "run_record.json")  # serialises


def test_goal_unreachable_after_replanning_stops_the_run_for_a_person():
    simulation = ScriptedSimulation(infeasible_at("screw"), infeasible_at("lid"))

    record = run_to_remove_the_lid(cell_simulation=simulation)

    assert record.status == "goal_unreachable"
    assert record.entries[-1].simulation_report is None
    assert record.entries[-1].plan_result.reason


def test_operation_ids_are_unique_across_the_run():
    record = run_to_remove_the_lid(
        cell_simulation=ScriptedSimulation(infeasible_at("screw"))
    )

    ids = [op.id for plan in record.operation_plans for op in plan.operations]
    assert len(ids) == len(set(ids))
    second = record.operation_plans[1]
    assert set(second.sequences["robot_1"]) <= {op.id for op in second.operations}


def test_planners_can_ignore_the_penalty_helper_and_read_the_raw_feedback():
    seen = []

    def own_planner(request: PlanningRequest):
        seen.append(request.feedback)
        return v1_plan(request)

    record = run_to_remove_the_lid(
        planner=own_planner, cell_simulation=ScriptedSimulation(infeasible_at("screw"))
    )

    assert seen[0] == () and isinstance(seen[1][0], InfeasibleOperation)
    assert record.status == "goal_reached"


def test_penalty_cost_rises_with_each_failure_of_a_step_and_it_is_banned_past_the_retry_limit():
    record = run_to_remove_the_lid(
        cell_simulation=ScriptedSimulation(fails_at("lid")),
        settings=RunSettings(goal=Goal(absent=("lid",)), retry_limit=2),
    )
    plan = record.operation_plans[0]
    request = PlanningRequest(
        lid_screwed_on_base_graph(),
        product_model(),
        transformer_cell(),
        record.entries[0].inspection,
        record.settings,
        record.entries[1].feedback,
        earlier_plans=(plan,),
        travel_times=travel_times(),
    )
    step = StepKey(("base", "lid"), "lid", Process.GRIPPING_AND_REMOVING, ("robot_1",))
    failure = OperationOutcome(
        next(op.id for op in plan.operations if op.target == "lid"),
        False,
        ("base", "lid"),
    )

    once = penalties(request)
    twice = penalties(replace(request, feedback=request.feedback + (failure,)))
    thrice = penalties(replace(request, feedback=request.feedback + (failure, failure)))

    assert not once.banned(step) and once.cost(8.0, step) > 8.0
    assert not twice.banned(step) and twice.cost(8.0, step) > once.cost(8.0, step)
    assert thrice.banned(step)


def test_rejected_simulation_request_stops_the_run():
    class RejectingSimulation:
        def simulate(self, request):
            return ModelLevelStandIn().simulate(None)

    record = run_to_remove_the_lid(cell_simulation=RejectingSimulation())

    assert record.status == "request_rejected"
    assert (
        record.to_dict()["entries"][0]["simulation_report"]["rejection"]["reason_code"]
        == "invalid_request"
    )


def test_a_run_on_the_real_cell_is_refused_when_the_cell_file_is_simulation_only():
    with pytest.raises(ValueError, match="simulation-only cell file"):
        run_to_remove_the_lid(
            settings=RunSettings(goal=Goal(absent=("lid",)), target="cell"),
            simulation_only=True,
        )


def test_a_run_on_the_real_cell_is_refused_before_planning_with_the_stand_in():
    asked = []

    def planner(request):
        asked.append(request)
        return v1_plan(request)

    with pytest.raises(ValueError, match="produces none"):
        run_to_remove_the_lid(
            settings=RunSettings(goal=Goal(absent=("lid",)), target="cell"),
            planner=planner,
        )
    assert asked == []


def test_a_simulation_run_goes_on_with_a_simulation_only_cell_file():
    assert run_to_remove_the_lid(simulation_only=True).status == "goal_reached"


def test_a_run_on_the_real_cell_goes_on_with_a_cell_file_that_is_not_simulation_only():
    record = run_to_remove_the_lid(
        settings=RunSettings(goal=Goal(absent=("lid",)), target="cell"),
        cell_simulation=SegmentSimulation(),
    )

    assert record.status == "goal_reached"


# --- cell-targeted runs go segment by segment ---------------------------------------------------


class SegmentSimulation(ScriptedSimulation):
    """The stand-in that returns NC programs named after the request and, like the real cell
    simulation, observed snapshots of every inspection of the request it was sent."""

    produces_nc_programs = True

    def simulate(self, request):
        report = super().simulate(request)
        n = len(self.requests)
        return replace(
            report,
            nc_programs={"CH1": {"file_name": "ch1.mpf", "text": f"segment {n}"}},
        )


class CountingPlanner:
    def __init__(self):
        self.calls = 0
        self.requests = []

    def __call__(self, request):
        self.calls += 1
        self.requests.append(request)
        return v1_plan(request)


def run_on_the_cell(simulation, planner=None, **overrides):
    return run_to_remove_the_lid(
        settings=RunSettings(goal=Goal(absent=("lid",)), target="cell"),
        cell_simulation=simulation,
        planner=planner or v1_plan,
        **overrides,
    )


def fails_at_first_inspection():
    """The request's first inspection sees the state the request started from."""

    def fault(request, report):
        ops = request.operation_plan.operations
        first = next(op for op in ops if op.process is Process.INSPECTING)
        wrong = replace(
            report.observed_snapshots[first.id],
            assembly_xml=request.start_snapshot.assembly_xml,
        )
        return replace(
            report, observed_snapshots={**report.observed_snapshots, first.id: wrong}
        )

    return fault


def test_a_cell_run_re_simulates_the_rest_of_the_plan_after_each_inspection():
    simulation, planner = SegmentSimulation(), CountingPlanner()

    record = run_on_the_cell(simulation, planner)

    [plan] = record.operation_plans
    ends = [
        i for i, op in enumerate(plan.operations) if op.process is Process.INSPECTING
    ]
    assert len(ends) == 3  # the start inspection and the two of the plan
    assert record.status == "goal_reached"
    assert planner.calls == 1
    assert len(record.simulation_reports) == 3
    assert [e.decision for e in record.entries] == [
        "resimulate_after_inspection",
        "resimulate_after_inspection",
        "goal_reached",
    ]
    assert [r.nc_programs["CH1"]["text"] for r in record.simulation_reports] == [
        "segment 1",
        "segment 2",
        "segment 3",
    ]
    assert simulation.requests[0].operation_plan == plan
    for request, previous, end in zip(simulation.requests[1:], record.entries, ends):
        assert request.start_snapshot == previous.inspection
        assert request.operation_plan.start_snapshot == previous.inspection
        assert request.operation_plan.operations == plan.operations[end + 1 :]


def test_a_re_simulation_keeps_the_ids_of_the_plan_and_only_the_sync_points_inside_it():
    simulation = SegmentSimulation()

    record = run_on_the_cell(simulation)

    [plan] = record.operation_plans
    for request in simulation.requests[1:]:
        ids = {op.id for op in request.operation_plan.operations}
        assert ids <= {op.id for op in plan.operations}
        assert all(set(seq) <= ids for seq in request.operation_plan.sequences.values())
        assert all(
            s.before in ids and s.until in ids
            for s in request.operation_plan.sync_points
        )


def test_a_simulation_run_still_rehearses_the_plan_once_and_reads_every_inspection():
    simulation = SegmentSimulation()

    record = run_to_remove_the_lid(cell_simulation=simulation)

    assert len(simulation.requests) == 1
    assert [e.decision for e in record.entries] == ["goal_reached"]
    assert "segment" not in record.to_dict()["entries"][0]


def test_an_unexpected_inspection_in_a_later_segment_replans_from_the_observed_snapshot():
    simulation = SegmentSimulation(None, fails_at_first_inspection())
    planner = CountingPlanner()

    record = run_on_the_cell(simulation, planner)

    assert planner.calls == 2
    assert record.status == "goal_reached"
    decisions = [e.decision for e in record.entries]
    assert decisions[:2] == ["resimulate_after_inspection", "replan_failed_operation"]
    first, second = record.operation_plans
    assert second.start_snapshot == record.entries[1].inspection
    # the replanned plan is run segment by segment as well
    assert decisions[2:].count("resimulate_after_inspection") >= 1
    assert decisions[-1] == "goal_reached"


def test_an_infeasible_operation_in_a_re_simulation_replans_from_the_inspected_snapshot():
    simulation = SegmentSimulation(None, None, infeasible_at("lid"))
    planner = CountingPlanner()

    record = run_on_the_cell(simulation, planner)

    decisions = [e.decision for e in record.entries]
    assert decisions[:3] == [
        "resimulate_after_inspection",
        "resimulate_after_inspection",
        "replan_infeasible_operation",
    ]
    assert planner.calls == 2
    second_inspection = record.entries[1].inspection
    assert second_inspection != record.entries[0].inspection
    assert planner.requests[1].start_snapshot == second_inspection


def test_the_replan_limit_counts_planning_rounds_not_segments():
    simulation = SegmentSimulation(None, fails_at_first_inspection())

    record = run_on_the_cell(simulation, max_rounds=1)

    assert record.status == "replan_limit_reached"
    assert len(record.operation_plans) == 1


def test_many_segments_do_not_use_up_the_replan_limit():
    record = run_on_the_cell(SegmentSimulation(), max_rounds=1)

    assert record.status == "goal_reached"
    assert len(record.simulation_reports) == 3


def test_operations_after_the_last_inspection_end_the_run_without_the_goal_observed():
    def plan_without_the_last_inspection(request):
        plan = v1_plan(request)
        ops = plan.operations[:-1]
        ids = {op.id for op in ops}
        return replace(
            plan,
            operations=ops,
            sequences={
                r: tuple(i for i in seq if i in ids)
                for r, seq in plan.sequences.items()
            },
        )

    record = run_on_the_cell(
        SegmentSimulation(), planner=plan_without_the_last_inspection
    )

    assert record.status == "goal_not_reached"
    assert len(record.simulation_reports) == 3


def test_a_re_simulation_drops_the_sync_points_of_operations_that_are_done():
    plan = run_on_the_cell(SegmentSimulation()).operation_plans[0]
    a, b, c, d = (op.id for op in plan.operations[:4])
    points = (
        SyncPoint("robot_1", d, c, "planner", "order"),  # inside the rest
        SyncPoint("robot_2", c, a, "planner", "order"),  # its partner is done
        SyncPoint(
            "robot_2", a, c, "planner", "order"
        ),  # waits before an operation that is done
    )

    rest = _rest_of(
        replace(plan, sync_points=points), plan.operations[2:], start_snapshot()
    )

    assert rest.sync_points == (points[0],)
    assert a not in {i for seq in rest.sequences.values() for i in seq}


def test_the_goal_is_reached_only_when_the_last_segments_inspection_observes_it():
    record = run_on_the_cell(SegmentSimulation())

    assert record.final_snapshot == record.entries[-1].inspection
    assert [e.decision for e in record.entries].count("goal_reached") == 1


@requires_sim
def test_against_the_real_cell_simulation_the_segment_after_the_start_inspection_holds_motion():
    start = start_snapshot()
    robots = tuple(r.id for r in transformer_cell().robots)

    def operation(op_id, process, target, tool, state):
        return Operation(
            op_id,
            process,
            target,
            (Share("robot_1", tool, path_duration=5.0),),
            expected_product_state=state,
        )

    def inspection(op_id, state):
        shares = tuple(Share(r, None, path_duration=10.0) for r in robots)
        return Operation(
            op_id, Process.INSPECTING, None, shares, expected_product_state=state
        )

    operations = (
        inspection("inspection_1", ("base", "lid", "screw")),
        operation("op_1", Process.UNSCREWING, "screw", "T6", ("base", "lid")),
        inspection("inspection_2", ("base", "lid")),
        operation("op_2", Process.GRIPPING_AND_REMOVING, "lid", "H5", ("base",)),
        inspection("inspection_3", ("base",)),
    )
    plan = OperationPlan(operations, {}, (), PlanMetrics(0.0), start)
    simulation, _ = loopback_simulation()

    record = run_to_remove_the_lid(
        settings=RunSettings(goal=Goal(absent=("lid",)), target="cell"),
        planner=lambda request: plan,
        cell_simulation=simulation,
    )

    assert record.status == "goal_reached"
    first, second, third = (
        "\n".join(p["text"] for p in r.nc_programs.values())
        for r in record.simulation_reports
    )
    assert "---- op_" not in first  # the start inspection ends the first segment
    assert "---- op_1: unscrewing screw" in second and "---- op_2" not in second
    assert "---- op_2: gripping_and_removing lid" in third
