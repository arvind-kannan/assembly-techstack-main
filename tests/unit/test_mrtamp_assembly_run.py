"""Assembly runs at the run controller seam: complete graph (reversal) + v1 planner + model-level stand-in."""

import json
from dataclasses import replace

import networkx as nx
import pytest
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState
from diss_assembly_dataclasses.feedback import InfeasibleOperation, ReasonCode
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.operation_plan import OperationPlan
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell

from disassembly_pomdp.graph_reversal import AssemblyConstants, complete_by_reversal
from disassembly_pomdp.graph_steps import Step
from mrtamp.assembly_cell import with_assembly_capabilities
from mrtamp.b1_planner import plan as b1_plan
from mrtamp.cell_simulation import SimulationRequest
from mrtamp.exported_graph import assembly_from_scratch_start, disassembled_state
from mrtamp.planning import Goal, PlanningRequest, RunSettings
from mrtamp.run_controller import run
from mrtamp.stand_in import ModelLevelStandIn
from mrtamp.step_expansion import expand_step, is_gripping_at_supply
from mrtamp.v1_planner import make_v1_planner
from mrtamp.v1_planner import plan as v1_plan
from tests.mrtamp_scenarios import (
    ALL,
    BASE_ONLY_XML,
    SUPPLY_SLOTS,
    assembly_cell,
    base_only_snapshot,
    lid_screwed_on_base_graph,
    product_model,
    start_snapshot,
    travel_times,
    two_robot_cell,
)

CONSTANTS = AssemblyConstants(
    screw_in_time=4.0,
    supply_travel_time=6.0,
    insert_approach_time=2.0,
    insert_feed_time=3.0,
)


def complete_graph():
    return complete_by_reversal(lid_screwed_on_base_graph(), CONSTANTS)


def stand_in():
    return ModelLevelStandIn()


def assemble(**overrides):
    kwargs = dict(
        sequence_graph=complete_graph(),
        product_model=product_model(),
        cell=assembly_cell(),
        start_snapshot=base_only_snapshot(),
        settings=RunSettings(goal=Goal(present=ALL)),
        planner=v1_plan,
        cell_simulation=stand_in(),
        supply_slots=SUPPLY_SLOTS,
        travel_times=travel_times(),
    )
    kwargs.update(overrides)
    return run(**kwargs)


LID_ON_BASE_XML = """<assembly name="lid_on_base">
	<part name="base" mesh_path="base.stl" density="8000.0" material="Generic" />
	<part name="lid" mesh_path="lid.stl" density="8000.0" material="Generic" />
	<connection type="generic" part_name1="base" part_name2="lid" />
</assembly>"""


def assembly_request(**overrides):
    kwargs = dict(
        sequence_graph=complete_graph(),
        product_model=product_model(),
        cell=assembly_cell(),
        start_snapshot=base_only_snapshot(),
        settings=RunSettings(goal=Goal(present=ALL)),
        supply_slots=SUPPLY_SLOTS,
        travel_times=travel_times(),
    )
    kwargs.update(overrides)
    return PlanningRequest(**kwargs)


def test_assembly_run_goes_from_the_base_parts_alone_to_the_full_product_model():
    record = assemble()

    assert record.status == "goal_reached"
    plan = record.operation_plans[-1]
    assert [(op.process, op.target) for op in plan.operations] == [
        (Process.GRIPPING, "lid"),
        (Process.INSERTING, "lid"),
        (Process.RELEASING, "lid"),
        (Process.INSPECTING, None),  # screwing in depends on the uninspected inserting
        (Process.SCREWING_IN, "screw"),
        (Process.INSPECTING, None),
    ]
    assert record.final_snapshot.product_state_key == ALL
    assert record.final_snapshot.cell_state.holds == {}


def test_the_run_refuses_a_start_snapshot_whose_product_state_is_not_in_the_graph():
    stray = Cell_Assembly_Snapshot(
        '<assembly name="stray"><part name="lid" mesh_path="lid.stl" density="8000.0" material="Generic" /></assembly>',
        CellState(mounted_tools={"robot_1": "H5", "robot_2": None}),
    )

    with pytest.raises(ValueError, match="not a product state of the sequence graph"):
        assemble(start_snapshot=stray)


def test_an_assembly_from_scratch_starts_at_the_base_parts_which_must_be_a_node_of_the_graph():
    assert assembly_from_scratch_start(complete_graph(), ("base",)) == ("base",)
    with pytest.raises(ValueError, match="full teardown"):
        assembly_from_scratch_start(complete_graph(), ("lid",))


def test_a_reassembly_starts_at_the_one_disassembled_state_of_the_graph():
    graph = complete_graph()
    nx.set_node_attributes(graph, {n: n == ("base", "screw") for n in graph}, "is_goal")

    assert disassembled_state(graph) == ("base", "screw")


@pytest.mark.parametrize("goals", [(), (("base",), ("base", "screw"))])
def test_without_exactly_one_disassembled_state_the_start_is_refused(goals):
    graph = complete_graph()
    nx.set_node_attributes(graph, {n: n in goals for n in graph}, "is_goal")

    with pytest.raises(ValueError, match=f"{len(goals)} disassembled states"):
        disassembled_state(graph)


def test_assembly_plan_metrics_add_the_assembly_constants_and_the_tool_change():
    record = assemble()

    # insert lid 8+6+2+3 s with H5, inspection 10 s, tool change H5 -> T6 80 s, screw in 5+4 s, inspection 10 s
    metrics = record.operation_plans[-1].metrics
    assert metrics.makespan == 19.0 + 10.0 + 80.0 + 9.0 + 10.0
    assert metrics.tool_changes == 1


def test_the_run_controller_never_reverses_the_graph_itself():
    forward_only = lid_screwed_on_base_graph()
    edges_before = forward_only.number_of_edges()

    record = assemble(sequence_graph=forward_only)

    assert record.status == "goal_unreachable"
    assert forward_only.number_of_edges() == edges_before


def test_a_disassembly_run_does_not_use_inverse_steps():
    record = run(
        sequence_graph=complete_graph(),
        product_model=product_model(),
        cell=assembly_cell(),
        start_snapshot=start_snapshot(),
        settings=RunSettings(goal=Goal(absent=("lid",))),
        planner=v1_plan,
        cell_simulation=ModelLevelStandIn(),
        travel_times=travel_times(),
    )

    assert record.status == "goal_reached"
    assert [op.process for op in record.operation_plans[-1].operations] == [
        Process.UNSCREWING,
        Process.INSPECTING,
        Process.GRIPPING_AND_REMOVING,
        Process.INSPECTING,
    ]


def test_goal_unreachable_when_the_cell_file_has_no_supply_slot_for_a_part():
    record = assemble(supply_slots=())

    assert record.status == "goal_unreachable"


def test_cell_state_in_the_snapshot_shows_the_held_part_between_gripping_and_releasing():
    plan = v1_plan(assembly_request())
    assert isinstance(plan, OperationPlan)
    grip, insert, release, *_ = plan.operations
    cell = assembly_cell()
    probe = Operation(
        "probe",
        Process.INSPECTING,
        None,
        tuple(Share(r.id, None) for r in cell.robots),
        expected_product_state=("base",),
    )

    report = stand_in().simulate(
        SimulationRequest(
            replace(plan, operations=(grip, probe, insert, release)),
            base_only_snapshot(),
            cell,
            product_model=product_model(),
            travel_times=travel_times(),
        )
    )

    assert report.infeasible is None
    assert report.observed_snapshots["probe"].cell_state.holds == {
        grip.robots[0]: "lid"
    }


def test_inserting_a_part_that_is_not_held_is_a_cell_state_mismatch():
    plan = v1_plan(assembly_request())

    report = stand_in().simulate(
        SimulationRequest(
            replace(plan, operations=plan.operations[1:]),
            base_only_snapshot(),
            assembly_cell(),
            product_model=product_model(),
            travel_times=travel_times(),
        )
    )

    assert report.infeasible.operation_id == plan.operations[1].id
    assert report.infeasible.reason_code.value == "cell_state_mismatch"


def test_step_expansion_is_a_helper_a_planner_may_replace():
    inverse = Step.from_edge(
        next(
            s
            for _, _, s in complete_graph().edges(data=True)
            if s["process"] == "inserting"
        )
    )

    ops = expand_step(inverse, SUPPLY_SLOTS)
    assert [o.process for o in ops] == [
        Process.GRIPPING,
        Process.INSERTING,
        Process.RELEASING,
    ]
    assert sum(o.duration for o in ops) == inverse.action_time

    def inserting_only(step, supply_slots=()):
        ops = expand_step(step, SUPPLY_SLOTS)
        return [o for o in ops if o.process is Process.INSERTING] or ops

    plan = v1_plan(assembly_request(), expand=inserting_only)
    assert [op.process for op in plan.operations] == [
        Process.INSERTING,
        Process.INSPECTING,
        Process.SCREWING_IN,
        Process.INSPECTING,
    ]


def test_run_record_of_an_assembly_run_is_json(tmp_path):
    record = assemble()
    path = tmp_path / "run_record.json"

    record.save_json(path)

    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["status"] == "goal_reached" and d["settings"]["goal"]["present"] == list(
        ALL
    )
    assert d["product_folder"] == str(
        product_model().folder
    )  # recorded runs stay readable


def test_step_expansion_refuses_a_supply_time_longer_than_the_step():
    step = Step.from_edge(
        {
            "process": "inserting",
            "tool": "H5",
            "target_part": "lid",
            "action_time": 3.0,
            "supply_time": 6.0,
        }
    )

    with pytest.raises(ValueError):
        expand_step(step, SUPPLY_SLOTS)


def test_a_run_that_removes_and_inserts_parts_reaches_its_mixed_goal():
    # lid on base, screw out: the screw is inserted (an inverse step), then the lid is milled off (a removing step)
    start = Cell_Assembly_Snapshot(
        LID_ON_BASE_XML, CellState(mounted_tools={"robot_1": "H5", "robot_2": None})
    )

    record = run(
        sequence_graph=complete_graph(),
        product_model=product_model(),
        cell=assembly_cell(),
        start_snapshot=start,
        settings=RunSettings(goal=Goal(present=("screw",), absent=("lid",))),
        planner=v1_plan,
        cell_simulation=stand_in(),
        travel_times=travel_times(),
    )

    assert record.status == "goal_reached"
    assert record.final_snapshot.product_state_key == ("base", "screw")
    processes = [
        op.process
        for op in record.operation_plans[-1].operations
        if op.process is not Process.INSPECTING
    ]
    assert processes == [Process.SCREWING_IN, Process.MILLING]


# --- replanning an assembly run -----------------------------------------------------------------


class RefusingSupply:
    """The stand-in, except that every gripping at the supply is out of the robot's reach."""

    def __init__(self):
        self.calls = 0

    def simulate(self, request):
        self.calls += 1
        report = stand_in().simulate(request)
        ops = request.operation_plan.operations
        i = next((i for i, op in enumerate(ops) if is_gripping_at_supply(op)), None)
        if i is None:
            return report
        reason = InfeasibleOperation(
            ops[i].id, ops[i].robots[0], ReasonCode.OUT_OF_REACH, "unreachable slot"
        )
        return replace(
            report,
            feasible_operations=tuple(op.id for op in ops[:i]),
            infeasible=reason,
            observed_snapshots={},
            outcomes=(),
        )


class LidNeverGoesIn:
    """The stand-in, except that the inserting of the lid never takes effect: the inspection after it sees the base."""

    def __init__(self):
        self.calls = 0

    def simulate(self, request):
        self.calls += 1
        report = stand_in().simulate(request)
        ops = request.operation_plan.operations
        inspection = next(
            (
                op
                for op in ops
                if op.process is Process.INSPECTING
                and "lid" in op.expected_product_state
            ),
            None,
        )
        if inspection is None or inspection.id not in report.observed_snapshots:
            return report
        seen = report.observed_snapshots[inspection.id]
        return replace(
            report,
            observed_snapshots={
                **report.observed_snapshots,
                inspection.id: replace(seen, assembly_xml=BASE_ONLY_XML),
            },
        )


def both_robots_assemble() -> Cell:
    cell = with_assembly_capabilities(two_robot_cell())
    first, second = cell.robots
    return replace(
        cell, robots=(first, replace(second, capabilities=first.capabilities))
    )


@pytest.mark.parametrize(
    "planner, cell, assembling_robots",
    [(v1_plan, assembly_cell(), 1), (b1_plan, both_robots_assemble(), 2)],
    ids=["v1", "b1"],
)
def test_an_assembly_run_whose_supply_slot_no_robot_can_reach_ends_goal_unreachable(
    planner, cell, assembling_robots
):
    simulation = RefusingSupply()

    record = assemble(planner=planner, cell=cell, cell_simulation=simulation)

    assert record.status == "goal_unreachable"
    assert (
        simulation.calls == assembling_robots
    )  # each robot's gripping is refused once


@pytest.mark.parametrize("planner", [v1_plan, b1_plan], ids=["v1", "b1"])
def test_a_failed_inserting_is_retried_up_to_the_retry_limit_and_then_excluded(planner):
    simulation = LidNeverGoesIn()

    record = assemble(
        planner=planner,
        cell_simulation=simulation,
        settings=RunSettings(goal=Goal(present=ALL), retry_limit=1),
    )

    assert record.status == "goal_unreachable"
    assert simulation.calls == 2  # the first try and one retry


def test_each_plan_function_of_an_assembly_run_gets_the_parts_inserted_so_far():
    snapshots = []

    def recording(product_model, snapshot, operation, robot, tool):
        snapshots.append((operation, snapshot))
        return operation

    record = assemble(
        planner=make_v1_planner(
            {Process.INSERTING: recording, Process.SCREWING_IN: recording}
        )
    )

    assert record.status == "goal_reached"
    assert [op.process for op, _ in snapshots] == [
        Process.INSERTING,
        Process.SCREWING_IN,
    ]
    for operation, snapshot in snapshots:
        assert snapshot.product_state_key == operation.step.before
    assert "lid" in snapshots[-1][1].product_state_key  # inserted earlier in the run
