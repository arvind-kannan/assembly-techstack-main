"""Loading, unloading, reconfiguring and machine operations at model level, on the Transformer cell file (AT 277):
a hand-written operation plan through the DMG, and each reason an operation on the cell state is infeasible."""

from dataclasses import replace
from pathlib import Path

import pytest
from diss_assembly_dataclasses.cell_state import (
    Cell_Assembly_Snapshot,
    CellState,
    Location,
)
from diss_assembly_dataclasses.feedback import InfeasibleOperation, ReasonCode
from diss_assembly_dataclasses.operation import (
    MachineOperationParameters,
    Operation,
    PalletTransferParameters,
    ReconfiguringParameters,
    Share,
)
from diss_assembly_dataclasses.operation_plan import OperationPlan, PlanMetrics
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import (
    CouplingPlace,
    Interface,
    Station,
    StationKind,
    Tool,
    Tooling,
)

from disassembly_pomdp.travel_time_table import TravelTimeTable
from mrtamp.cell_file import load_cell_file, with_station
from mrtamp.cell_operations import (
    REWORK_MILLING_CYCLE_TIME,
    apply_cell_operation,
    cycle_time,
)
from mrtamp.cell_simulation import SimulationRequest
from mrtamp.stand_in import ModelLevelStandIn
from tests.mrtamp_scenarios import START_XML, product_model

CELL_FILE = Path(__file__).resolve().parents[2] / "cases" / "transformer_cell.json"
ON_STORAGE = Location.at("pallet_storage", "nsr_a")
ON_DMG = Location.at("dmg", "table")
IN_HAND = Location.held_by("robot_1")


def cell_file():
    return load_cell_file(CELL_FILE)


def start_state() -> CellState:
    f = cell_file()
    return CellState(mounted_tools=f.mounted_tools, locations=f.locations)


def transfer(
    op_id,
    process,
    location: Location,
    tool="pizzagreifer",
    robot="robot_1",
    pallet="pallet",
):
    return Operation(
        op_id,
        process,
        None,
        (Share(robot, tool, path_duration=20.0),),
        parameters=PalletTransferParameters(
            pallet, location.station, location.coupling_place
        ).to_dict(),
    )


def machine_operation(op_id="mill", station="dmg", cycle=None):
    return Operation(
        op_id,
        Process.REWORK_MILLING,
        None,
        (Share("robot_1", None, path_duration=1.0),),
        parameters=MachineOperationParameters(station, cycle).to_dict(),
    )


def the_plan():
    return (
        transfer("unload_storage", Process.UNLOADING, ON_STORAGE),
        transfer("load_dmg", Process.LOADING, ON_DMG),
        machine_operation(cycle=600.0),
        transfer("unload_dmg", Process.UNLOADING, ON_DMG),
        transfer("load_storage", Process.LOADING, ON_STORAGE),
    )


def test_the_cell_file_has_the_dmg_the_pallet_storage_the_pallet_and_the_pizzagreifer():
    f = cell_file()
    dmg, storage = f.cell.station("dmg"), f.cell.station("pallet_storage")

    assert dmg.kind is StationKind.MACHINE and dmg.processes == (
        Process.REWORK_MILLING,
    )
    assert [c.accepts for c in dmg.coupling_places] == [Interface.ZERO_POINT_CLAMPING]
    assert [c.accepts for c in storage.coupling_places] == [Interface.NSR_A]
    assert [p.name for p in f.cell.pallets] == ["pallet"]
    assert (
        Interface.NSR_A
        in next(t for t in f.cell.tools if t.name == "pizzagreifer").grips
    )
    assert "pizzagreifer" in f.cell.robots[0].tools
    assert f.locations == {"pallet": ON_STORAGE}
    assert "estimated" in f.notes["dmg"] and "placeholder" in f.notes["pallet_storage"]


def test_the_dmg_coupling_place_is_the_taught_set_down_point_in_the_cell_frame():
    from scipy.spatial.transform import Rotation

    dmg = cell_file().cell.station("dmg")
    rotation = Rotation.from_euler("xyz", dmg.pose[3:])
    point = rotation.apply(dmg.coupling_places[0].pose[:3]) + dmg.pose[:3]

    assert point == pytest.approx((0.706583, 2.892454, 0.911799), abs=1e-3)


def test_a_hand_written_plan_moves_the_pallet_into_the_dmg_and_back():
    cell, state = cell_file().cell, start_state()
    expected = {
        "unload_storage": {"pallet": IN_HAND},
        "load_dmg": {"pallet": ON_DMG},
        "mill": {"pallet": ON_DMG},
        "unload_dmg": {"pallet": IN_HAND},
        "load_storage": {"pallet": ON_STORAGE},
    }

    for op in the_plan():
        state = apply_cell_operation(op, state, cell)
        assert isinstance(state, CellState), state
        assert state.locations == expected[op.id], op.id
        assert (
            state.mounted_tools["robot_1"] == "pizzagreifer"
        )  # mounted by the first unloading
    assert state.holds == {}


def plan_request(operations, cell_state: CellState, travel_times=TravelTimeTable()):
    snapshot = Cell_Assembly_Snapshot(START_XML, cell_state)
    plan = OperationPlan(
        tuple(operations),
        {"robot_1": tuple(op.id for op in operations)},
        (),
        PlanMetrics(0.0),
        snapshot,
    )
    return SimulationRequest(
        plan,
        snapshot,
        cell_file().cell,
        product_model=product_model(),
        travel_times=travel_times,
    )


def inspection(op_id):
    return Operation(
        op_id,
        Process.INSPECTING,
        None,
        (Share("robot_2", "zivid"),),
        expected_product_state=("base", "lid", "screw"),
    )


def test_the_stand_in_runs_the_plan_and_shows_the_cell_state_at_each_inspection():
    operations = []
    for op in the_plan():
        operations += [op, inspection(f"look_after_{op.id}")]

    report = ModelLevelStandIn().simulate(plan_request(operations, start_state()))

    assert report.ok and report.infeasible is None
    assert len(report.feasible_operations) == 10
    seen = {
        k.removeprefix("look_after_"): s.cell_state.locations["pallet"]
        for k, s in report.observed_snapshots.items()
    }
    assert seen == {
        "unload_storage": IN_HAND,
        "load_dmg": ON_DMG,
        "mill": ON_DMG,
        "unload_dmg": IN_HAND,
        "load_storage": ON_STORAGE,
    }
    # robot_1 mounts the pizzagreifer once, before the first unloading
    robot_1 = [e for e in report.timeline if e["robot"] == "robot_1"]
    assert [e["tool"] for e in robot_1 if e["kind"] == "tool_change"] == [
        "pizzagreifer"
    ]
    assert robot_1[0]["kind"] == "tool_change"
    assert robot_1[1]["operation_id"] == "unload_storage"
    # robot_2 mounts the zivid once for the inspections
    assert report.metrics.tool_changes == 2
    # the product state is left alone by operations on the cell state alone
    assert all(
        o.observed_product_state == ("base", "lid", "screw") for o in report.outcomes
    )
    assert all(
        s.product_state_key == ("base", "lid", "screw")
        for s in report.observed_snapshots.values()
    )


def test_the_machine_operation_lasts_its_cycle_time_else_the_skill_default():
    with_default = TravelTimeTable({REWORK_MILLING_CYCLE_TIME: 900.0})

    assert cycle_time(machine_operation(cycle=600.0), with_default) == 600.0
    assert cycle_time(machine_operation(), with_default) == 900.0
    assert cycle_time(machine_operation(), TravelTimeTable()) is None

    on_dmg = start_state().with_location("pallet", ON_DMG)
    report = ModelLevelStandIn().simulate(
        plan_request((machine_operation(),), on_dmg, with_default)
    )
    assert report.metrics.makespan == 900.0


def infeasible(op, state, cell=None) -> InfeasibleOperation:
    after = apply_cell_operation(op, state, cell or cell_file().cell)
    assert isinstance(after, InfeasibleOperation), after
    return after


def with_tool(cell, tool: Tool):
    robots = tuple(
        replace(r, tools=(*r.tools, tool.name)) if r.id == "robot_1" else r
        for r in cell.robots
    )
    return replace(cell, robots=robots, tools=(*cell.tools, tool))


@pytest.mark.parametrize(
    "op, state, cell, code, text",
    [
        pytest.param(
            transfer("x", Process.UNLOADING, ON_STORAGE, tool="H5", robot="robot_2"),
            "start",
            None,
            ReasonCode.UNSUPPORTED_PROCESS,
            "robot_2 cannot carry out unloading",
            id="no-capability",
        ),
        pytest.param(
            transfer("x", Process.UNLOADING, ON_STORAGE, tool="default_gripper"),
            "start",
            None,
            ReasonCode.UNSUPPORTED_PROCESS,
            "tool default_gripper grips none of the interfaces of pallet",
            id="tool-does-not-grip",
        ),
        pytest.param(
            transfer("x", Process.LOADING, Location.at("tool_storage", "pizzagreifer")),
            "in_hand",
            None,
            ReasonCode.CELL_STATE_MISMATCH,
            "accepts sws, which pallet does not carry",
            id="place-does-not-accept",
        ),
        pytest.param(
            transfer("x", Process.LOADING, ON_DMG, pallet="pallet_2"),
            "two_pallets",
            "two_pallets",
            ReasonCode.CELL_STATE_MISMATCH,
            "coupling place table of dmg is occupied by pallet",
            id="place-occupied",
        ),
        pytest.param(
            transfer("x", Process.LOADING, ON_DMG),
            "start",
            None,
            ReasonCode.CELL_STATE_MISMATCH,
            "pallet is not at robot_1",
            id="pallet-not-in-hand",
        ),
        pytest.param(
            transfer("x", Process.UNLOADING, ON_DMG),
            "start",
            None,
            ReasonCode.CELL_STATE_MISMATCH,
            "pallet is not at coupling place table of dmg",
            id="pallet-not-there",
        ),
        pytest.param(
            machine_operation(),
            "start",
            None,
            ReasonCode.CELL_STATE_MISMATCH,
            "no pallet on a coupling place of dmg",
            id="machine-without-pallet",
        ),
        pytest.param(
            machine_operation(station="pallet_storage"),
            "start",
            None,
            ReasonCode.UNSUPPORTED_PROCESS,
            "pallet_storage is not a machine of the cell",
            id="not-a-machine",
        ),
        pytest.param(
            machine_operation(),
            "start",
            "dmg_without_rework",
            ReasonCode.UNSUPPORTED_PROCESS,
            "dmg does not offer rework_milling",
            id="machine-does-not-offer",
        ),
    ],
)
def test_an_infeasible_operation_on_the_cell_state_has_its_reason(
    op, state, cell, code, text
):
    f = cell_file()
    states = {
        "start": start_state(),
        "in_hand": start_state().with_location("pallet", IN_HAND),
        "two_pallets": start_state()
        .with_location("pallet", ON_DMG)
        .with_location("pallet_2", IN_HAND),
    }
    cells = {
        None: f.cell,
        "two_pallets": replace(
            f.cell,
            pallets=(*f.cell.pallets, replace(f.cell.pallets[0], name="pallet_2")),
        ),
        "dmg_without_rework": with_station(
            f.cell, replace(f.cell.station("dmg"), processes=())
        ),
    }

    result = infeasible(op, states[state], cells[cell])

    assert result.reason_code is code
    assert text in result.text


def test_reconfiguring_moves_a_tooling_into_the_gripper_and_checks_where_it_goes():
    f = cell_file()
    desk = Station(
        "clamping_desk",
        StationKind.FIXTURE,
        (0.0,) * 6,
        coupling_places=(
            CouplingPlace("jaw_rack", (0.0,) * 6, Interface.FINGER_INTERFACE),
        ),
    )
    cell = replace(
        with_station(f.cell, desk),
        tooling=(Tooling("jaws", (Interface.NSR_A, Interface.FINGER_INTERFACE)),),
    )
    cell = with_tool(cell, Tool("parallel_gripper", Process.GRIPPING))
    on_rack = start_state().with_location(
        "jaws", Location.at("clamping_desk", "jaw_rack")
    )
    op = Operation(
        "reconf",
        Process.RECONFIGURING,
        None,
        (Share("robot_1", "pizzagreifer"),),
        parameters=ReconfiguringParameters(
            "jaws",
            Location.at("clamping_desk", "jaw_rack"),
            Location.in_tool("parallel_gripper"),
        ).to_dict(),
    )

    after = apply_cell_operation(op, on_rack, cell)
    assert after.locations["jaws"] == Location.in_tool("parallel_gripper")

    occupied = on_rack.with_location("old_jaws", Location.in_tool("parallel_gripper"))
    assert (
        "parallel_gripper is occupied by old_jaws"
        in infeasible(op, occupied, cell).text
    )


def test_the_stand_in_reports_the_first_infeasible_cell_operation_of_a_plan():
    plan = (the_plan()[0], machine_operation())

    report = ModelLevelStandIn().simulate(plan_request(plan, start_state()))

    assert report.feasible_operations == ("unload_storage",)
    assert report.infeasible.reason_code is ReasonCode.CELL_STATE_MISMATCH


def test_check_feasibility_applies_the_cell_operation_checks():
    snapshot = Cell_Assembly_Snapshot(START_XML, start_state())
    answer = ModelLevelStandIn().check_feasibility(
        (machine_operation(),),
        snapshot,
        cell_file().cell,
        product_model=product_model(),
        travel_times=TravelTimeTable(),
    )

    assert not answer.feasible
    assert answer.infeasible.text == "no pallet on a coupling place of dmg"
