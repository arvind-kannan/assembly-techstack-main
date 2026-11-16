# crc_diss_assembly_dataclasses/tests/test_ppr_types.py
"""The PPR contract types round-trip through to_dict/from_dict as plain data."""

import json

from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState
from diss_assembly_dataclasses.feedback import Anomaly, InfeasibleOperation, OperationOutcome, ReasonCode, feedback_from_dict
from diss_assembly_dataclasses.operation import Operation, Share, StepRef
from diss_assembly_dataclasses.operation_plan import OperationPlan, PlanMetrics, SyncPoint
from diss_assembly_dataclasses.process import Process
import pytest

from diss_assembly_dataclasses.resource import (
    Cell,
    CouplingPlace,
    Interface,
    Pallet,
    Robot,
    Station,
    StationKind,
    SupplySlot,
    Tool,
    Tooling,
)


def through_json(d):
    return json.loads(json.dumps(d))


def test_process_is_a_closed_enum_that_includes_inspecting():
    assert Process("inspecting") is Process.INSPECTING
    assert Process("gripping_and_removing") is Process.GRIPPING_AND_REMOVING


def test_cell_with_robots_and_tools_round_trips():
    cell = Cell(
        robots=(
            Robot("robot_1", nc_channel=1, tools=("H5",), capabilities=(Process.GRIPPING_AND_REMOVING,)),
            Robot("robot_2", nc_channel=2, tools=(), capabilities=(Process.INSPECTING,)),
        ),
        tools=(Tool("H5", Process.GRIPPING_AND_REMOVING),),
        stations=(Station("tool_storage", StationKind.TOOL_STORAGE, ORIGIN),),
    )

    assert Cell.from_dict(through_json(cell.to_dict())) == cell


def test_the_cell_holds_no_times():
    """Times are in the cell's travel-time table, not in the cell (ADR-0012)."""
    assert "timing" not in Cell(robots=()).to_dict()


ORIGIN = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def the_supply() -> Station:
    return Station(
        "supply",
        StationKind.SUPPLY,
        (-1.2, -2.6, 0.8, 0.0, 0.0, 1.5708),
        slots=(
            SupplySlot("lid_slot", (0.4, 0.0, 0.0, 0.0, 0.0, 0.0), part="lid"),
            SupplySlot("nest_h5", (0.0, 0.2, 0.0, 0.0, 0.0, 0.0), screw_type="H5", robots=("robot_2",)),
        ),
    )


def the_mill() -> Station:
    return Station(
        "dmg_dmu_60",
        StationKind.MACHINE,
        (2.0, 0.5, 0.9, 0.0, 0.0, 3.1416),
        processes=(Process.MILLING,),
        coupling_places=(CouplingPlace("table", (0.0, 0.0, 0.1, 0.0, 0.0, 0.0), Interface.ZERO_POINT_CLAMPING),),
    )


def test_station_kind_is_a_closed_enum_with_machines_and_process_modules():
    assert StationKind("process_module") is StationKind.PROCESS_MODULE
    assert [k.value for k in StationKind] == [
        "tool_storage", "supply", "dropoff", "fixture", "machine", "process_module",
    ]


def test_supply_station_with_its_slots_round_trips():
    supply = the_supply()
    assert Station.from_dict(through_json(supply.to_dict())) == supply
    assert supply.slots[1].serves("robot_2") and not supply.slots[1].serves("robot_1")
    assert supply.slots[0].serves("robot_1")


def test_machine_with_its_processes_and_a_coupling_place_round_trips():
    mill = the_mill()
    assert Station.from_dict(through_json(mill.to_dict())) == mill


def test_only_a_supply_station_has_supply_slots():
    with pytest.raises(ValueError, match="supply slots"):
        Station("dropoff", StationKind.DROPOFF, ORIGIN, slots=the_supply().slots)


def test_only_machines_and_process_modules_offer_processes():
    with pytest.raises(ValueError, match="processes"):
        Station("fixture", StationKind.FIXTURE, ORIGIN, processes=(Process.HOLDING,))
    press = Station("bearing_press", StationKind.PROCESS_MODULE, ORIGIN, processes=(Process.INSERTING,))
    assert press.processes == (Process.INSERTING,)


def test_a_pose_has_six_numbers():
    with pytest.raises(ValueError, match="pose"):
        Station("supply", StationKind.SUPPLY, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="pose"):
        SupplySlot("lid_slot", (0.0, 0.0, 0.0, 0.0, 0.0), part="lid")


def test_cell_with_stations_pallets_and_tooling_round_trips():
    cell = Cell(
        robots=(Robot("robot_1", nc_channel=1, tools=("pizzagreifer",), capabilities=(Process.GRIPPING,)),),
        tools=(Tool("pizzagreifer", Process.GRIPPING, grips=(Interface.NSR_A,)),),
        stations=(
            Station(
                "tool_storage",
                StationKind.TOOL_STORAGE,
                (0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
                coupling_places=(CouplingPlace("place_1", ORIGIN, Interface.SWS),),
            ),
            the_supply(),
            the_mill(),
        ),
        pallets=(Pallet("lid_pallet", (Interface.NSR_A, Interface.ZERO_POINT_CLAMPING)),),
        tooling=(Tooling("press_tool_bearing", (Interface.NSR_A,)), Tooling("gripper_jaws", (Interface.FINGER_INTERFACE,))),
    )

    assert Cell.from_dict(through_json(cell.to_dict())) == cell
    assert cell.station("supply") == the_supply()
    assert cell.station("no_such") is None


ASSEMBLY_XML = """<assembly name="lid_on_base">
	<part name="base" mesh_path="base.stl" density="8000.0" material="Generic" />
	<part name="lid" mesh_path="lid.stl" density="8000.0" material="Generic" />
	<connection type="generic" part_name1="base" part_name2="lid" />
</assembly>"""


def test_cell_assembly_snapshot_round_trips_with_the_assembly_as_xml_text():
    snapshot = Cell_Assembly_Snapshot(
        assembly_xml=ASSEMBLY_XML,
        cell_state=CellState(holds={"robot_1": "lid"}, mounted_tools={"robot_1": "H5", "robot_2": None}),
    )

    d = through_json(snapshot.to_dict())

    assert d["assembly_xml"] == ASSEMBLY_XML
    assert Cell_Assembly_Snapshot.from_dict(d) == snapshot


def test_snapshot_names_its_product_state_by_the_parts_present():
    snapshot = Cell_Assembly_Snapshot(ASSEMBLY_XML, CellState())
    assert snapshot.product_state_key == ("base", "lid")


def removing_the_lid() -> Operation:
    return Operation(
        id="op_1",
        process=Process.GRIPPING_AND_REMOVING,
        target="lid",
        shares=(Share(robot="robot_1", tool="H5", path=((0.0, 0.0, 10.0, 0.0, 0.0, 0.0), (0.0, 0.0, 60.0, 0.0, 0.0, 0.0)), path_duration=18.5),),
        direction=(0.0, 0.0, 1.0),
        expected_product_state=("base",),
    )


def test_operation_with_one_share_per_robot_round_trips():
    op = removing_the_lid()
    assert Operation.from_dict(through_json(op.to_dict())) == op


def test_inspection_is_an_operation_over_all_robots():
    inspection = Operation(
        id="inspection_1",
        process=Process.INSPECTING,
        target=None,
        shares=(Share("robot_1", None, path_duration=10.0), Share("robot_2", None, path_duration=10.0)),
        expected_product_state=("base",),
    )
    assert Operation.from_dict(through_json(inspection.to_dict())) == inspection
    assert inspection.robots == ("robot_1", "robot_2")


def test_operation_names_the_sequence_graph_step_it_carries_out_and_round_trips():
    gripping = Operation(
        id="op_1",
        process=Process.GRIPPING,
        target="lid",
        shares=(Share("robot_1", "H5"),),
        expected_product_state=("base",),
        step=StepRef(before=("base",), target="lid", process=Process.INSERTING),
    )
    assert Operation.from_dict(through_json(gripping.to_dict())) == gripping


def test_an_operation_without_a_step_keeps_its_dict_without_the_key():
    op = removing_the_lid()
    assert "step" not in op.to_dict()
    assert Operation.from_dict({**op.to_dict(), "step": None}) == op


def test_operation_plan_with_sync_points_and_plan_metrics_round_trips():
    op = removing_the_lid()
    plan = OperationPlan(
        operations=(op,),
        sequences={"robot_1": ("op_1",), "robot_2": ()},
        sync_points=(SyncPoint(robot="robot_2", before="op_2", until="op_1", layer="planner", reason="order"),),
        metrics=PlanMetrics(makespan=18.5, busy_time={"robot_1": 18.5, "robot_2": 0.0}, tool_changes=0, inspections=0),
        start_snapshot=Cell_Assembly_Snapshot(ASSEMBLY_XML, CellState(mounted_tools={"robot_1": "H5"})),
    )

    assert OperationPlan.from_dict(through_json(plan.to_dict())) == plan
    assert plan.operation("op_1") == op


def test_feedback_round_trips_with_reference_fields_only():
    feedback = [
        OperationOutcome(operation_id="op_1", succeeded=False, observed_product_state=("base", "lid"),
                         reason_code=ReasonCode.CONTACT_ABORT, text="force limit"),
        InfeasibleOperation(operation_id="op_2", robot="robot_1", reason_code=ReasonCode.CELL_STATE_MISMATCH,
                            text="lid is not in the assembly"),
        Anomaly(operation_id="op_3", robot="robot_2", reason_code=ReasonCode.NEAR_LIMIT, text="clearance 2 mm"),
    ]

    assert [feedback_from_dict(through_json(f.to_dict())) for f in feedback] == feedback
