# crc_diss_assembly_dataclasses/tests/test_machine_operations_and_items.py
"""Machine operations, loading, unloading and reconfiguring with their parameters, and where each pallet and
tooling is in the cell state (AT 277)."""

import json

import pytest

from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState, Location
from diss_assembly_dataclasses.operation import (
    MachineOperationParameters,
    Operation,
    PalletTransferParameters,
    ReconfiguringParameters,
    Share,
    parameters_of,
)
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Station, StationKind


def through_json(d):
    return json.loads(json.dumps(d))


def test_rework_milling_loading_unloading_and_reconfiguring_are_processes():
    assert Process("rework_milling") is Process.REWORK_MILLING
    assert Process("loading") is Process.LOADING
    assert Process("unloading") is Process.UNLOADING
    assert Process("reconfiguring") is Process.RECONFIGURING
    assert Process.REWORK_MILLING is not Process.MILLING


def test_a_location_is_a_coupling_place_a_robot_or_a_tool():
    assert Location.at("dmg", "table").to_dict() == {"station": "dmg", "coupling_place": "table"}
    assert Location.held_by("robot_1").to_dict() == {"robot": "robot_1"}
    assert Location.in_tool("pizzagreifer").to_dict() == {"tool": "pizzagreifer"}
    for location in (Location.at("dmg", "table"), Location.held_by("robot_1"), Location.in_tool("pizzagreifer")):
        assert Location.from_dict(through_json(location.to_dict())) == location


@pytest.mark.parametrize(
    "fields",
    [{}, {"station": "dmg"}, {"coupling_place": "table"}, {"robot": "robot_1", "tool": "pizzagreifer"},
     {"station": "dmg", "coupling_place": "table", "robot": "robot_1"}],
)
def test_a_location_is_exactly_one_of_them(fields):
    with pytest.raises(ValueError, match="exactly one"):
        Location(**fields)


def test_machine_operation_names_its_station_and_cycle_time():
    op = Operation(
        "mill_1", Process.REWORK_MILLING, "BG_Tellerrad_V1p", (Share("robot_1", None),),
        parameters=MachineOperationParameters("dmg", cycle_time=600.0).to_dict(),
    )

    assert parameters_of(Operation.from_dict(through_json(op.to_dict()))) == MachineOperationParameters("dmg", 600.0)


def test_machine_operation_without_a_cycle_time_leaves_it_to_the_skill_default():
    params = MachineOperationParameters("dmg")

    assert params.cycle_time is None
    assert MachineOperationParameters.from_dict(through_json(params.to_dict())) == params


@pytest.mark.parametrize("process", [Process.LOADING, Process.UNLOADING])
def test_loading_and_unloading_name_pallet_station_and_coupling_place(process):
    op = Operation(
        "load_1", process, None, (Share("robot_1", "pizzagreifer"),),
        parameters=PalletTransferParameters("pallet_1", "dmg", "table").to_dict(),
    )

    assert parameters_of(Operation.from_dict(through_json(op.to_dict()))) == PalletTransferParameters(
        "pallet_1", "dmg", "table"
    )


def test_reconfiguring_names_the_tooling_and_where_it_goes_from_and_to():
    params = ReconfiguringParameters("jaws_small", Location.at("clamping_desk", "jaw_rack"),
                                     Location.in_tool("pizzagreifer"))
    op = Operation("reconf_1", Process.RECONFIGURING, None, (Share("robot_1", "pizzagreifer"),),
                   parameters=params.to_dict())

    assert parameters_of(Operation.from_dict(through_json(op.to_dict()))) == params


def test_other_processes_have_no_parameter_type():
    op = Operation("grip_1", Process.GRIPPING, "lid", (Share("robot_1", "gripper"),))

    assert parameters_of(op) is None


def test_cell_state_says_where_each_pallet_and_tooling_is_and_round_trips():
    state = CellState(
        holds={"robot_2": "lid"},
        mounted_tools={"robot_1": "pizzagreifer"},
        locations={
            "pallet_1": Location.at("dmg", "table"),
            "jaws_small": Location.in_tool("pizzagreifer"),
            "press_tool": Location.at("press", "tool_seat"),
            "jaws_large": Location.held_by("robot_1"),
        },
        station_poses={"dmg": (0.1, 2.9, 0.0, 0.0, 0.0, 1.57)},
    )

    assert CellState.from_dict(through_json(state.to_dict())) == state


def test_the_snapshot_carries_the_new_cell_state_fields():
    snapshot = Cell_Assembly_Snapshot(
        "<assembly/>",
        CellState(locations={"pallet_1": Location.at("pallet_storage", "nsr_a")},
                  station_poses={"dmg": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}),
    )

    assert Cell_Assembly_Snapshot.from_dict(through_json(snapshot.to_dict())) == snapshot


def test_a_cell_state_without_pallets_or_tooling_keeps_its_dict():
    state = CellState({"robot_1": "lid"}, {"robot_1": "gripper"})

    assert state.to_dict() == {"holds": {"robot_1": "lid"}, "mounted_tools": {"robot_1": "gripper"}}
    assert CellState.from_dict({"holds": {}, "mounted_tools": {}}) == CellState()


def test_occupant_of_a_coupling_place():
    state = CellState(locations={"pallet_1": Location.at("dmg", "table")})

    assert state.occupant(Location.at("dmg", "table")) == "pallet_1"
    assert state.occupant(Location.at("pallet_storage", "table")) is None


def test_with_location_moves_an_item_and_leaves_the_rest():
    state = CellState({"robot_2": "lid"}, {"robot_1": "pizzagreifer"},
                      {"pallet_1": Location.at("pallet_storage", "nsr_a")})

    moved = state.with_location("pallet_1", Location.held_by("robot_1"))

    assert moved.locations == {"pallet_1": Location.held_by("robot_1")}
    assert (moved.holds, moved.mounted_tools) == (state.holds, state.mounted_tools)
    assert state.locations == {"pallet_1": Location.at("pallet_storage", "nsr_a")}


def test_a_measured_station_pose_overrides_the_nominal_one():
    dmg = Station("dmg", StationKind.MACHINE, (1.0, 2.0, 0.0, 0.0, 0.0, 0.0))
    measured = (1.01, 2.02, 0.0, 0.0, 0.0, 0.01)

    assert CellState().station_pose(dmg) == dmg.pose
    assert CellState(station_poses={"dmg": measured}).station_pose(dmg) == measured


def test_a_measured_station_pose_has_six_numbers():
    with pytest.raises(ValueError, match="six numbers"):
        CellState(station_poses={"dmg": (1.0, 2.0)})
