"""Supply slots (#151, AT 174): a slot's pose is relative to its supply station; planners read it in the cell frame."""

import math

import pytest
from diss_assembly_dataclasses.resource import Cell, Station, StationKind, SupplySlot

from mrtamp.supply import pose_in_cell_frame, slot_for_part, slots_in_cell_frame

ORIGIN = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_a_slot_pose_shifts_by_the_station_position():
    assert pose_in_cell_frame(
        (-1.2, -2.6, 0.8, 0.0, 0.0, 0.0), (0.4, 0.0, 0.0, 0.0, 0.0, 0.0)
    ) == pytest.approx((-0.8, -2.6, 0.8, 0.0, 0.0, 0.0))


def test_a_slot_pose_turns_with_the_station_rotation():
    # a station turned 90 degrees about z: the slot's x offset points along the cell's y axis
    station = (1.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2)
    slot = (0.5, 0.0, 0.1, 0.0, 0.0, math.pi / 4)
    assert pose_in_cell_frame(station, slot) == pytest.approx(
        (1.0, 0.5, 0.1, 0.0, 0.0, 3 * math.pi / 4), abs=1e-9
    )


def test_roll_pitch_yaw_are_fixed_axis_x_then_y_then_z():
    # pitch 90 degrees, then yaw 90 degrees, about the fixed axes: the station's x axis ends on the cell's -z
    station = (0.0, 0.0, 0.0, 0.0, math.pi / 2, math.pi / 2)
    x, y, z, *_ = pose_in_cell_frame(station, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert (x, y, z) == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)


def test_slots_in_cell_frame_are_the_slots_of_every_supply_station_with_cell_frame_poses():
    nest = SupplySlot(
        "nest_h5", (0.0, 0.2, 0.0, 0.0, 0.0, 0.0), screw_type="H5", robots=("robot_2",)
    )
    cell = Cell(
        robots=(),
        stations=(
            Station("dropoff", StationKind.DROPOFF, ORIGIN),
            Station(
                "supply",
                StationKind.SUPPLY,
                (-1.2, -2.6, 0.8, 0.0, 0.0, 0.0),
                slots=(nest,),
            ),
        ),
    )

    (resolved,) = slots_in_cell_frame(cell)

    assert resolved.name == "nest_h5" and resolved.robots == ("robot_2",)
    assert resolved.pose == pytest.approx((-1.2, -2.4, 0.8, 0.0, 0.0, 0.0))


def test_slot_for_part_finds_the_slot_that_serves_a_part():
    lid = SupplySlot("lid_slot", ORIGIN, part="lid")
    assert (
        slot_for_part((SupplySlot("nest", ORIGIN, screw_type="H5"), lid), "lid") == lid
    )
    assert slot_for_part((lid,), "gear") is None
