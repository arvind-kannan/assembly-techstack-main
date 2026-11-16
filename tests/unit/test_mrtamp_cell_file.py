import json
from dataclasses import replace

import pytest
from diss_assembly_dataclasses.cell_state import Location
from diss_assembly_dataclasses.resource import Station, StationKind, SupplySlot

from disassembly_pomdp.graph_export import fingerprint_of_cell
from disassembly_pomdp.travel_time_table import TravelTimeTable, table_path
from mrtamp.cell_file import CellFile, load_cell_file, save_cell_file, with_station
from tests.mrtamp_scenarios import transformer_cell, travel_times

SUPPLY = Station(
    "supply",
    StationKind.SUPPLY,
    (-1.2, -2.6, 0.8, 0.0, 0.0, 0.0),
    slots=(SupplySlot("lid_slot", (0.4, 0.0, 0.0, 0.0, 0.0, 0.0), part="lid"),),
)
DESCRIPTIONS = {"robot_1": "robots/nj290.urdf", "supply": "stations/supply.urdf"}


def a_cell_file(**changes) -> CellFile:
    return replace(
        CellFile(
            with_station(transformer_cell(), SUPPLY),
            {"robot_1": "H5", "robot_2": None},
            DESCRIPTIONS,
            travel_times=travel_times(),
        ),
        **changes,
    )


def test_cell_file_round_trips_the_cell_the_mounted_tools_the_descriptions_and_the_travel_times(
    tmp_path,
):
    original = a_cell_file()

    save_cell_file(original, tmp_path / "cell.json")

    assert load_cell_file(tmp_path / "cell.json") == original


def test_cell_file_round_trips_where_pallets_and_tooling_start_and_the_notes(tmp_path):
    original = a_cell_file(
        locations={"pallet": Location.at("pallet_storage", "nsr_a")},
        notes={"pallet_storage": "pose is a placeholder"},
    )

    save_cell_file(original, tmp_path / "cell.json")

    assert load_cell_file(tmp_path / "cell.json") == original


def test_a_cell_file_without_pallets_tooling_or_notes_keeps_its_json(tmp_path):
    save_cell_file(a_cell_file(), tmp_path / "cell.json")

    data = json.loads((tmp_path / "cell.json").read_text())
    assert "locations" not in data and "notes" not in data


def test_the_times_are_in_a_travel_time_table_file_next_to_the_cell_file(tmp_path):
    save_cell_file(a_cell_file(), tmp_path / "cell.json")

    assert "timing" not in json.loads((tmp_path / "cell.json").read_text())["cell"]
    assert table_path(tmp_path / "cell.json") == tmp_path / "cell.travel_times.json"
    assert json.loads(table_path(tmp_path / "cell.json").read_text()) == {
        "times": travel_times().times
    }


def test_a_cell_file_that_still_holds_the_times_is_refused(tmp_path):
    """A cell file from before AT 174: its times go into the travel-time table (ADR-0012)."""
    path = tmp_path / "cell.json"
    save_cell_file(a_cell_file(), path)
    data = json.loads(path.read_text())
    data["cell"]["timing"] = {"inspection_time": 10.0}
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="travel-time table"):
        load_cell_file(path)


def test_a_cell_file_without_its_travel_time_table_is_refused(tmp_path):
    path = tmp_path / "cell.json"
    save_cell_file(a_cell_file(), path)
    table_path(path).unlink()

    with pytest.raises(FileNotFoundError, match="cell.travel_times.json"):
        load_cell_file(path)


def test_simulation_only_flag_is_saved_and_loaded_and_defaults_to_false(tmp_path):
    save_cell_file(a_cell_file(simulation_only=True), tmp_path / "sim.json")
    save_cell_file(a_cell_file(), tmp_path / "real.json")

    assert load_cell_file(tmp_path / "sim.json").simulation_only is True
    assert load_cell_file(tmp_path / "real.json").simulation_only is False


def test_with_station_replaces_the_station_of_that_name_or_adds_it():
    cell = with_station(transformer_cell(), SUPPLY)
    moved = replace(SUPPLY, pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    assert cell.station("supply") == SUPPLY
    assert with_station(cell, moved).station("supply") == moved
    assert len(with_station(cell, moved).stations) == len(cell.stations)


def fingerprint_of(cell_file: CellFile, path) -> str:
    save_cell_file(cell_file, path)
    return fingerprint_of_cell(path)


def test_the_fingerprint_changes_with_a_slot_pose_a_station_pose_a_description_or_a_time(
    tmp_path,
):
    slot = replace(SUPPLY.slots[0], pose=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    changed = {
        "slot pose": a_cell_file(
            cell=with_station(transformer_cell(), replace(SUPPLY, slots=(slot,)))
        ),
        "station pose": a_cell_file(
            cell=with_station(
                transformer_cell(),
                replace(SUPPLY, pose=(-1.2, -2.6, 0.9, 0.0, 0.0, 0.0)),
            )
        ),
        "description": a_cell_file(
            descriptions={**DESCRIPTIONS, "supply": "stations/supply_v2.urdf"}
        ),
        "time": a_cell_file(
            travel_times=TravelTimeTable(
                {**travel_times().times, "inspection_time": 12.0}
            )
        ),
    }
    original = fingerprint_of(a_cell_file(), tmp_path / "original.json")

    for what, cell_file in changed.items():
        assert fingerprint_of(cell_file, tmp_path / f"{what}.json") != original, what


def test_a_cell_without_a_cell_file_has_no_fingerprint(tmp_path):
    assert fingerprint_of_cell(tmp_path / "missing.json") is None
