"""Cell file choice of the Product explorer (AT 245), on a copy of the tiny test product through the explorer's API.

Seam: the HTTP API and the session behind it. The tiny product has one screw (head H5) and gripper-removable
parts, so the available actions of its start state show which tools a cell file brings.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from diss_assembly_dataclasses.assembly import Assembly
from fastapi.testclient import TestClient

from disassembly_pomdp.travel_time_table import (
    TravelTimeTable,
    load_travel_time_table,
    save_travel_time_table,
    table_path,
)
from product_explorer.server import create_app
from product_explorer.session import ExplorerError, ExplorerSession

TINY = Path(__file__).resolve().parents[1] / "test_products" / "tiny_test_product"
CASES = Path(__file__).resolve().parents[2] / "cases"


@pytest.fixture(autouse=True)
def no_contact_detection(monkeypatch):
    """The test product ships its contact data, so nothing here may detect contacts."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("contact detection ran")

    monkeypatch.setattr(Assembly, "find_contacts", refuse)
    monkeypatch.setattr(Assembly, "check_pair_for_contact_new", refuse)


@pytest.fixture
def folder(tmp_path) -> Path:
    copy = tmp_path / "tiny_test_product"
    shutil.copytree(TINY, copy)
    return copy


@pytest.fixture
def session(folder) -> ExplorerSession:
    session = ExplorerSession()
    session.load(folder)
    return session


@pytest.fixture
def client(session) -> TestClient:
    return TestClient(create_app(session))


def _cell_file(tmp_path, name, tools, times=None) -> Path:
    """A cell file in the existing format with the given tools: [(name, process)], and its travel-time table
    with `times` over the Transformer cell's. Each robot of the Transformer cell carries the tools whose
    process it can carry out (#230)."""
    data = json.loads((CASES / "transformer_cell.json").read_text(encoding="utf-8"))
    data["cell"]["tools"] = [{"name": n, "process": p} for n, p in tools]
    for robot in data["cell"]["robots"]:
        robot["tools"] = [n for n, p in tools if p in robot["capabilities"]]
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    table = load_travel_time_table(CASES / "transformer_cell.json")
    save_travel_time_table(TravelTimeTable({**table.times, **(times or {})}), path)
    return path


@pytest.fixture
def grippers_only(tmp_path) -> Path:
    return _cell_file(tmp_path, "grippers.json", [("G1", "gripping_and_removing")])


@pytest.fixture
def screwdrivers_only(tmp_path) -> Path:
    return _cell_file(tmp_path, "screwdrivers.json", [("H5", "screwing_in")])


def _choose(client, path):
    return client.post("/api/cell", json={"path": str(path)})


def _capabilities(view) -> set[str]:
    return {a["capability"] for a in view["recheck"]["available"]}


def _generate(client) -> dict:
    assert client.post("/api/generate").status_code == 202
    with client.websocket_connect("/ws/job") as ws:
        while True:
            job = ws.receive_json()
            if not job["running"]:
                break
    assert job["error"] is None, job["error"]
    return client.get("/api/state").json()


def test_without_a_cell_file_the_default_cell_is_used_and_the_view_says_so(client):
    cell = client.get("/api/state").json()["cell"]
    assert cell["default"] and cell["name"] is None
    assert cell["screwdrivers"] == ["H5", "H6"] and cell["grippers"] == [
        "default_gripper"
    ]


def test_the_real_cell_file_makes_its_screwdriver_and_gripper_the_tools(client):
    view = _choose(client, CASES / "transformer_cell.json").json()
    assert view["cell"] == {
        "name": "transformer_cell.json",
        "path": str((CASES / "transformer_cell.json").resolve()),
        "default": False,
        "screwdrivers": ["H5"],
        "grippers": ["default_gripper"],
    }


def test_screwing_is_available_only_with_a_screwdriver_and_gripping_and_removing_only_with_a_gripper(
    client, grippers_only, screwdrivers_only
):
    both = _choose(client, CASES / "transformer_cell.json").json()
    assert _capabilities(both) == {
        "ScrewingCapability",
        "GrippingAndRemovingCapability",
    }

    view = _choose(client, grippers_only).json()
    assert _capabilities(view) == {"GrippingAndRemovingCapability"}

    view = _choose(client, screwdrivers_only).json()
    assert _capabilities(view) == {"ScrewingCapability"}


def test_choosing_a_cell_file_rechecks_at_once_marks_the_graph_stale_and_is_one_undoable_edit(
    client, grippers_only
):
    view = _generate(client)
    assert not view["graph"]["stale"] and not view["can_undo"]
    default_capabilities = _capabilities(_choose(client, grippers_only).json()) | {
        "ScrewingCapability"
    }

    view = client.get("/api/state").json()
    assert view["graph"]["stale"] and view["can_undo"] and not view["unsaved"]
    assert "ScrewingCapability" not in _capabilities(view)

    undone = client.post("/api/undo").json()
    assert undone["cell"]["default"] and not undone["can_undo"]
    assert _capabilities(undone) == default_capabilities

    redone = client.post("/api/redo").json()
    assert redone["cell"]["name"] == "grippers.json" and redone["can_undo"]
    assert "ScrewingCapability" not in _capabilities(redone)


def test_a_graph_built_with_one_cell_file_is_stale_for_another_and_current_after_a_rebuild(
    client, grippers_only, screwdrivers_only
):
    _choose(client, grippers_only)
    built = _generate(client)["graph"]
    assert not built["stale"] and built["cell_fingerprint"]

    after = _choose(client, screwdrivers_only).json()["graph"]
    assert after["stale"] and after["cell_fingerprint"] == built["cell_fingerprint"]

    rebuilt = _generate(client)["graph"]
    assert not rebuilt["stale"]
    assert rebuilt["cell_fingerprint"] != built["cell_fingerprint"]


def test_a_graph_built_with_the_default_cell_is_stale_for_a_cell_file(
    client, grippers_only
):
    assert _generate(client)["graph"]["cell_fingerprint"] is None
    assert _choose(client, grippers_only).json()["graph"]["stale"]


def test_a_changed_cell_file_chosen_again_is_a_change_and_the_same_file_is_not(
    client, grippers_only
):
    _choose(client, grippers_only)
    _generate(client)
    again = _choose(client, grippers_only).json()
    assert not again["graph"]["stale"]  # same file, same content: no edit

    table = table_path(grippers_only)
    data = json.loads(table.read_text(encoding="utf-8"))
    data["times"]["inspection_time"] = 99.0
    table.write_text(json.dumps(data), encoding="utf-8")
    assert _choose(client, grippers_only).json()["graph"][
        "stale"
    ]  # a changed travel-time table changes the cell


def test_the_travel_time_table_of_the_cell_file_replaces_the_default_settings(
    session, tmp_path
):
    path = _cell_file(
        tmp_path, "slow.json", [("H5", "screwing_in")], {"milling_penalty": 7.0}
    )
    session.set_cell(path)
    assert session.cell_choice.cell.settings.milling_penalty == 7.0
    assert session.cell_choice.cell.settings.feed_rate_milling == 1000.0


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "{}",
        '{"cell": {"robots": [], "tools": [{"name": "X", "process": "no_such"}], "stations": []}}',
        '{"cell": {"robots": [], "tools": [{"name": "Z", "process": "inspecting"}], "stations": []}}',
        '{"cell": {"robots": [], "tools": [{"name": "H5", "process": "unscrewing"}], "stations": [], "timing": {}}}',
    ],
    ids=[
        "not_json",
        "no_cell",
        "unknown_process",
        "no_tool_for_any_action",
        "times_in_the_cell",
    ],
)
def test_a_malformed_or_unusable_cell_file_is_a_user_error_and_the_cell_stays(
    client, tmp_path, grippers_only, content
):
    _choose(client, grippers_only)
    bad = tmp_path / "bad.json"
    bad.write_text(content, encoding="utf-8")
    save_travel_time_table(TravelTimeTable(), bad)

    response = _choose(client, bad)

    assert response.status_code == 400
    view = client.get("/api/state").json()
    assert view["cell"]["name"] == "grippers.json"
    assert _capabilities(view) == {"GrippingAndRemovingCapability"}
    client.post("/api/undo")  # one edit only: the failed choice left no history
    assert client.get("/api/state").json()["cell"]["default"]


def test_opening_a_product_returns_to_the_default_cell(client, folder, grippers_only):
    _choose(client, grippers_only)
    view = client.post("/api/load", json={"folder": str(folder)}).json()
    assert view["cell"]["default"] and not view["can_undo"]


def test_a_missing_cell_file_is_a_user_error(client, tmp_path):
    assert _choose(client, tmp_path / "nowhere.json").status_code == 400
    assert client.get("/api/state").json()["cell"]["default"]


def test_choosing_a_cell_file_needs_a_product(tmp_path, grippers_only):
    client = TestClient(create_app(ExplorerSession()))
    assert _choose(client, grippers_only).status_code == 409


def _export(session, out):
    session.generate().join()
    assert session.job.error is None
    session.store()
    session.export(out)
    return json.loads((out / "sequence_graph.json").read_text(encoding="utf-8"))[
        "provenance"
    ]


def test_the_export_records_the_chosen_cell_file_relative_to_the_export_folder(
    session, folder, tmp_path, grippers_only
):
    session.set_cell(grippers_only)

    provenance = _export(session, tmp_path / "out")

    assert provenance["cell_file"] == "../grippers.json"
    assert provenance["product_folder"] == "../tiny_test_product"
    assert provenance["fingerprints"]["cell_file"] == session.cell_choice.fingerprint
    assert str(tmp_path) not in json.dumps(provenance) and str(tmp_path).replace(
        "\\", "/"
    ) not in json.dumps(provenance)


def test_the_export_without_a_cell_file_records_none(session, tmp_path):
    provenance = _export(session, tmp_path / "out")

    assert (
        provenance["cell_file"] is None
        and provenance["fingerprints"]["cell_file"] is None
    )


def test_an_export_into_the_3d_files_repository_is_refused(session, tmp_path):
    session.generate().join()
    session.store()
    repo = tmp_path / "crc_1574_3D_Files"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    with pytest.raises(ExplorerError, match="3D files repository"):
        session.export(repo / "exports" / "tiny")
    assert not (repo / "exports").exists()
