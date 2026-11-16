"""Product model edits with undo and redo (AT 240), on a copy of the tiny test product through the explorer's API.

Seam: the HTTP API and the session behind it. Renaming a part is the first edit; the tests of the mechanism
(the edit history, the stale sequence graph, the re-check of the available actions) go through it. Removing a
part cannot be reached by a rename, so that one case works on the session with a part-removing edit.
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from diss_assembly_dataclasses.assembly import Assembly
from fastapi.testclient import TestClient

from product_explorer.server import create_app
from product_explorer.session import ExplorerSession

TINY = Path(__file__).resolve().parents[1] / "test_products" / "tiny_test_product"
PARTS = ["01_base", "02_target", "03_cover", "04_screw"]


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


def _generate(client) -> dict:
    assert client.post("/api/generate").status_code == 202
    with client.websocket_connect("/ws/job") as ws:
        while True:
            job = ws.receive_json()
            if not job["running"]:
                break
    assert job["error"] is None, job["error"]
    return client.get("/api/state").json()


def _rename(client, part, name):
    return client.post("/api/rename", json={"part": part, "name": name})


def _structure(session) -> tuple:
    """Everything an undo must restore: part order, connection name pairs, setup."""
    return (
        list(session.assembly.parts),
        [(c.part_name1, c.part_name2) for c in session.assembly.connections],
        session.setup,
    )


def test_a_rename_changes_the_session_view_and_marks_edits_unsaved_and_the_graph_stale(
    client,
):
    view = _generate(client)
    assert not view["unsaved"] and not view["graph"]["stale"]
    view = _rename(client, "04_screw", "05_bolt").json()
    assert view["product"]["parts"] == ["01_base", "02_target", "03_cover", "05_bolt"]
    assert view["unsaved"] and view["graph"]["stale"]
    assert view["can_undo"] and not view["can_redo"]


def test_a_rename_follows_into_the_product_setup(client):
    view = _rename(client, "02_target", "02_rotor").json()
    assert view["setup"] == {"base_parts": ["01_base"], "target_parts": ["02_rotor"]}


def test_undo_restores_the_product_model_and_setup_exactly_and_redo_reapplies_the_edit(
    client, session
):
    before = _structure(session)
    _generate(client)
    after_edit = _rename(client, "01_base", "00_plate").json()
    assert after_edit["setup"]["base_parts"] == ["00_plate"]
    renamed = _structure(session)
    assert renamed != before

    undone = client.post("/api/undo").json()
    assert _structure(session) == before
    assert undone["graph"]["stale"] and not undone["unsaved"]
    assert undone["can_redo"] and not undone["can_undo"]

    redone = client.post("/api/redo").json()
    assert _structure(session) == renamed
    assert redone["graph"]["stale"] and redone["unsaved"]
    assert redone["can_undo"] and not redone["can_redo"]


def test_a_new_edit_after_an_undo_clears_the_redo_history(client):
    _rename(client, "04_screw", "05_bolt")
    client.post("/api/undo")
    assert client.get("/api/state").json()["can_redo"]
    _rename(client, "03_cover", "03_lid")
    view = client.get("/api/state").json()
    assert not view["can_redo"]
    assert client.post("/api/redo").status_code == 409


def test_undo_and_redo_without_history_are_refused(client):
    assert client.post("/api/undo").status_code == 409
    assert client.post("/api/redo").status_code == 409


@pytest.mark.parametrize(
    "part, name, fragment",
    [
        ("04_screw", "03_cover", "already named"),
        ("no_such_part", "x", "Unknown part"),
        ("04_screw", "  ", "needs a name"),
    ],
)
def test_a_rename_that_does_not_fit_is_a_user_error(client, part, name, fragment):
    response = _rename(client, part, name)
    assert response.status_code == 400 and fragment in response.json()["detail"]
    assert not client.get("/api/state").json()["can_undo"]


def test_opening_a_product_clears_the_undo_and_redo_history(client, folder):
    _rename(client, "04_screw", "05_bolt")
    client.post("/api/undo")
    view = client.post("/api/load", json={"folder": str(folder)}).json()
    assert not view["can_undo"] and not view["can_redo"] and not view["unsaved"]


def test_the_available_actions_are_rechecked_at_once_without_extracting_a_graph(
    client, monkeypatch
):
    from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent

    def refuse(*_args, **_kwargs):
        raise AssertionError("graph extraction ran")

    monkeypatch.setattr(DisassemblyGraphAgent, "build_graph", refuse)
    view = _rename(client, "04_screw", "05_bolt").json()
    available = {a["target_part"] for a in view["recheck"]["available"]}
    assert "05_bolt" in available and "04_screw" not in available


def test_a_part_whose_removal_is_rejected_shows_its_rejection_reason_on_the_edited_model(
    client,
):
    _rename(client, "04_screw", "05_bolt")
    base = client.get("/api/rejection", params={"part": "01_base"}).json()
    assert not base["available"] and "base part" in base["text"]
    assert "05_bolt" in base["text"] and "04_screw" not in base["text"]
    assert client.get("/api/rejection", params={"part": "04_screw"}).status_code == 400


def test_rebuild_extracts_the_graph_of_the_edited_model_and_clears_the_stale_state(
    client,
):
    _generate(client)
    _rename(client, "04_screw", "05_bolt")
    view = _generate(client)
    assert not view["graph"]["stale"]
    targets = {e["target_part"] for e in view["graph"]["edges"].values()}
    assert "05_bolt" in targets and "04_screw" not in targets


def test_rebuild_keeps_the_current_product_state_by_part_names(client):
    view = _generate(client)
    first = next(
        e
        for e in view["next_steps"]
        if view["graph"]["edges"][e]["target_part"] == "02_target"
    )
    view = client.post("/api/step", json={"edge": first}).json()
    assert "02_target" not in view["parts_present"]
    view = _rename(client, "04_screw", "05_bolt").json()
    assert view["parts_present"] == ["01_base", "03_cover", "05_bolt"]
    assert "02_target" not in {a["target_part"] for a in view["recheck"]["available"]}
    view = _generate(client)
    assert view["parts_present"] == ["01_base", "03_cover", "05_bolt"]
    assert len(view["path"]) == 1


def test_stepping_and_going_back_wait_for_a_rebuild_but_restart_starts_from_the_full_product(
    client,
):
    view = _generate(client)
    edge = view["next_steps"][0]
    client.post("/api/step", json={"edge": edge})
    view = _rename(client, "03_cover", "03_lid").json()
    assert view["next_steps"] == []
    assert client.post("/api/back").status_code == 409
    view = client.post("/api/restart").json()
    assert view["parts_present"] == ["01_base", "02_target", "03_lid", "04_screw"]
    assert len(view["recheck"]["available"]) >= 3


def test_an_edit_that_removes_the_current_product_state_restarts_from_the_full_product(
    session, client
):
    view = _generate(client)
    edge = next(
        e
        for e in view["next_steps"]
        if view["graph"]["edges"][e]["target_part"] == "04_screw"
    )
    client.post("/api/step", json={"edge": edge})
    assert "04_screw" not in session.parts_present()

    def delete_screw(assembly, _setup):
        assembly.remove_part(assembly.parts["04_screw"])
        return {}

    session._apply_edit(delete_screw)
    assert session.parts_present() == ["01_base", "02_target", "03_cover"]
    assert session.view()["path"] == []
    assert session.undo() is None
    assert "04_screw" in session.part_names


def test_the_setup_change_is_part_of_the_history(client):
    _generate(client)
    view = client.post(
        "/api/setup", json={"base_parts": ["01_base"], "target_parts": ["03_cover"]}
    ).json()
    assert (
        view["graph"]["stale"] and not view["unsaved"]
    )  # the product model is untouched
    view = client.post("/api/undo").json()
    assert view["setup"]["target_parts"] == ["02_target"]


def test_jump_and_export_are_refused_while_the_graph_is_stale_or_edits_are_unsaved(
    client, tmp_path
):
    view = _generate(client)
    node = next(n for n in view["graph"]["nodes"] if n != view["graph"]["root"])
    client.post("/api/store")
    _rename(client, "04_screw", "05_bolt")
    out = {"out_dir": str(tmp_path / "out")}
    assert client.post("/api/jump", json={"node": node}).status_code == 409
    assert client.post("/api/export", json=out).status_code == 409
    view = _generate(client)  # rebuilt, but the edit is not saved
    node = next(n for n in view["graph"]["nodes"] if n != view["graph"]["root"])
    assert not view["graph"]["stale"] and view["unsaved"]
    response = client.post("/api/jump", json={"node": node})
    assert response.status_code == 409 and "unsaved" in response.json()["detail"]
    assert client.post("/api/export", json=out).status_code == 409
    view = client.post("/api/save").json()
    assert (
        not view["unsaved"] and not view["graph"]["stale"]
    )  # saving does not make the graph stale
    assert client.post("/api/jump", json={"node": node}).status_code == 200
    client.post("/api/store")
    assert client.post("/api/export", json=out).status_code == 200


def test_edits_are_refused_while_the_rebuild_runs(session, client):
    session.job.running = True
    for call in (
        lambda: _rename(client, "04_screw", "05_bolt"),
        lambda: client.post("/api/undo"),
        lambda: client.post("/api/redo"),
        lambda: client.post(
            "/api/setup", json={"base_parts": [], "target_parts": ["02_target"]}
        ),
    ):
        assert call().status_code == 409
    session.job.running = False


def test_a_renamed_product_saves_and_loads_again_with_its_contact_data(
    client, session, folder
):
    _rename(client, "04_screw", "05_bolt")
    assert client.post("/api/save").status_code == 200
    fresh = ExplorerSession()
    fresh.load(folder)
    assert fresh.part_names == ["01_base", "02_target", "03_cover", "05_bolt"]
    assert any(
        "05_bolt" in (c.part_name1, c.part_name2) for c in fresh.assembly.connections
    )
    assert fresh.assembly.parts.keys() == session.assembly.parts.keys()


def test_the_3d_meshes_follow_a_rename_and_an_undo(client):
    assert set(client.get("/api/meshes").json()["parts"]) == set(PARTS)
    _rename(client, "04_screw", "05_bolt")
    assert "05_bolt" in client.get("/api/meshes").json()["parts"]
    client.post("/api/undo")
    assert set(client.get("/api/meshes").json()["parts"]) == set(PARTS)
