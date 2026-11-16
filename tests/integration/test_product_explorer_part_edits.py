"""Part edits of the product explorer (AT 241), on a copy of the tiny test product through the explorer's API.

Seam: the HTTP API and the session behind it. The edits reuse the edit history of AT 240; these tests check each
edit, that it is one undoable step, and that it survives save and reload.
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


def _structure(session) -> tuple:
    """Everything an undo must restore: parts with material and density, connections with their contact data."""
    return (
        [(p.name, p.material, p.density) for p in session.assembly.parts.values()],
        [
            (
                type(c).__name__,
                c.part_name1,
                c.part_name2,
                len(getattr(c, "points", ())),
            )
            for c in session.assembly.connections
        ],
        session.setup,
    )


def _rules(tmp_path, text) -> str:
    path = tmp_path / "rules.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _rename_by_rules(client, tmp_path, text):
    return client.post(
        "/api/rename-by-rules", json={"rules_file": _rules(tmp_path, text)}
    )


def _material(client, part, material, density):
    return client.post(
        "/api/part/properties",
        json={"part": part, "material": material, "density": density},
    )


def _delete(client, part):
    return client.post("/api/part/delete", json={"part": part})


def _merge(client, parts, name):
    return client.post("/api/merge", json={"parts": parts, "name": name})


def _reloaded(folder) -> ExplorerSession:
    fresh = ExplorerSession()
    fresh.load(folder)
    return fresh


# ------------------------------------------------------------ rename by a rules file
def test_a_rules_file_renames_all_matching_parts_in_one_edit_through_connections_and_setup(
    client, session, tmp_path
):
    view = _rename_by_rules(
        client, tmp_path, "01_base 00_plate\n\n02_target 02_rotor\nno_such_part x\n"
    ).json()
    assert view["product"]["parts"] == [
        "00_plate",
        "02_rotor",
        "03_cover",
        "04_screw",
    ]
    assert view["setup"] == {"base_parts": ["00_plate"], "target_parts": ["02_rotor"]}
    names = {
        n for c in session.assembly.connections for n in (c.part_name1, c.part_name2)
    }
    assert names <= set(view["product"]["parts"]) and "00_plate" in names
    assert view["unsaved"]
    assert len(session._undo) == 1


def test_a_rules_file_can_swap_two_names(client, tmp_path):
    view = _rename_by_rules(
        client, tmp_path, "03_cover 04_screw\n04_screw 03_cover\n"
    ).json()
    assert view["product"]["parts"] == ["01_base", "02_target", "03_cover", "04_screw"]
    assert client.get("/api/meshes").status_code == 200


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("01_base\n", "expected 'old new'"),
        ("no_such_part x\n", "No rule"),
        ("01_base 03_cover\n", "several parts the name 03_cover"),
        ("01_base x\n02_target x\n", "several parts the name x"),
    ],
)
def test_a_rules_file_that_does_not_fit_is_a_user_error(
    client, tmp_path, text, fragment
):
    response = _rename_by_rules(client, tmp_path, text)
    assert response.status_code == 400 and fragment in response.json()["detail"]
    assert not client.get("/api/state").json()["can_undo"]


def test_a_missing_rules_file_is_a_user_error(client, tmp_path):
    response = client.post(
        "/api/rename-by-rules", json={"rules_file": str(tmp_path / "nope.txt")}
    )
    assert response.status_code == 400


# ------------------------------------------------------------ material and density
def test_material_and_density_are_saved_with_the_product_model_and_read_back_after_a_reload(
    client, folder
):
    view = _material(client, "03_cover", "Aluminium", 2700.0).json()
    assert view["product"]["properties"]["03_cover"] == {
        "material": "Aluminium",
        "density": 2700.0,
    }
    client.post("/api/save")
    fresh = _reloaded(folder)
    part = fresh.assembly.parts["03_cover"]
    assert (part.material, part.density) == ("Aluminium", 2700.0)
    assert fresh.assembly.parts["01_base"].density == 8000.0


def test_the_weight_of_a_part_follows_its_density(session, client):
    part = session.assembly.parts["03_cover"]
    part.mesh  # loads the mesh and the weight
    before = part.weight
    _material(client, "03_cover", "Aluminium", 4000.0)
    assert session.assembly.parts["03_cover"].weight == pytest.approx(before / 2)


@pytest.mark.parametrize(
    "material, density, fragment",
    [("", 1.0, "material"), ("Steel", 0, "density"), ("Steel", -3, "density")],
)
def test_a_material_or_density_that_does_not_fit_is_a_user_error(
    client, material, density, fragment
):
    response = _material(client, "03_cover", material, density)
    assert response.status_code == 400 and fragment in response.json()["detail"]
    assert _material(client, "no_such_part", "Steel", 1.0).status_code == 400
    assert not client.get("/api/state").json()["can_undo"]


# ------------------------------------------------------------ delete
def test_deleting_a_part_removes_it_with_its_connections_and_from_the_setup(
    client, session
):
    view = _delete(client, "02_target").json()
    assert view["product"]["parts"] == ["01_base", "03_cover", "04_screw"]
    assert view["setup"] == {"base_parts": ["01_base"], "target_parts": []}
    assert not any(
        "02_target" in (c.part_name1, c.part_name2)
        for c in session.assembly.connections
    )
    assert view["unsaved"] and len(session._undo) == 1


def test_the_3d_meshes_follow_a_delete_and_its_undo(client):
    client.get("/api/meshes")
    _delete(client, "04_screw")
    assert "04_screw" not in client.get("/api/meshes").json()["parts"]
    client.post("/api/undo")
    assert "04_screw" in client.get("/api/meshes").json()["parts"]


def test_unknown_parts_and_the_last_part_cannot_be_deleted(client):
    assert _delete(client, "no_such_part").status_code == 400
    for part in ("01_base", "02_target", "03_cover"):
        assert _delete(client, part).status_code == 200
    assert _delete(client, "04_screw").status_code == 400


def test_a_deleted_part_stays_deleted_after_save_and_reload(client, folder):
    _delete(client, "04_screw")
    client.post("/api/save")
    fresh = _reloaded(folder)
    assert fresh.part_names == ["01_base", "02_target", "03_cover"]
    assert all(
        "04_screw" not in (c.part_name1, c.part_name2)
        for c in fresh.assembly.connections
    )


# ------------------------------------------------------------ merge
def test_merging_parts_writes_the_mesh_inside_the_product_folder_and_replaces_the_members_after_reload(
    client, session, folder
):
    view = _merge(client, ["03_cover", "04_screw"], "05_lid").json()
    assert view["product"]["parts"] == ["01_base", "02_target", "05_lid"]
    mesh = folder / "tiny_test_product" / "05_lid.stl"
    assert mesh.is_file()
    assert client.post("/api/save").status_code == 200
    fresh = _reloaded(folder)
    assert fresh.part_names == ["01_base", "02_target", "05_lid"]
    assert Path(fresh.assembly.parts["05_lid"].mesh_path).resolve() == mesh.resolve()
    assert fresh.assembly.parts["05_lid"].mesh is not None
    assert all(
        {"03_cover", "04_screw"}.isdisjoint((c.part_name1, c.part_name2))
        for c in fresh.assembly.connections
    )


def test_merging_joins_the_contact_data_of_one_neighbour_into_one_connection(
    client, session, folder
):
    def points(a, b):
        return sum(
            len(c.points)
            for c in session.assembly.connections
            if {c.part_name1, c.part_name2} == {a, b} and hasattr(c, "points")
        )

    expected = points("01_base", "03_cover") + points("01_base", "04_screw")
    _merge(client, ["03_cover", "04_screw"], "05_lid")
    geometric = [
        c
        for c in session.assembly.connections
        if {c.part_name1, c.part_name2} == {"01_base", "05_lid"}
        and hasattr(c, "points")
    ]
    assert len(geometric) == 1 and len(geometric[0].points) == expected > 0
    client.post("/api/save")
    saved = [
        c
        for c in _reloaded(folder).assembly.connections
        if {c.part_name1, c.part_name2} == {"01_base", "05_lid"}
        and hasattr(c, "points")
    ]
    assert len(saved) == 1 and len(saved[0].points) == expected


def test_the_members_of_a_merge_are_replaced_in_the_setup_and_merging_base_with_target_is_refused(
    client,
):
    view = _merge(client, ["01_base", "03_cover"], "05_frame").json()
    assert view["setup"] == {"base_parts": ["05_frame"], "target_parts": ["02_target"]}
    client.post("/api/undo")
    response = _merge(client, ["01_base", "02_target"], "05_x")
    assert (
        response.status_code == 400
        and "base and of the target" in response.json()["detail"]
    )


@pytest.mark.parametrize(
    "parts, name, fragment",
    [
        (["03_cover"], "05_lid", "at least two"),
        (["03_cover", "03_cover"], "05_lid", "at least two"),
        ([], "05_lid", "at least two"),
        (["03_cover", "no_such_part"], "05_lid", "Unknown parts"),
        (["03_cover", "04_screw"], "01_base", "already named"),
        (["03_cover", "04_screw"], "  ", "file name"),
        (["03_cover", "04_screw"], "a/b", "file name"),
    ],
)
def test_merging_fewer_than_two_parts_or_with_a_bad_name_is_a_user_error(
    client, folder, parts, name, fragment
):
    response = _merge(client, parts, name)
    assert response.status_code == 400 and fragment in response.json()["detail"]
    assert not client.get("/api/state").json()["can_undo"]
    assert not (folder / "tiny_test_product" / "05_lid.stl").exists()


def test_a_merge_never_overwrites_a_mesh_file_of_the_product_folder(client, folder):
    mesh = folder / "tiny_test_product" / "07_old.stl"
    mesh.write_bytes(b"keep me")
    response = _merge(client, ["03_cover", "04_screw"], "07_old")
    assert (
        response.status_code == 400 and "pick another name" in response.json()["detail"]
    )
    assert mesh.read_bytes() == b"keep me"


def test_exploring_works_on_a_merged_product_after_a_rebuild(client):
    _merge(client, ["03_cover", "04_screw"], "05_lid")
    assert client.post("/api/generate").status_code == 202
    with client.websocket_connect("/ws/job") as ws:
        while True:
            job = ws.receive_json()
            if not job["running"]:
                break
    assert job["error"] is None, job["error"]
    view = client.get("/api/state").json()
    assert not view["graph"]["stale"]
    targets = {e["target_part"] for e in view["graph"]["edges"].values()}
    assert "05_lid" in targets and "03_cover" not in targets


# ------------------------------------------------------------ one undo, one redo
EDITS = {
    "rules": lambda client, tmp_path: _rename_by_rules(
        client, tmp_path, "01_base 00_plate\n04_screw 05_bolt\n"
    ),
    "material": lambda client, _tmp: _material(client, "03_cover", "Aluminium", 2700.0),
    "delete": lambda client, _tmp: _delete(client, "03_cover"),
    "merge": lambda client, _tmp: _merge(client, ["03_cover", "04_screw"], "05_lid"),
}


@pytest.mark.parametrize("edit", EDITS)
def test_each_edit_is_reverted_by_one_undo_and_reapplied_by_one_redo(
    client, session, tmp_path, edit
):
    before = _structure(session)
    assert EDITS[edit](client, tmp_path).status_code == 200
    after = _structure(session)
    assert after != before

    undone = client.post("/api/undo").json()
    assert _structure(session) == before
    assert not undone["can_undo"] and undone["can_redo"] and not undone["unsaved"]

    redone = client.post("/api/redo").json()
    assert _structure(session) == after
    assert redone["can_undo"] and not redone["can_redo"] and redone["unsaved"]


@pytest.mark.parametrize("edit", EDITS)
def test_each_edit_marks_the_graph_stale_and_is_refused_while_the_rebuild_runs(
    client, session, tmp_path, edit
):
    client.post("/api/generate")
    with client.websocket_connect("/ws/job") as ws:
        while ws.receive_json()["running"]:
            pass
    assert EDITS[edit](client, tmp_path).json()["graph"]["stale"]
    client.post("/api/undo")
    session.job.running = True
    assert EDITS[edit](client, tmp_path).status_code == 409
    session.job.running = False
