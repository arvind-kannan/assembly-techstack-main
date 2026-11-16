"""Mesh update of the product explorer (AT 244): the open product takes the meshes of a new CAD export folder in
one undoable edit, as a background job, and answers with a report. Seam: the explorer's HTTP API and the session
behind it.

The per-pair contact detection is replaced by a recorder that hands out the contacts the tiny test product ships,
so the tests see which pairs the update detected again (and can wait on a gate to hold the job).
"""

import hashlib
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

pytest.importorskip("fastapi")
from diss_assembly_dataclasses.assembly import Assembly
from fastapi.testclient import TestClient

from product_explorer.product_folder import load_product_folder
from product_explorer.server import create_app
from product_explorer.session import ExplorerSession

TINY = Path(__file__).resolve().parents[1] / "test_products" / "tiny_test_product"


class Detection:
    """Stands in for `Assembly.check_pair_for_contact_new`: records the pair, adds the tiny product's contacts."""

    def __init__(self, monkeypatch):
        _xml, tiny, _base, _target = load_product_folder(TINY)
        self.contacts = tiny.connections
        self.pairs: list[frozenset] = []
        self.gate = threading.Event()
        self.gate.set()

        def check(assembly, pair, contact_tollerance, dot_product_threshold):
            self.gate.wait(timeout=30)
            names = frozenset(p.name for p in pair)
            self.pairs.append(names)
            assembly.connections += [
                c for c in self.contacts if {c.part_name1, c.part_name2} == names
            ]

        monkeypatch.setattr(Assembly, "check_pair_for_contact_new", check)


@pytest.fixture
def detection(monkeypatch) -> Detection:
    return Detection(monkeypatch)


@pytest.fixture
def folder(tmp_path) -> Path:
    copy = tmp_path / "tiny_test_product"
    shutil.copytree(TINY, copy)
    return copy


@pytest.fixture
def session(folder, detection) -> ExplorerSession:
    session = ExplorerSession()
    session.load(folder)
    return session


@pytest.fixture
def client(session) -> TestClient:
    return TestClient(create_app(session))


def _shift(source: Path, target: Path, by: float) -> None:
    mesh = o3d.io.read_triangle_mesh(str(source))
    mesh.translate((by, 0.0, 0.0))
    mesh.compute_triangle_normals()
    o3d.io.write_triangle_mesh(str(target), mesh)


@pytest.fixture
def cad(tmp_path) -> Path:
    """A new CAD export of the tiny product: 01_base moved, 02_target gone, 05_extra new, the rest as it was."""
    new = tmp_path / "cad"
    shutil.copytree(TINY / "tiny_test_product", new)
    _shift(new / "01_base.stl", new / "01_base.stl", 0.5)
    (new / "02_target.stl").unlink()
    _shift(new / "04_screw.stl", new / "05_extra.stl", 0.3)
    return new


def _check(client, folder: Path, **options):
    return client.post("/api/update/check", json={"folder": str(folder), **options})


def _update(client, folder: Path, add=(), **options):
    return client.post(
        "/api/update", json={"folder": str(folder), "add": list(add), **options}
    )


def _wait(client, timeout: float = 30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = client.get("/api/state").json()
        if not view["job"]["running"]:
            return view
        time.sleep(0.02)
    raise AssertionError("the job did not finish")


def _updated(client, folder: Path, add=()) -> dict:
    assert _update(client, folder, add).status_code == 202
    view = _wait(client)
    assert view["job"]["error"] is None
    return view


def _vertices(session, part: str) -> np.ndarray:
    return np.asarray(session.assembly.parts[part].mesh.vertices).copy()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _structure(session) -> tuple:
    return (
        [
            (p.name, p.mesh_path, p.material, p.density)
            for p in session.assembly.parts.values()
        ],
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


def test_the_check_lists_changed_unchanged_gone_and_new_files_and_changes_nothing(
    client, session, cad
):
    before = _structure(session)
    plan = _check(client, cad).json()
    assert plan["changed"] == ["01_base"]
    assert plan["unchanged"] == ["03_cover", "04_screw"]
    assert plan["removed"] == ["02_target"]
    assert plan["merged"] == []
    assert plan["new_files"] == ["05_extra.stl"]
    assert _structure(session) == before
    assert not client.get("/api/state").json()["can_undo"]


def test_a_changed_part_gets_the_new_mesh_and_detects_its_contacts_again(
    client, session, cad, detection
):
    old = _vertices(session, "01_base")
    view = _updated(client, cad)
    new = _vertices(session, "01_base")
    assert np.allclose(new[:, 0], old[:, 0] + 0.5)
    assert frozenset({"01_base", "03_cover"}) in detection.pairs
    assert (
        frozenset({"03_cover", "04_screw"}) not in detection.pairs
    )  # nothing about it changed
    assert view["unsaved"]
    geometric = {
        frozenset({c.part_name1, c.part_name2})
        for c in session.assembly.connections
        if c.connection_type == "geometric"
    }
    assert frozenset({"01_base", "03_cover"}) in geometric


def test_renames_properties_and_hand_made_connections_of_a_changed_part_are_kept(
    client, session, cad
):
    client.post("/api/rename", json={"part": "01_base", "name": "base"})
    client.post(
        "/api/part/properties",
        json={"part": "base", "material": "Brass", "density": 8500.0},
    )
    client.post(
        "/api/connection", json={"kind": "fixed", "part1": "base", "part2": "03_cover"}
    )
    _updated(client, cad)
    part = session.assembly.parts["base"]
    assert (part.material, part.density) == ("Brass", 8500.0)
    assert any(
        c.connection_type == "fixed"
        and {c.part_name1, c.part_name2} == {"base", "03_cover"}
        for c in session.assembly.connections
    )
    assert session.setup["base_parts"] == ["base"]
    assert "base" in session.assembly.parts and "01_base" not in session.assembly.parts


def test_a_part_whose_mesh_file_is_gone_is_removed_with_its_connections_and_setup_entry(
    client, session, cad
):
    view = _updated(client, cad)
    assert "02_target" not in session.assembly.parts
    assert not any(
        "02_target" in (c.part_name1, c.part_name2)
        for c in session.assembly.connections
    )
    assert view["setup"]["target_parts"] == []
    assert view["update_report"]["removed"] == ["02_target"]


def test_new_mesh_files_join_only_when_checked_and_detect_contacts_with_their_neighbours(
    client, session, cad, detection
):
    view = _updated(client, cad)
    assert "05_extra" not in session.assembly.parts
    assert view["update_report"]["not_added"] == ["05_extra.stl"]
    client.post("/api/undo")
    detection.pairs.clear()
    view = _updated(client, cad, add=["05_extra.stl"])
    assert "05_extra" in session.assembly.parts
    assert view["update_report"]["added"] == ["05_extra"]
    assert view["update_report"]["not_added"] == []
    assert any("05_extra" in pair for pair in detection.pairs)


def test_the_add_list_must_name_new_files_that_do_not_clash_with_a_part_name(
    client, session, cad, tmp_path
):
    assert _update(client, cad, add=["nothing.stl"]).status_code == 400
    assert _update(client, cad, add=["01_base.stl"]).status_code == 400
    client.post("/api/rename", json={"part": "03_cover", "name": "05_extra"})
    assert _check(client, cad).json()["clashing"] == ["05_extra.stl"]
    assert _update(client, cad, add=["05_extra.stl"]).status_code == 400


def test_the_report_is_in_the_session_view_and_the_status_log(client, cad):
    view = _updated(client, cad, add=["05_extra.stl"])
    report = view["update_report"]
    assert report["changed"] == ["01_base"]
    assert report["removed"] == ["02_target"]
    assert report["added"] == ["05_extra"]
    assert report["unchanged"] == 2
    assert report["merged"] == []
    assert any("01_base" in line and "02_target" in line for line in view["status_log"])


def test_one_undo_reverts_the_whole_update_and_one_redo_reapplies_it(
    client, session, cad
):
    before = _structure(session)
    _updated(client, cad, add=["05_extra.stl"])
    after = _structure(session)
    assert after != before
    assert client.post("/api/undo").status_code == 200
    assert _structure(session) == before
    assert not client.get("/api/state").json()["can_undo"]
    assert client.post("/api/redo").status_code == 200
    assert _structure(session) == after


def test_undo_restores_the_old_mesh_even_after_the_product_model_is_copied(
    client, session, cad
):
    old = _vertices(session, "01_base")
    old_file = Path(session.assembly.parts["01_base"].mesh_path)
    old_digest = _digest(old_file)
    _updated(client, cad)
    client.post("/api/undo")
    # an edit copies the product model, which reloads every mesh from its path
    client.post(
        "/api/part/properties",
        json={"part": "01_base", "material": "X", "density": 1.0},
    )
    assert np.allclose(_vertices(session, "01_base"), old)
    assert _digest(old_file) == old_digest


def test_the_cad_folder_is_never_written_and_new_meshes_go_into_the_product_folder(
    client, session, folder, cad
):
    digests = {p.name: _digest(p) for p in cad.iterdir()}
    _updated(client, cad, add=["05_extra.stl"])
    assert {p.name: _digest(p) for p in cad.iterdir()} == digests
    for name in ("01_base", "05_extra"):
        path = Path(session.assembly.parts[name].mesh_path).resolve()
        assert folder.resolve() in path.parents
        assert path.name == f"{name}.stl"


def test_a_second_update_matches_by_the_same_file_names(client, session, cad, tmp_path):
    _updated(client, cad)
    again = tmp_path / "cad2"
    shutil.copytree(cad, again)
    _shift(again / "03_cover.stl", again / "03_cover.stl", 0.25)
    plan = _check(client, again).json()
    assert plan["changed"] == ["03_cover"]
    assert plan["unchanged"] == ["01_base", "04_screw"]
    assert plan["removed"] == []


def test_save_and_reload_keep_the_updated_product_with_portable_paths(
    client, session, folder, cad
):
    _updated(client, cad, add=["05_extra.stl"])
    client.post("/api/save")
    xml = next(folder.glob("*.xml")).read_text(encoding="utf-8")
    assert str(folder.parent) not in xml and "\\" not in xml
    moved = folder.parent / "moved"
    shutil.copytree(folder, moved)
    other = ExplorerSession()
    other.load(moved)
    assert sorted(other.assembly.parts) == [
        "01_base",
        "03_cover",
        "04_screw",
        "05_extra",
    ]
    assert np.allclose(_vertices(other, "01_base"), _vertices(session, "01_base"))


def test_merged_subassemblies_are_skipped_and_listed_for_a_hand_merge(
    client, session, cad
):
    client.post("/api/merge", json={"parts": ["03_cover", "04_screw"], "name": "lid"})
    plan = _check(client, cad).json()
    assert plan["merged"] == ["lid"]
    assert plan["removed"] == ["02_target"]
    view = _updated(client, cad)
    assert "lid" in session.assembly.parts
    assert view["update_report"]["merged"] == ["lid"]
    assert any("lid" in line and "re-merge" in line for line in view["status_log"])


def test_undoing_a_merge_makes_its_members_plain_parts_again(client, session, cad):
    client.post("/api/merge", json={"parts": ["03_cover", "04_screw"], "name": "lid"})
    client.post("/api/undo")
    plan = _check(client, cad).json()
    assert plan["merged"] == []
    assert plan["unchanged"] == ["03_cover", "04_screw"]


def test_an_update_with_nothing_to_do_makes_no_edit(client, session, tmp_path):
    same = tmp_path / "same"
    shutil.copytree(TINY / "tiny_test_product", same)
    assert _update(client, same).status_code == 202
    view = _wait(client)
    assert not view["can_undo"] and not view["unsaved"]
    assert (
        view["update_report"]["changed"] == []
        and view["update_report"]["removed"] == []
    )


def test_while_the_update_runs_edits_are_refused_with_409(client, cad, detection):
    detection.gate.clear()
    assert _update(client, cad).status_code == 202
    job = client.get("/api/state").json()["job"]
    assert job["running"] and job["kind"] == "update"
    assert _update(client, cad).status_code == 409
    assert (
        client.post("/api/rename", json={"part": "03_cover", "name": "x"}).status_code
        == 409
    )
    assert client.post("/api/undo").status_code == 409
    assert client.post("/api/part/delete", json={"part": "03_cover"}).status_code == 409
    assert client.post("/api/load", json={"folder": str(TINY)}).status_code == 409
    detection.gate.set()
    view = _wait(client)
    assert view["job"]["error"] is None and view["update_report"]["changed"] == [
        "01_base"
    ]


def test_the_job_reports_the_pairs_it_checks(client, session, cad, detection):
    view = _updated(client, cad)
    assert view["job"]["kind"] == "update"
    assert (
        view["job"]["expanded"] == view["job"]["discovered"] == len(detection.pairs) > 0
    )


def test_a_product_that_was_never_saved_cannot_be_updated(
    cad, meshes_of_tiny, detection
):
    client = TestClient(create_app())
    assert (
        client.post("/api/import", json={"folder": str(meshes_of_tiny)}).status_code
        == 202
    )
    _wait(client)
    assert _update(client, cad).status_code == 409
    assert _check(client, cad).status_code == 409


@pytest.fixture
def meshes_of_tiny(tmp_path) -> Path:
    path = tmp_path / "import_folder"
    shutil.copytree(TINY / "tiny_test_product", path)
    return path


def test_bad_folders_and_a_product_that_is_not_open_are_refused(client, tmp_path):
    assert _check(client, tmp_path / "nowhere").status_code == 400
    assert _update(client, tmp_path / "nowhere").status_code == 400
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _check(client, empty).status_code == 400
    assert _check(TestClient(create_app()), empty).status_code == 409


def test_two_meshes_with_one_file_name_in_the_new_folder_are_refused(
    client, cad, tmp_path
):
    for sub in ("a", "b"):
        (cad / sub).mkdir()
        shutil.copyfile(cad / "03_cover.stl", cad / sub / "03_cover.stl")
    assert _check(client, cad, depth=1).status_code == 400
    assert _check(client, cad, depth=0).status_code == 200


def test_a_failed_detection_keeps_the_open_product_and_its_history(
    client, session, cad, monkeypatch
):
    before = _structure(session)

    def broken(*_args, **_kwargs):
        raise RuntimeError("detection broke")

    monkeypatch.setattr(Assembly, "check_pair_for_contact_new", broken)
    assert _update(client, cad).status_code == 202
    view = _wait(client)
    assert "detection broke" in view["job"]["error"]
    assert _structure(session) == before
    assert not view["can_undo"]


def test_the_mesh_version_changes_with_an_update_and_its_undo_and_redo(
    client, session, cad
):
    first = client.get("/api/state").json()["product"]["mesh_version"]
    updated = _updated(client, cad)["product"]["mesh_version"]
    assert updated != first
    assert client.post("/api/undo").json()["product"]["mesh_version"] == first
    assert client.post("/api/redo").json()["product"]["mesh_version"] == updated
    client.post("/api/undo")
    again = _updated(client, cad, add=["05_extra.stl"])["product"]["mesh_version"]
    assert again not in (first, updated)


def test_an_edit_that_keeps_the_meshes_keeps_the_mesh_version(client, cad):
    first = client.get("/api/state").json()["product"]["mesh_version"]
    view = client.post(
        "/api/part/properties",
        json={"part": "01_base", "material": "X", "density": 1.0},
    ).json()
    assert view["product"]["mesh_version"] == first


def test_a_gone_part_frees_its_name_for_a_mesh_file_with_another_ending(
    client, session, cad
):
    mesh = o3d.io.read_triangle_mesh(str(TINY / "tiny_test_product" / "02_target.stl"))
    o3d.io.write_triangle_mesh(str(cad / "02_target.obj"), mesh)
    plan = _check(client, cad).json()
    assert plan["removed"] == ["02_target"] and "02_target.obj" not in plan["clashing"]
    view = _updated(client, cad, add=["02_target.obj"])
    assert "02_target" in session.assembly.parts
    assert view["update_report"]["added"] == ["02_target"]


def test_a_nothing_to_change_update_shows_a_finished_update_job(client, tmp_path):
    same = tmp_path / "same"
    shutil.copytree(TINY / "tiny_test_product", same)
    assert _update(client, same).status_code == 202
    job = client.get("/api/state").json()["job"]
    assert job["kind"] == "update" and not job["running"]
