"""Mesh folder import of the Product explorer (#243): a mesh folder becomes a product in one step, contact
detection runs as a background job with progress and cancel, and the first save writes a product folder.
Seam: the explorer's HTTP API.

The fast tests replace the per-pair contact detection with a fake that hands out the contacts the tiny test
product ships (and can wait on a gate, so job states are deterministic). One `geom` test runs the real detection.
"""

import re
import shutil
import threading
import time
from pathlib import Path

import open3d as o3d
import pytest

pytest.importorskip("fastapi")
from diss_assembly_dataclasses.assembly import Assembly
from fastapi.testclient import TestClient

from product_explorer.product_folder import load_product_folder
from product_explorer.server import create_app

TINY = Path(__file__).resolve().parents[1] / "test_products" / "tiny_test_product"
PARTS = ["01_base", "02_target", "03_cover", "04_screw"]


@pytest.fixture
def meshes(tmp_path) -> Path:
    """The tiny test product's meshes as a plain mesh folder."""
    folder = tmp_path / "mesh_folder"
    shutil.copytree(TINY / "tiny_test_product", folder)
    return folder


class FakeDetection:
    """Stands in for `Assembly.check_pair_for_contact_new`: adds the tiny product's contacts of the pair."""

    def __init__(self, monkeypatch):
        _xml, tiny, _base, _target = load_product_folder(TINY)
        self.contacts = tiny.connections
        self.gate = threading.Event()
        self.gate.set()
        self.pairs = 0
        self.fail = False

        def check(assembly, pair, contact_tollerance, dot_product_threshold):
            self.gate.wait(timeout=30)
            if self.fail:
                raise RuntimeError("detection broke")
            self.pairs += 1
            names = {p.name for p in pair}
            assembly.connections += [
                c for c in self.contacts if {c.part_name1, c.part_name2} == names
            ]

        monkeypatch.setattr(Assembly, "check_pair_for_contact_new", check)


@pytest.fixture
def detection(monkeypatch) -> FakeDetection:
    return FakeDetection(monkeypatch)


def _import(client, folder: Path, **options):
    return client.post("/api/import", json={"folder": str(folder), **options})


def _wait(client, timeout: float = 30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = client.get("/api/state").json()
        if not view["job"]["running"]:
            return view
        time.sleep(0.02)
    raise AssertionError("the job did not finish")


def _imported(client, folder: Path, **options) -> dict:
    assert _import(client, folder, **options).status_code == 202
    return _wait(client)


def test_a_mesh_folder_becomes_an_open_product_with_connections_and_an_empty_setup(
    meshes, detection
):
    client = TestClient(create_app())
    view = _imported(client, meshes)
    assert view["job"]["error"] is None and not view["job"]["cancelled"]
    assert view["product"]["name"] == "mesh_folder"  # default: the folder name
    assert sorted(view["product"]["parts"]) == PARTS
    assert view["setup"] == {"base_parts": [], "target_parts": []}
    assert detection.pairs == 6  # every pair of 4 parts was checked
    assert view["job"]["expanded"] == view["job"]["discovered"] == 6
    assert view["unsaved"] is True and view["product"]["folder"] is None


def test_the_product_name_option_replaces_the_folder_name(meshes, detection):
    view = _imported(TestClient(create_app()), meshes, name="My product")
    assert view["product"]["name"] == "My product"


def test_an_empty_setup_shows_both_notices(meshes, detection):
    view = _imported(TestClient(create_app()), meshes)
    assert view["notices"] == [
        "no base parts: every part may be removed",
        "no target parts: no goal states",
    ]


def test_a_mesh_folder_with_no_meshes_is_a_user_error(tmp_path, detection):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("not a mesh")
    response = _import(TestClient(create_app()), empty)
    assert response.status_code == 400 and "no meshes" in response.json()["detail"]


def test_meshes_in_subfolders_need_the_subfolder_depth(tmp_path, meshes, detection):
    nested = tmp_path / "nested"
    (nested / "group").mkdir(parents=True)
    for stl in meshes.glob("*.stl"):
        shutil.copy(stl, nested / "group" / stl.name)
    client = TestClient(create_app())
    assert _import(client, nested).status_code == 400
    assert sorted(_imported(client, nested, depth=1)["product"]["parts"]) == PARTS


def test_two_meshes_with_one_name_are_refused(tmp_path, meshes, detection):
    clash = tmp_path / "clash"
    (clash / "a").mkdir(parents=True)
    (clash / "b").mkdir()
    shutil.copy(meshes / "01_base.stl", clash / "a" / "same.stl")
    shutil.copy(meshes / "02_target.stl", clash / "b" / "same.stl")
    response = _import(TestClient(create_app()), clash, depth=1)
    assert response.status_code == 400 and "same" in response.json()["detail"]


def test_a_missing_folder_and_bad_options_are_user_errors(tmp_path, meshes, detection):
    client = TestClient(create_app())
    assert _import(client, tmp_path / "nowhere").status_code == 400
    assert _import(client, meshes, tolerance=0).status_code == 400
    assert _import(client, meshes, angle=2).status_code == 400


def test_while_detection_runs_edits_and_other_imports_are_refused(meshes, detection):
    client = TestClient(create_app())
    detection.gate.clear()
    assert _import(client, meshes).status_code == 202
    assert client.get("/api/state").json()["job"]["running"]
    assert _import(client, meshes).status_code == 409
    assert (
        client.post(
            "/api/setup", json={"base_parts": [], "target_parts": []}
        ).status_code
        == 409
    )
    assert (
        client.post("/api/rename", json={"part": "a", "name": "b"}).status_code == 409
    )
    assert client.post("/api/undo").status_code == 409
    assert client.post("/api/load", json={"folder": str(TINY)}).status_code == 409
    detection.gate.set()
    assert _wait(client)["product"]["name"] == "mesh_folder"


def test_cancel_discards_the_import_and_keeps_the_open_product_with_its_history(
    meshes, detection, tmp_path
):
    tiny = tmp_path / "tiny"
    shutil.copytree(TINY, tiny)
    client = TestClient(create_app())
    client.post("/api/load", json={"folder": str(tiny)})
    client.post("/api/rename", json={"part": "04_screw", "name": "05_screw"})
    before = client.get("/api/state").json()
    detection.gate.clear()
    _import(client, meshes)
    assert client.post("/api/cancel").status_code == 200
    detection.gate.set()
    after = _wait(client)
    assert after["job"]["cancelled"] is True
    assert {k: after[k] for k in ("product", "setup", "can_undo", "unsaved")} == {
        k: before[k] for k in ("product", "setup", "can_undo", "unsaved")
    }
    assert after["can_undo"] is True


def test_cancel_without_a_running_import_is_a_conflict():
    assert TestClient(create_app()).post("/api/cancel").status_code == 409


def test_a_failed_detection_keeps_the_open_product(meshes, detection, tmp_path):
    tiny = tmp_path / "tiny"
    shutil.copytree(TINY, tiny)
    client = TestClient(create_app())
    client.post("/api/load", json={"folder": str(tiny)})
    detection.fail = True
    view = _imported(client, meshes)
    assert "detection broke" in view["job"]["error"]
    assert view["product"]["name"] == "tiny_test_product"


def test_import_clears_the_undo_and_redo_history(meshes, detection, tmp_path):
    tiny = tmp_path / "tiny"
    shutil.copytree(TINY, tiny)
    client = TestClient(create_app())
    client.post("/api/load", json={"folder": str(tiny)})
    client.post("/api/rename", json={"part": "04_screw", "name": "05_screw"})
    client.post("/api/undo")
    assert client.get("/api/state").json()["can_redo"] is True
    view = _imported(client, meshes)
    assert view["can_undo"] is False and view["can_redo"] is False


def _box_folder(tmp_path: Path, size: float, count: int = 2) -> Path:
    folder = tmp_path / f"boxes_{size}"
    folder.mkdir()
    for i in range(count):
        box = o3d.geometry.TriangleMesh.create_box(size, size, size)
        box.translate((i * size, 0, 0))
        box.compute_triangle_normals()
        o3d.io.write_triangle_mesh(str(folder / f"box_{i}.stl"), box)
    return folder


def test_a_product_smaller_than_5_mm_or_larger_than_5_m_gets_a_warning(
    tmp_path, detection
):
    client = TestClient(create_app())
    small = _imported(client, _box_folder(tmp_path, 1.0))["warnings"]
    large = _imported(client, _box_folder(tmp_path, 6000.0))["warnings"]
    normal = _imported(client, _box_folder(tmp_path, 100.0))["warnings"]
    assert any("mm across" in w for w in small)
    assert any("m across" in w and "mm across" not in w for w in large)
    assert not any("across" in w for w in normal)


def test_parts_with_no_contacts_are_listed_in_a_warning(tmp_path, meshes, detection):
    detection.contacts = [
        c for c in detection.contacts if "04_screw" not in (c.part_name1, c.part_name2)
    ]
    warnings = _imported(TestClient(create_app()), meshes)["warnings"]
    assert warnings == ["Parts with no contacts: 04_screw"]


def test_a_product_where_everything_touches_has_no_warning(meshes, detection):
    assert _imported(TestClient(create_app()), meshes)["warnings"] == []


def test_first_save_needs_a_target_folder_and_a_new_one(meshes, detection, tmp_path):
    client = TestClient(create_app())
    _imported(client, meshes)
    assert client.post("/api/save").status_code == 400
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "other.txt").write_text("x")
    assert client.post("/api/save", json={"folder": str(taken)}).status_code == 400
    assert client.get("/api/state").json()["unsaved"] is True
    assert list(taken.iterdir()) == [taken / "other.txt"]


def test_first_save_writes_the_product_folder_and_it_loads_from_anywhere(
    meshes, detection, tmp_path, monkeypatch
):
    client = TestClient(create_app())
    _imported(client, meshes)
    client.post(
        "/api/setup", json={"base_parts": ["01_base"], "target_parts": ["02_target"]}
    )
    client.post("/api/rename", json={"part": "04_screw", "name": "05_screw"})
    target = tmp_path / "products" / "mine"
    assert client.post("/api/save", json={"folder": str(target)}).status_code == 200
    view = client.get("/api/state").json()
    assert view["unsaved"] is False and view["product"]["folder"] == str(target)
    text = (target / "mesh_folder.xml").read_text(encoding="utf-8")
    assert "\\" not in text and not re.search(r'="[A-Za-z]:|="/', text)
    paths = re.findall(r'(?:mesh_path|_filename)="([^"]*)"', text)
    assert paths and all((target / p).is_file() for p in paths)
    assert sorted(p.name for p in (target / "mesh_folder").glob("*.stl")) == [
        f"{n}.stl" for n in PARTS
    ]

    shutil.rmtree(meshes)  # the product no longer depends on the import folder
    moved = shutil.move(str(target), str(tmp_path / "elsewhere"))
    monkeypatch.chdir(tmp_path)
    reopened = TestClient(create_app())
    loaded = reopened.post("/api/load", json={"folder": str(moved)}).json()
    assert sorted(loaded["product"]["parts"]) == sorted(
        [*(set(PARTS) - {"04_screw"}), "05_screw"]
    )
    assert loaded["setup"] == {"base_parts": ["01_base"], "target_parts": ["02_target"]}
    again = load_product_folder(Path(moved))[1]
    assert len(again.connections) == len(detection.contacts)


def test_undo_after_the_first_save_still_writes_portable_paths(
    meshes, detection, tmp_path
):
    client = TestClient(create_app())
    _imported(client, meshes)
    client.post("/api/rename", json={"part": "04_screw", "name": "05_screw"})
    target = tmp_path / "mine"
    client.post("/api/save", json={"folder": str(target)})
    client.post("/api/undo")
    shutil.rmtree(meshes)
    assert client.post("/api/save").status_code == 200
    text = (target / "mesh_folder.xml").read_text(encoding="utf-8")
    assert str(meshes).replace("\\", "/") not in text.replace("\\", "/")
    assert all(
        (target / p).is_file()
        for p in re.findall(r'(?:mesh_path|_filename)="([^"]*)"', text)
    )


def test_an_unsaved_import_refuses_export_and_store(meshes, detection, tmp_path):
    client = TestClient(create_app())
    _imported(client, meshes)
    assert client.post("/api/store").status_code == 409
    assert (
        client.post("/api/export", json={"out_dir": str(tmp_path / "out")}).status_code
        == 409
    )
    assert client.post("/api/jump", json={"node": "s0"}).status_code == 409


@pytest.mark.geom
@pytest.mark.slow
def test_real_contact_detection_finds_the_contacts_of_the_tiny_product(meshes):
    client = TestClient(create_app())
    view = _imported(client, meshes)
    assert view["job"]["error"] is None
    assert sorted(view["product"]["parts"]) == PARTS
