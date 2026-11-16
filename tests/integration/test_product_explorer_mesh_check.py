"""A part without a usable mesh (#266): the Product explorer names it when a product is imported or opened and
refuses the export while it stands. Seam: the explorer's HTTP API.

The core already refuses a mesh file that is empty or unreadable, so the flagged product here is the tiny test
product with the mesh of its screw swapped for one with a NaN vertex: it loads, but its vertex mean is not a
number. The untouched copy is the control.
"""

import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from product_explorer.server import create_app
from product_explorer.session import ExplorerSession
from tests.integration.test_product_explorer_import import (  # noqa: F401 (fixtures)
    TINY,
    _imported,
    detection,
    meshes,
)

NAN_OBJ = "v nan 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"


def _break_mesh(mesh_folder: Path, part: str) -> None:
    (mesh_folder / f"{part}.stl").unlink()
    (mesh_folder / f"{part}.obj").write_text(NAN_OBJ)


def _break_product(folder: Path, *parts: str) -> None:
    for part in parts:
        _break_mesh(folder / "tiny_test_product", part)
    xml = folder / "tiny_test_product.xml"
    text = xml.read_text(encoding="utf-8")
    for part in parts:
        text = text.replace(f"{part}.stl", f"{part}.obj")
    xml.write_text(text, encoding="utf-8")


@pytest.fixture
def flagged(tmp_path) -> Path:
    folder = tmp_path / "flagged"
    shutil.copytree(TINY, folder)
    _break_product(folder, "04_screw")
    return folder


@pytest.fixture
def control(tmp_path) -> Path:
    folder = tmp_path / "control"
    shutil.copytree(TINY, folder)
    return folder


def _open(client, folder: Path) -> dict:
    response = client.post("/api/load", json={"folder": str(folder)})
    assert response.status_code == 200, response.text
    return response.json()


def _generate(client) -> dict:
    assert client.post("/api/generate").status_code == 202
    for _ in range(600):
        view = client.get("/api/state").json()
        if not view["job"]["running"]:
            assert view["job"]["error"] is None, view["job"]["error"]
            return view
        time.sleep(0.05)
    raise AssertionError("the graph did not finish")


def test_opening_a_product_folder_names_the_part_with_no_usable_mesh(flagged):
    view = _open(TestClient(create_app()), flagged)
    assert view["warnings"] == ["Parts with no usable mesh: 04_screw"]
    assert view["product"]["parts"]  # the product still opens


def test_the_control_product_opens_with_no_message(control):
    assert _open(TestClient(create_app()), control)["warnings"] == []


def test_opening_another_product_clears_the_message(flagged, control):
    client = TestClient(create_app())
    _open(client, flagged)
    assert _open(client, control)["warnings"] == []


def test_several_parts_without_a_usable_mesh_are_one_message_naming_each_once(
    tmp_path,
):
    folder = tmp_path / "two"
    shutil.copytree(TINY, folder)
    _break_product(folder, "03_cover", "04_screw")
    warnings = _open(TestClient(create_app()), folder)["warnings"]
    assert warnings == ["Parts with no usable mesh: 03_cover, 04_screw"]


def test_importing_a_mesh_folder_names_the_part_with_no_usable_mesh(
    meshes,  # noqa: F811
    detection,  # noqa: F811
):
    _break_mesh(meshes, "04_screw")
    warnings = _imported(TestClient(create_app()), meshes)["warnings"]
    assert warnings == ["Parts with no usable mesh: 04_screw"]


def test_a_mesh_file_that_is_empty_is_still_refused_naming_the_file(control):
    (control / "tiny_test_product" / "04_screw.stl").write_bytes(b"")
    response = TestClient(create_app()).post("/api/load", json={"folder": str(control)})
    assert response.status_code == 400 and "04_screw.stl" in response.json()["detail"]


def test_a_flagged_product_refuses_the_export_naming_the_part(flagged, tmp_path):
    client = TestClient(create_app())
    _open(client, flagged)  # no graph needed: the refusal comes first
    response = client.post("/api/export", json={"out_dir": str(tmp_path / "out")})
    assert response.status_code == 400
    assert "04_screw" in response.json()["detail"]


def test_the_export_succeeds_once_the_mesh_is_fixed(flagged, tmp_path):
    client = TestClient(create_app())
    _open(client, flagged)
    out = tmp_path / "out"
    assert client.post("/api/export", json={"out_dir": str(out)}).status_code == 400
    shutil.rmtree(flagged)
    shutil.copytree(TINY, flagged)
    _open(client, flagged)
    _generate(client)
    client.post("/api/store")
    assert client.post("/api/export", json={"out_dir": str(out)}).status_code == 200


def test_a_screw_group_naming_a_part_that_is_not_in_the_product_is_reported_by_generate_and_export(
    control, tmp_path, monkeypatch
):
    session = ExplorerSession()
    session.load(control)
    session.generate().join(timeout=60)
    assert session.job.error is None and session.warnings == []
    monkeypatch.setattr(session, "_collapsed_groups", lambda: [["04_screw", "ghost"]])
    session.generate().join(timeout=60)
    assert session.warnings == [
        "Screw group names parts that are not in the product: ghost"
    ]
    session.store()
    session.export(tmp_path / "out")  # a warning, not a refusal
    assert len(session.warnings) == 1  # reported once, however often it is checked


@pytest.mark.parametrize("case", ["angle_grinder", "battery_replacement"])
def test_the_grinder_and_battery_products_open_with_no_message(case):
    folder = Path(__file__).resolve().parents[2] / "cases" / case / "model"
    assert _open(TestClient(create_app()), folder)["warnings"] == []


def test_part_positions_leave_out_a_part_whose_real_mesh_has_no_finite_position(
    flagged,
):
    from mrtamp.screw_groups import part_positions

    folder = flagged
    positions = part_positions(
        (folder / "tiny_test_product.xml").read_text(encoding="utf-8"), folder
    )
    assert "04_screw" not in positions and "01_base" in positions


def test_importing_an_empty_mesh_file_fails_the_job_naming_the_file(
    meshes,  # noqa: F811
    detection,  # noqa: F811
):
    (meshes / "04_screw.stl").write_bytes(b"")
    client = TestClient(create_app())
    assert client.post("/api/import", json={"folder": str(meshes)}).status_code == 202
    for _ in range(600):
        job = client.get("/api/state").json()["job"]
        if not job["running"]:
            break
        time.sleep(0.05)
    assert "04_screw.stl" in job["error"]
