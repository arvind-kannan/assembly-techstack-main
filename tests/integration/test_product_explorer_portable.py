"""The Product explorer's portable product folder (#238): load repairs legacy absolute paths, save writes
relative forward-slash paths, and a saved folder loads from anywhere. Seam: the explorer's HTTP API.

Uses a copy of the EMO product folder (no meshes are loaded, so the tests stay in the merge gate).
"""

import hashlib
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from product_explorer.server import create_app

EMO = Path(__file__).resolve().parents[2] / "emo_sr_v2"
PARTS = [
    "01_upper_lid",
    "02_stator",
    "03_rotor_short",
    "04_screw_1",
    "05_screw_2",
    "06_screw_3",
    "07_screw_4",
    "08_lower_lid",
]


def _copy_emo(tmp_path: Path, name: str = "emo_sr_v2") -> Path:
    folder = tmp_path / name
    shutil.copytree(EMO, folder)
    return folder


def _make_legacy(folder: Path) -> None:
    """Rewrite the product model XML the way an old Windows machine wrote it: absolute backslash paths
    to meshes and contact data that do not exist here (the files stay next to the XML)."""
    xml = folder / "EMO_SR_V2.xml"
    text = xml.read_text(encoding="utf-8")
    text = re.sub(
        r'mesh_path="EMO_SR_V2/([^"]+)"',
        lambda m: 'mesh_path="C:\\\\old_pc\\\\EMO_SR_V2\\\\' + m[1] + '"',
        text,
    )
    text = re.sub(
        r'_filename="EMO_SR_V2_data/([^"]+)"',
        lambda m: '_filename="D:\\\\old_pc\\\\EMO_SR_V2_data\\\\' + m[1] + '"',
        text,
    )
    xml.write_text(text, encoding="utf-8")


def _tree_hashes(folder: Path) -> dict:
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _load(client, folder: Path):
    return client.post("/api/load", json={"folder": str(folder)})


def test_legacy_absolute_paths_load_and_are_repaired_by_file_name(tmp_path):
    folder = _copy_emo(tmp_path)
    _make_legacy(folder)
    client = TestClient(create_app())
    response = _load(client, folder)
    assert response.status_code == 200, response.text
    view = response.json()
    assert set(view["product"]["parts"]) == set(PARTS)
    assert view["setup"] == {
        "base_parts": ["08_lower_lid", "02_stator"],
        "target_parts": ["03_rotor_short"],
    }


def test_a_mesh_missing_next_to_the_xml_is_a_user_error_naming_the_part(tmp_path):
    folder = _copy_emo(tmp_path)
    _make_legacy(folder)
    (folder / "EMO_SR_V2" / "02_stator.stl").unlink()
    response = _load(TestClient(create_app()), folder)
    assert response.status_code == 400
    assert "02_stator" in response.json()["detail"]


def test_a_folder_with_two_xmls_is_refused(tmp_path):
    folder = _copy_emo(tmp_path)
    shutil.copy(folder / "EMO_SR_V2.xml", folder / "other.xml")
    response = _load(TestClient(create_app()), folder)
    assert (
        response.status_code == 400
        and "exactly one assembly XML" in response.json()["detail"]
    )


def test_save_writes_every_path_relative_with_forward_slashes(tmp_path):
    folder = _copy_emo(tmp_path)
    _make_legacy(folder)
    client = TestClient(create_app())
    _load(client, folder)
    assert client.post("/api/save").status_code == 200
    text = (folder / "EMO_SR_V2.xml").read_text(encoding="utf-8")
    assert "C:" not in text and "D:" not in text and "\\" not in text
    paths = re.findall(r'(?:mesh_path|_filename)="([^"]*)"', text)
    assert paths and all(not Path(p).is_absolute() for p in paths)
    assert all(
        (folder / p).is_file() for p in paths
    )  # every relative path resolves next to the XML


def test_a_saved_legacy_product_loads_again_and_keeps_its_parts_and_setup(tmp_path):
    folder = _copy_emo(tmp_path)
    _make_legacy(folder)
    client = TestClient(create_app())
    before = _load(client, folder).json()
    client.post("/api/save")
    after = _load(TestClient(create_app()), folder).json()
    assert after["product"]["parts"] == before["product"]["parts"]
    assert after["setup"] == before["setup"]


def test_saved_folder_moved_elsewhere_loads_the_same_from_another_working_directory(
    tmp_path, monkeypatch
):
    folder = _copy_emo(tmp_path)
    _make_legacy(folder)
    client = TestClient(create_app())
    before = _load(client, folder).json()
    client.post("/api/save")
    moved = shutil.move(str(folder), str(tmp_path / "elsewhere" / "emo_sr_v2"))
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    after = _load(TestClient(create_app()), Path(moved)).json()
    assert after["product"]["parts"] == before["product"]["parts"]
    assert after["setup"] == before["setup"]


def test_saving_writes_the_product_model_and_leaves_the_setup_file_unchanged(tmp_path):
    folder = _copy_emo(tmp_path)
    client = TestClient(create_app())
    _load(client, folder)
    setup_before = (folder / "product_setup.json").read_bytes()
    assert client.post("/api/save").status_code == 200
    assert (folder / "product_setup.json").read_bytes() == setup_before


def test_opening_and_exploring_writes_nothing_to_the_product_folder(tmp_path):
    folder = _copy_emo(tmp_path)
    _make_legacy(folder)
    client = TestClient(create_app())
    before = _tree_hashes(folder)
    _load(client, folder)
    assert (
        client.get("/api/rejection", params={"part": "04_screw_1"}).status_code == 200
    )
    assert _tree_hashes(folder) == before


def test_save_without_a_loaded_product_is_a_conflict(tmp_path):
    assert TestClient(create_app()).post("/api/save").status_code == 409
