"""The tiny test product (#239): a product folder kept with the tests so the explorer's API tests run in the
merge gate. Four box parts, one screw connection, committed contact data, a product setup with a base part
and a target part. Seam: the explorer's HTTP API; every test works on a temporary copy.

Regenerate the folder with `tools/make_tiny_test_product.py`; no test runs contact detection.
"""

import hashlib
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.connection import Geometric_Connection, Screw_Connection
from fastapi.testclient import TestClient

from disassembly_pomdp.env_builder import build_env
from product_explorer.product_folder import load_product_folder
from product_explorer.server import create_app

TINY = Path(__file__).resolve().parents[1] / "test_products" / "tiny_test_product"
PARTS = ["01_base", "02_target", "03_cover", "04_screw"]
SETUP = {"base_parts": ["01_base"], "target_parts": ["02_target"]}


def _tree_hashes(folder: Path) -> dict:
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def tiny_copy(tmp_path) -> Path:
    folder = tmp_path / "tiny_test_product"
    shutil.copytree(TINY, folder)
    return folder


@pytest.fixture(autouse=True)
def no_contact_detection(monkeypatch):
    """The test product ships its contact data, so nothing here may detect contacts."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("contact detection ran")

    monkeypatch.setattr(Assembly, "find_contacts", refuse)
    monkeypatch.setattr(Assembly, "check_pair_for_contact_new", refuse)


def _generate(client) -> dict:
    """Start Generate graph and follow its progress over the WebSocket until it is done."""
    assert client.post("/api/generate").status_code == 202
    with client.websocket_connect("/ws/job") as ws:
        while True:
            job = ws.receive_json()
            if not job["running"]:
                break
    assert job["error"] is None, job["error"]
    return client.get("/api/state").json()


def test_the_test_product_loads_through_the_api_with_its_setup(tiny_copy):
    client = TestClient(create_app())
    response = client.post("/api/load", json={"folder": str(tiny_copy)})
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["product"]["parts"] == PARTS
    assert view["setup"] == SETUP


def test_its_product_model_builds_an_env_through_the_shared_env_builder(tiny_copy):
    _xml, assembly, base, target = load_product_folder(tiny_copy)
    env = build_env(tiny_copy, assembly=assembly, base_parts=base, target_parts=target)
    assert sorted(env.assembly.parts) == PARTS
    assert (
        sorted(build_env(tiny_copy).assembly.parts) == PARTS
    )  # and straight from the folder


def test_it_has_a_screw_connection_led_by_the_screw_and_a_geometric_connection(
    tiny_copy,
):
    _xml, assembly, _base, _target = load_product_folder(tiny_copy)
    screws = [c for c in assembly.connections if isinstance(c, Screw_Connection)]
    assert len(screws) == 1 and screws[0].part_name1 == "04_screw"
    assert any(isinstance(c, Geometric_Connection) for c in assembly.connections)


def test_the_sequence_graph_has_an_available_action_from_the_base_state(tiny_copy):
    client = TestClient(create_app())
    assert client.post("/api/load", json={"folder": str(tiny_copy)}).status_code == 200
    view = _generate(client)
    assert view["current"] == view["graph"]["root"]
    assert view["next_steps"]
    assert view["graph"]["edges"][view["next_steps"][0]]["target_part"] == "02_target"


def test_working_on_a_copy_leaves_the_committed_test_product_byte_identical(tiny_copy):
    before = _tree_hashes(TINY)
    client = TestClient(create_app())
    assert client.post("/api/load", json={"folder": str(tiny_copy)}).status_code == 200
    assert client.post("/api/save").status_code == 200
    assert client.post("/api/store").status_code == 200
    assert _tree_hashes(TINY) == before


def test_the_tests_that_use_it_are_in_the_fast_suite(request):
    here = [i for i in request.session.items if i.module is request.module]
    assert len(here) >= 6
    for item in here:
        marks = {m.name for m in item.iter_markers()}
        assert not marks & {"slow", "geom", "gpu"}, item.name
