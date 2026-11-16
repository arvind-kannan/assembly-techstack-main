"""The extraction depth in the Product explorer (AT 269), on a copy of the tiny test product.

Seam: the HTTP API and the session behind it. The tiny product's target part comes out first, and the cover
and the screw after it, so a full teardown has steps out of the disassembled states that the default depth
leaves out.
"""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from diss_assembly_dataclasses.assembly import Assembly
from fastapi.testclient import TestClient

from disassembly_pomdp.graph_extraction import DISASSEMBLED_STATES, FULL_TEARDOWN
from product_explorer.server import create_app
from product_explorer.session import ExplorerError, ExplorerSession

TINY = Path(__file__).resolve().parents[1] / "test_products" / "tiny_test_product"


@pytest.fixture(autouse=True)
def no_contact_detection(monkeypatch):
    """The test product ships its contact data, so nothing here may detect contacts."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("contact detection ran")

    monkeypatch.setattr(Assembly, "find_contacts", refuse)
    monkeypatch.setattr(Assembly, "check_pair_for_contact_new", refuse)


@pytest.fixture
def session(tmp_path) -> ExplorerSession:
    copy = tmp_path / "tiny_test_product"
    shutil.copytree(TINY, copy)
    session = ExplorerSession()
    session.load(copy)
    return session


@pytest.fixture
def client(session) -> TestClient:
    return TestClient(create_app(session))


def _generate(client, body=None) -> dict:
    response = (
        client.post("/api/generate")
        if body is None
        else client.post("/api/generate", json=body)
    )
    assert response.status_code == 202, response.text
    with client.websocket_connect("/ws/job") as ws:
        while True:
            job = ws.receive_json()
            if not job["running"]:
                break
    assert job["error"] is None, job["error"]
    return client.get("/api/state").json()


def _steps_out_of_disassembled_states(graph: dict) -> list[str]:
    disassembled = {i for i, n in graph["nodes"].items() if n.get("is_goal")}
    assert disassembled
    return [e for e, s in graph["edges"].items() if s["source"] in disassembled]


def test_by_default_generate_graph_stops_at_the_disassembled_states(client):
    graph = _generate(client)["graph"]

    assert graph["extraction_depth"] == DISASSEMBLED_STATES
    assert _steps_out_of_disassembled_states(graph) == []


def test_a_full_teardown_is_a_choice_when_the_graph_is_generated(client):
    default = _generate(client)["graph"]
    view = _generate(client, {"extraction_depth": FULL_TEARDOWN})
    graph = view["graph"]

    assert graph["extraction_depth"] == FULL_TEARDOWN
    assert _steps_out_of_disassembled_states(graph)
    assert len(graph["edges"]) > len(default["edges"])
    assert not graph["stale"]  # the depth is no edit of the product setup
    assert view["setup"] == {"base_parts": ["01_base"], "target_parts": ["02_target"]}


def test_an_unknown_depth_is_refused(client):
    response = client.post("/api/generate", json={"extraction_depth": "halfway"})

    assert response.status_code == 400 and "depth" in response.json()["detail"]


@pytest.mark.parametrize("depth", [DISASSEMBLED_STATES, FULL_TEARDOWN])
def test_the_export_records_the_depth_of_the_graph(session, tmp_path, depth):
    session.generate(depth).join()
    assert session.job.error is None
    session.store()

    out = session.export(tmp_path / "out")

    for name in ("sequence_graph.json", "manifest.json"):
        provenance = json.loads((out / name).read_text(encoding="utf-8"))["provenance"]
        assert provenance["extraction_depth"] == depth, name


def test_an_unknown_depth_is_refused_by_the_session(session):
    with pytest.raises(ExplorerError, match="depth"):
        session.generate("halfway")
