"""The Product explorer server (#134) on the battery replacement product: 51 parts, 41 interchangeable lid screws.

A module of its own so the EMO session of the other server tests is freed before the battery loads.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from product_explorer.server import create_app
from tests.integration.test_product_explorer_server import generate

BATTERY = (
    Path(__file__).resolve().parents[2] / "cases" / "battery_replacement" / "model"
)

pytestmark = pytest.mark.slow


def test_battery_loads_and_its_graph_is_generated():
    client = TestClient(create_app())
    view = client.post("/api/load", json={"folder": str(BATTERY)}).json()
    assert len(view["product"]["parts"]) == 51
    assert len(client.get("/api/meshes").json()["parts"]) == 51
    screw = client.get(
        "/api/rejection", params={"part": "lid_screw_07"}
    ).json()  # before any graph
    assert screw["available"], screw["text"]
    view = generate(client)
    graph = view["graph"]
    assert len(graph["nodes"]) == 43 and len(graph["collapsed_screw_groups"][0]) == 41
    assert any(n["is_goal"] for n in graph["nodes"].values())
    screw = client.get("/api/rejection", params={"part": "lid_screw_07"}).json()
    assert screw["available"] and "interchangeable screws" in screw["text"]
