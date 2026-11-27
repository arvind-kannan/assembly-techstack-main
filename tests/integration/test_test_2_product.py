"""The 18-part upload opens portably and supports a bounded parallel graph check."""

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from product_explorer.server import create_app

CASE = Path(__file__).resolve().parents[2] / "cases" / "test_2"


@pytest.mark.geom
@pytest.mark.integration
def test_test_2_ui_loads_all_parts_and_generates_small_parallel_graph(tmp_path):
    case = tmp_path / "test_2"
    shutil.copytree(
        CASE, case, ignore=shutil.ignore_patterns("test_2_data", "__pycache__")
    )
    subprocess.run(
        [sys.executable, str(case / "run_ui.py"), "--prepare-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((case / "import_manifest.json").read_text())
    assert len(manifest["original_sha256"]) == 18
    for name, digest in manifest["original_sha256"].items():
        assert (
            hashlib.sha256((case / "source" / name).read_bytes()).hexdigest() == digest
        )

    client = TestClient(create_app())
    response = client.post("/api/load", json={"folder": str(case / "model")})
    assert response.status_code == 200, response.text
    view = response.json()
    names = {str(i) for i in range(18)}
    assert set(view["product"]["parts"]) == names
    assert view["setup"]["base_parts"] == ["8"]
    assert set(view["setup"]["target_parts"]) == names - {"8"}
    assert len(view["connections"]) == len(manifest["contact_pairs"])
    assert client.get("/api/meshes").status_code == 200

    # Bound the smoke check to two targets; the checked-in setup still has 17.
    response = client.post(
        "/api/setup",
        json={"base_parts": sorted(names - {"10", "15"}), "target_parts": ["10", "15"]},
    )
    assert response.status_code == 200, response.text
    response = client.post(
        "/api/generate",
        json={
            "parallel": True,
            "n_robots": 2,
            "min_separation": 50,
            "extraction_depth": "full_teardown",
        },
    )
    assert response.status_code == 202, response.text
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        view = client.get("/api/state").json()
        if not view["job"]["running"]:
            break
        time.sleep(0.05)
    assert not view["job"]["running"], "bounded graph check timed out"
    assert view["job"]["error"] is None
    graph = view["graph"]
    assert len(graph["nodes"]) <= 4
    assert any(
        n["is_goal"] and set(n["parts_present"]) == names - {"10", "15"}
        for n in graph["nodes"].values()
    )
    assert any(e.get("parallel") for e in graph["edges"].values())
