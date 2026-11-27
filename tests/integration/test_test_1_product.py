"""The uploaded test_1 product opens and disassembles through the real explorer API."""

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

CASE = Path(__file__).resolve().parents[2] / "cases" / "test_1"


@pytest.mark.geom
@pytest.mark.integration
def test_test_1_portable_ui_parallel_disassembly(tmp_path):
    case = tmp_path / "test_1"
    shutil.copytree(
        CASE, case, ignore=shutil.ignore_patterns("test_1_data", "__pycache__")
    )
    subprocess.run(
        [sys.executable, str(case / "run_ui.py"), "--prepare-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((case / "import_manifest.json").read_text())
    for name, digest in manifest["original_sha256"].items():
        assert (
            hashlib.sha256((case / "source" / name).read_bytes()).hexdigest() == digest
        )

    client = TestClient(create_app())
    opened = client.post("/api/load", json={"folder": str(case / "model")})
    assert opened.status_code == 200, opened.text
    view = opened.json()
    assert set(view["product"]["parts"]) == {"0", "1", "2", "3", "4"}
    assert view["setup"] == {"base_parts": ["0"], "target_parts": ["1", "2", "3", "4"]}
    assert len(view["connections"]) == 4
    assert client.get("/api/meshes").status_code == 200

    generated = client.post(
        "/api/generate",
        json={
            "parallel": True,
            "n_robots": 2,
            "min_separation": 50,
            "extraction_depth": "full_teardown",
        },
    )
    assert generated.status_code == 202, generated.text
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        view = client.get("/api/state").json()
        if not view["job"]["running"]:
            break
        time.sleep(0.05)
    assert not view["job"]["running"], "graph generation timed out"
    assert view["job"]["error"] is None
    graph = view["graph"]
    assert any(
        n["is_goal"] and n["parts_present"] == ["0"] for n in graph["nodes"].values()
    )
    assert any(e.get("parallel") for e in graph["edges"].values())
    assert all("0" in n["parts_present"] for n in graph["nodes"].values())
