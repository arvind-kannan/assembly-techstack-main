"""The Product explorer server (#134) against the real build_env, on a copy of the EMO product folder."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from disassembly_pomdp.env_builder import build_env
from disassembly_pomdp.graph_export import read_export
from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent
from disassembly_pomdp.graph_reversal import complete_by_reversal
from mrtamp.exported_graph import load_export
from product_explorer.server import create_app

ROOT = Path(__file__).resolve().parents[2]
EMO = ROOT / "emo_sr_v2"

pytestmark = pytest.mark.slow


def _steps(graph) -> list:
    return sorted(
        (u, v, s["target_part"], bool(s.get("inverse")))
        for u, v, s in graph.edges(data=True)
    )


def generate(client) -> dict:
    """Start Generate graph and follow its progress over the WebSocket until it is done."""
    assert client.post("/api/generate").status_code == 202
    with client.websocket_connect("/ws/job") as ws:
        while True:
            job = ws.receive_json()
            if not job["running"]:
                break
    assert job["error"] is None, job["error"]
    assert job["expanded"] > 0
    return client.get("/api/state").json()


@pytest.fixture(scope="module")
def emo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("products") / "emo_sr_v2"
    shutil.copytree(EMO, folder)
    client = TestClient(create_app())
    view = client.post("/api/load", json={"folder": str(folder)}).json()
    assert view["graph"] is None
    return client, folder, generate(client)


def test_load_shows_the_product_and_its_setup(emo):
    client, _, view = emo
    assert view["product"]["name"] == "EMO_SR_V2"
    assert len(view["product"]["parts"]) == 8
    assert view["setup"] == {
        "base_parts": ["08_lower_lid", "02_stator"],
        "target_parts": ["03_rotor_short"],
    }
    meshes = client.get("/api/meshes").json()
    assert set(meshes["parts"]) == set(view["product"]["parts"])
    assert meshes["scene_radius"] > 0 and len(meshes["scene_center"]) == 3


def test_bad_folder_is_refused(emo, tmp_path):
    client, _, _ = emo
    response = client.post("/api/load", json={"folder": str(tmp_path)})
    assert response.status_code == 400 and "assembly XML" in response.json()["detail"]


def test_generated_graph_is_the_collapsed_extraction(emo):
    _, folder, view = emo
    expected = DisassemblyGraphAgent(build_env(folder)).build_graph(
        verbose=False, collapse_screw_groups=True
    )
    graph = view["graph"]
    assert not graph["stale"]
    assert len(graph["nodes"]) == expected.number_of_nodes()
    assert len(graph["edges"]) == expected.number_of_edges()
    assert graph["collapsed_screw_groups"] == expected.graph["collapsed_screw_groups"]
    assert view["current"] == graph["root"] and view["path"] == []
    assert {graph["edges"][e]["target_part"] for e in view["next_steps"]} == {
        s["target_part"]
        for _, _, s in expected.out_edges(
            next(n for n, d in expected.nodes(data=True) if d["depth"] == 0), data=True
        )
    }


def test_step_back_restart_and_jump(emo):
    client, _, view = emo
    first = view["next_steps"][0]
    view = client.post("/api/step", json={"edge": first}).json()
    assert (
        view["path"] == [first]
        and view["current"] == view["graph"]["edges"][first]["target"]
    )
    assert view["graph"]["edges"][first]["target_part"] not in view["parts_present"]
    view = client.post("/api/back").json()
    assert view["path"] == [] and view["current"] == view["graph"]["root"]
    goal = next(n for n, d in view["graph"]["nodes"].items() if d["is_goal"])
    view = client.post("/api/jump", json={"node": goal}).json()
    assert (
        view["current"] == goal
        and len(view["path"]) == view["graph"]["nodes"][goal]["depth"]
    )
    view = client.post("/api/restart").json()
    assert view["current"] == view["graph"]["root"] and view["path"] == []
    assert client.post("/api/step", json={"edge": "no-such-step"}).status_code == 400


def test_rejection_reason_of_a_picked_part(emo):
    client, _, _ = emo
    client.post("/api/restart")
    screw = client.get("/api/rejection", params={"part": "04_screw_1"}).json()
    assert screw["available"], screw[
        "text"
    ]  # checked like the extraction: no EOL sampling
    stator = client.get("/api/rejection", params={"part": "02_stator"}).json()
    assert not stator["available"] and "base part" in stator["text"]
    lid = client.get("/api/rejection", params={"part": "01_upper_lid"}).json()
    assert not lid["available"] and "screw" in lid["text"]


def test_rec_writes_the_path_as_a_sequence_graph_export(emo, tmp_path):
    client, folder, view = emo
    client.post("/api/restart")
    client.post("/api/rec", json={"on": True})
    for _ in range(2):
        view = client.post("/api/step", json={"edge": view["next_steps"][0]}).json()
    assert view["recorded"] == view["path"] and len(view["recorded"]) == 2
    view = client.post("/api/rec", json={"on": False}).json()
    out = tmp_path / "sequence.json"
    assert client.post("/api/rec/save", json={"path": str(out)}).status_code == 200
    export = load_export(
        out, product_folder=folder
    )  # the MRTAMP app reads it as any sequence graph export
    assert not export.warnings and export.graph.number_of_nodes() == 3
    removals = [
        s["target_part"]
        for _, _, s in export.graph.edges(data=True)
        if not s.get("inverse")
    ]
    assert removals == [
        view["graph"]["edges"][e]["target_part"] for e in view["recorded"]
    ]
    assert (
        export.provenance["product_name"] == "EMO_SR_V2"
        and export.provenance["fingerprints"]["product_setup"]
    )


def test_export_writes_graph_urdfs_and_manifest(emo, tmp_path):
    client, folder, view = emo
    response = client.post("/api/export", json={"out_dir": str(tmp_path)})
    assert response.status_code == 200, response.text
    graph, provenance = read_export(tmp_path / "sequence_graph.json")
    expected = complete_by_reversal(
        DisassemblyGraphAgent(build_env(folder)).build_graph(
            verbose=False, collapse_screw_groups=True
        )
    )
    assert set(graph.nodes) == set(
        expected.nodes
    )  # the exported file is the complete (reversed) extraction
    assert _steps(graph) == _steps(expected)
    assert provenance["product_model_xml"] == "EMO_SR_V2.xml"
    text = (tmp_path / "sequence_graph.json").read_text(encoding="utf-8")
    assert not Path(
        provenance["product_folder"]
    ).is_absolute()  # portable provenance (#246)
    assert str(folder) not in text and str(folder).replace("\\", "\\\\") not in text
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["fingerprints"] == provenance["fingerprints"]
    assert sorted(p.stem for p in (tmp_path / "urdf").glob("*.urdf")) == sorted(
        view["product"]["parts"]
    )
    urdf = (tmp_path / "urdf" / "02_stator.urdf").read_text(encoding="utf-8")
    assert str(tmp_path) not in urdf and "../mesh/" in urdf  # portable mesh paths


def test_editing_targets_marks_the_graph_stale_until_generated(emo, tmp_path):
    client, folder, _ = emo
    view = client.post(
        "/api/setup",
        json={
            "base_parts": ["08_lower_lid", "02_stator"],
            "target_parts": ["01_upper_lid"],
        },
    ).json()
    assert (
        view["graph"]["stale"] and view["next_steps"] == [] and not view["setup_stored"]
    )
    assert (
        client.post("/api/export", json={"out_dir": str(tmp_path)}).status_code == 409
    )
    view = generate(client)
    assert not view["graph"]["stale"]
    assert (
        client.post("/api/export", json={"out_dir": str(tmp_path)}).status_code == 409
    )  # setup not stored
    view = client.post("/api/store").json()
    assert view["setup_stored"]
    stored = json.loads((folder / "product_setup.json").read_text(encoding="utf-8"))
    assert stored["target_parts"] == ["01_upper_lid"]
    assert stored[
        "stuck_part_rules"
    ]  # fields the explorer does not edit survive the store
    assert (
        client.post("/api/export", json={"out_dir": str(tmp_path)}).status_code == 200
    )


def test_the_product_root_is_set_shown_and_recorded_in_the_export(
    emo, tmp_path, monkeypatch
):
    client, folder, _ = emo
    session = client.app.state.session
    monkeypatch.setattr(session, "config_dir", tmp_path / "config")
    monkeypatch.delenv("PRODUCT_EXPLORER_PRODUCT_ROOT", raising=False)
    assert client.get("/api/state").json()["product_root"]["path"] is None

    view = client.post("/api/product-root", json={"path": str(folder.parent)}).json()
    assert view["product_root"]["path"] == str(folder.parent.resolve())
    assert view["product_root"]["source"] == "app setting"
    out = tmp_path / "out"
    assert client.post("/api/export", json={"out_dir": str(out)}).status_code == 200
    _, provenance = read_export(out / "sequence_graph.json")
    assert provenance["product_folder"] == "emo_sr_v2"
    assert provenance["path_bases"]["product_folder"] == "product_root"
    assert str(folder.parent) not in (out / "sequence_graph.json").read_text(
        encoding="utf-8"
    )

    bad = client.post("/api/product-root", json={"path": "relative"})
    assert bad.status_code == 400
    view = client.post("/api/product-root", json={"path": None}).json()
    assert view["product_root"]["setting"] is None
