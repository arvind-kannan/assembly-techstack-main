"""Parallel analysis through the Product explorer API, without external assets."""

import pytest
from fastapi.testclient import TestClient

from product_explorer.server import create_app
from product_explorer.session import ExplorerError, ExplorerSession
from tests.unit.test_screw_macro import screw_agent


@pytest.fixture
def explorer(monkeypatch):
    agent = screw_agent(["A", "B", "C", "D"])
    for part in agent.base_assembly.parts.values():
        part.material = "Generic"
        part.density = 8000
    session = ExplorerSession()
    session.assembly = agent.base_assembly
    session.name = "synthetic"
    session.setup = {"base_parts": ["Base"], "target_parts": ["A", "B", "C", "D"]}
    monkeypatch.setattr("product_explorer.session._agent_for", lambda *args: agent)
    monkeypatch.setattr(
        "disassembly_pomdp.graph_extraction.check_parallel_geometry",
        lambda *args: (True, {"reason": None}),
    )
    return session, agent, TestClient(create_app(session))


def generate(session, client, **kwargs):
    response = client.post("/api/generate", json=kwargs)
    assert response.status_code == 202, response.text
    with client.websocket_connect("/ws/job") as socket:
        while socket.receive_json()["running"]:
            pass
    assert session.job.error is None, session.job.error
    return client.get("/api/state").json()


def test_parallel_controls_macro_shortest_and_stepping(explorer):
    session, agent, client = explorer
    view = generate(session, client, parallel=True, n_robots=2, min_separation=75)
    graph = view["graph"]
    assert graph["parallel"] and graph["n_robots"] == 2
    assert graph["min_separation"] == agent.parallel_geometry.min_separation == 75
    assert len(graph["nodes"]) == 2 and len(graph["edges"]) == 1
    assert graph["shortest"]["total_time"] == 32
    edge_id = view["next_steps"][0]
    edge = graph["edges"][edge_id]
    assert edge["screw_macro"] and len(edge["stages"]) == 2
    assert set(edge["target_parts"]) == {"A", "B", "C", "D"}
    updated = client.post("/api/step", json={"edge": edge_id})
    assert updated.status_code == 200
    assert updated.json()["parts_present"] == ["Base"]
    assert set(session._removal_order()) == {"A", "B", "C", "D"}
    assert client.post("/api/back").status_code == 200
    assert len(session.parts_present()) == 5


def test_sequential_default_and_exhaustive_parallel_choice(explorer):
    session, agent, client = explorer
    seq = generate(session, client)
    assert not seq["graph"]["parallel"]
    assert len(seq["graph"]["nodes"]) == 16
    full = generate(session, client, parallel=True, collapse_parallel_screws=False)
    assert len(full["graph"]["nodes"]) == 16
    assert not full["graph"]["collapse_parallel_screws"]


def test_rejections_have_sources_and_reasons(explorer, monkeypatch):
    session, agent, client = explorer
    monkeypatch.setattr(
        "disassembly_pomdp.graph_extraction.check_parallel_geometry",
        lambda *args: (False, {"reason": "parts_too_close", "parts": ["A", "B"]}),
    )
    view = generate(session, client, parallel=True)
    rejections = view["graph"]["rejected_attempts"]
    assert rejections and all(
        item["reason"] == "parts_too_close" for item in rejections
    )
    assert any(item["source"] == view["graph"]["root"] for item in rejections)
    assert all(not edge.get("rejected") for edge in view["graph"]["edges"].values())


@pytest.mark.parametrize(
    "body", [{"n_robots": 0}, {"n_robots": 1.5}, {"min_separation": -1}]
)
def test_bad_generation_controls_rejected(explorer, body):
    session, agent, client = explorer
    assert client.post("/api/generate", json=body).status_code == 422
    assert not session.job.running


def test_parallel_analysis_export_and_planner_guard(explorer, tmp_path):
    session, agent, client = explorer
    generate(session, client, parallel=True)
    out = tmp_path / "analysis"
    response = client.post("/api/export-analysis", json={"out_dir": str(out)})
    assert response.status_code == 200, response.text
    for name in (
        "graph.png",
        "graph.json",
        "graph.pkl",
        "shortest_path.png",
        "shortest_path.json",
        "rejected_parallel_graph.png",
        "rejected_parallel_graph.svg",
        "rejected_parallel_graph.json",
    ):
        assert (out / name).stat().st_size > 0, name
    # Planner reversal and single-operation recording must not misrepresent a batch.
    assert client.post("/api/rec", json={"on": True}).status_code == 400
    response = client.post("/api/export", json={"out_dir": str(out)})
    assert response.status_code == 400 and "analysis" in response.json()["detail"]
    session._serial += 1
    assert (
        client.post("/api/export-analysis", json={"out_dir": str(out)}).status_code
        == 409
    )


def test_source_folder_export_rejected(explorer, tmp_path):
    session, agent, client = explorer
    generate(session, client, parallel=True)
    session.folder = tmp_path
    with pytest.raises(ExplorerError, match="source"):
        session.export_analysis(tmp_path / "generated")


def test_no_goal_analysis_reports_error_and_removes_old_path_exports(
    explorer, tmp_path
):
    from product_explorer.graph_analysis import analysis_view, export_analysis

    session, agent, client = explorer
    generate(session, client, parallel=True)
    graph = agent.graph
    for _, attrs in graph.nodes(data=True):
        attrs.pop("is_goal", None)
    g = session._graph
    view = analysis_view(graph, agent, g.node_ids, g.edges)
    assert view["shortest"]["error"] and view["shortest"]["total_time"] is None
    for extension in ("png", "json", "pkl"):
        (tmp_path / f"shortest_path.{extension}").write_text("old result")
    export_analysis(graph, agent, tmp_path)
    assert not (tmp_path / "shortest_path.png").exists()
    assert (tmp_path / "graph.json").is_file()


def test_sequential_analysis_export_is_also_available(explorer, tmp_path):
    session, agent, client = explorer
    generate(session, client)
    result = client.post("/api/export-analysis", json={"out_dir": str(tmp_path)})
    assert result.status_code == 200, result.text
    assert (tmp_path / "shortest_path.png").is_file()
    assert (tmp_path / "graph.png").is_file()
