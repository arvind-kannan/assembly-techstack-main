"""Line-balance report: the CLI on a tiny exported sequence graph, the Gantt plot and time scaling."""

import json

import networkx as nx
import pytest

pytest.importorskip(
    "ortools",
    reason="line balancing needs ortools (not in uv.lock; uv add --dev ortools)",
)

from disassembly_pomdp.graph_export import export_graph
from disassembly_pomdp.graph_precedence import Task
from disassembly_pomdp.line_balance_report import main, plot_line_balance, scale_times

PROVENANCE = {"product_name": "toy", "fingerprints": {"product_model_xml": "abc123"}}


def write_chain_export(path):
    """screw holds lid on base: unscrew (5 s), then grip off the lid (8 s)."""
    g = nx.MultiDiGraph()
    full, no_screw, base = ("base", "lid", "screw"), ("base", "lid"), ("base",)
    for depth, node in enumerate((full, no_screw, base)):
        g.add_node(
            node, parts_present=list(node), n_parts_present=len(node), depth=depth
        )
    g.add_edge(
        full,
        no_screw,
        process="unscrewing",
        tool="T6",
        target_part="screw",
        action_time=5.0,
    )
    g.add_edge(
        no_screw,
        base,
        process="gripping and removing",
        tool="H5",
        target_part="lid",
        action_time=8.0,
    )
    export_graph(g, path, PROVENANCE)
    return path


def test_cli_prints_the_summary_and_writes_plan_and_gantt(tmp_path, capsys):
    export = write_chain_export(tmp_path / "graph.json")
    out = tmp_path / "out"

    main(["--graph", str(export), "--stations", "auto", "--out-dir", str(out)])

    printed = capsys.readouterr().out
    assert "Cycle time" in printed and "8.00 s" in printed
    plan = json.loads((out / "line_balance_plan.json").read_text(encoding="utf-8"))
    assert plan["provenance"] == PROVENANCE
    assert plan["cycle_time_s"] == 8.0
    assert plan["throughput_per_h"] == pytest.approx(450.0)  # 3600 s / 8 s
    assert [s["process"] for s in plan["stations"]] == [
        "unscrewing",
        "gripping and removing",
    ]
    assert plan["stations"][0]["steps"] == [
        {"target_part": "screw", "tool": "T6", "action_time_s": 5.0}
    ]
    assert (out / "line_balance_gantt.png").stat().st_size > 0


def test_cli_refuses_a_graph_that_is_not_an_export(tmp_path, capsys):
    pickle_file = tmp_path / "graph.pkl"
    pickle_file.write_bytes(b"\x80\x04not json")

    with pytest.raises(SystemExit):
        main(["--graph", str(pickle_file), "--out-dir", str(tmp_path)])

    assert "sequence graph export" in capsys.readouterr().err


@pytest.mark.parametrize("stations", ["0", "1"])  # the chain has two processes
def test_cli_reports_an_impossible_station_count_without_a_traceback(
    tmp_path, capsys, stations
):
    export = write_chain_export(tmp_path / "graph.json")

    with pytest.raises(SystemExit):
        main(
            ["--graph", str(export), "--stations", stations, "--out-dir", str(tmp_path)]
        )

    assert "station" in capsys.readouterr().err


def test_cli_refuses_a_json_file_that_is_not_an_export(tmp_path, capsys):
    other = tmp_path / "other.json"
    other.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["--graph", str(other), "--out-dir", str(tmp_path)])

    assert "sequence graph export" in capsys.readouterr().err


def report_with_processes(n_processes):
    """A plan report with one station per process, one 2-second step each."""
    return {
        "n_stations": n_processes,
        "n_stations_used": n_processes,
        "cycle_time_s": 2.0,
        "throughput_per_h": 1800.0,
        "utilization": 1.0,
        "stations": [
            {
                "station_id": i + 1,
                "process": f"process {i + 1}",
                "total_time_s": 2.0,
                "idle_time_s": 0.0,
                "utilization": 1.0,
                "steps": [
                    {"target_part": f"part {i + 1}", "tool": None, "action_time_s": 2.0}
                ],
            }
            for i in range(n_processes)
        ],
    }


def test_plot_saves_a_non_empty_file_with_the_agg_backend(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    plot_line_balance(report_with_processes(2), tmp_path / "gantt.png")

    assert (tmp_path / "gantt.png").stat().st_size > 0


def test_plot_gives_nine_processes_nine_distinct_styles():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    fig = plot_line_balance(report_with_processes(9))

    def style(patch):
        return patch.get_facecolor(), patch.get_hatch() or ""

    legend = fig.legends[0]
    assert [t.get_text() for t in legend.get_texts()] == [
        f"process {i}" for i in range(1, 10)
    ]
    assert len({style(h) for h in legend.legend_handles}) == 9
    assert (
        len({style(bar) for bar in fig.axes[0].patches}) == 9
    )  # one bar per station here


def test_scale_times_multiplies_every_action_time():
    tasks = {"screw": Task("screw", "unscrewing", "T6", 5.0, None)}

    assert scale_times(tasks, 1.5)["screw"].action_time == 7.5
    assert tasks["screw"].action_time == 5.0  # input untouched
    with pytest.raises(ValueError):
        scale_times(tasks, 0)
