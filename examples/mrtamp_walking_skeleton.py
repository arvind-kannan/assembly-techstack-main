# examples/mrtamp_walking_skeleton.py
"""
Minimum architecture example of the MRTAMP app: a disassembly run of the bolt-and-nut benchmark case.

The sequence graph is extracted and exported to a file (with its provenance), then the run controller
reads that file with the cell file and the product folder, asks the v1 baseline for an operation plan,
sends it through the model-level stand-in, reads the inspections and writes a run record as JSON.

Usage:
    python examples/mrtamp_walking_skeleton.py                 # extract, export, run
    python examples/mrtamp_walking_skeleton.py GRAPH CELL_FILE # run from an exported graph and its cell file
    python examples/mrtamp_walking_skeleton.py --sim HOST:PORT --planner NAME   # plan motion in the real cell simulation
"""

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (
    ROOT,
    ROOT / "src",
    ROOT / "crc_diss_assembly_dataclasses" / "src",
    ROOT / "examples",
):
    sys.path.insert(0, str(p))

from build_env_test_cases import (
    make_env_from_imported_xml_auto,
    make_external_case_observer,
)
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell, Robot, Station, StationKind, Tool

from disassembly_pomdp.graph_export import export_graph, make_provenance, read_export
from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent
from disassembly_pomdp.travel_time_table import TravelTimeTable
from mrtamp.cell_file import CellFile, load_cell_file, save_cell_file
from mrtamp.exported_graph import load_export, run_from_export
from mrtamp.plan_functions import removal_plan_functions
from mrtamp.planning import Goal, RunSettings
from mrtamp.stand_in import ModelLevelStandIn
from mrtamp.v1_planner import make_v1_planner

PRODUCT_FOLDER = ROOT / "benchmark_cases" / "benchmark_bolt_and_nut_case"
XML = PRODUCT_FOLDER / "xml" / "rotational_bolt_nut_lock_benchmark.xml"
BASE_PART = "part_j_elbow_pipe"
# The nut is a rotational lock that gripping and removing cannot release, so the lid is the only target part
# and the goal of the disassembly run.
GOAL = ("part_l_cover_lid",)
INSPECTION_TIME = 10.0  # placeholder until measured on the cell
STATION_KINDS = (
    StationKind.TOOL_STORAGE,
    StationKind.SUPPLY,
    StationKind.DROPOFF,
    StationKind.FIXTURE,
)
PLACEHOLDER_POSE = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)  # station poses until they are measured on the cell


def transformer_cell(graph) -> Cell:
    """Robot 1 carries out every step of the graph; Robot 2 inspects with the Zivid camera."""
    steps = [step for _, _, step in graph.edges(data=True)]
    tools = {step["tool"]: Process(step["process"]) for step in steps}
    return Cell(
        robots=(
            Robot(
                "robot_1",
                1,
                tools=tuple(tools),
                capabilities=tuple(set(tools.values())),
            ),
            Robot("robot_2", 2, tools=(), capabilities=(Process.INSPECTING,)),
        ),
        tools=tuple(Tool(name, process) for name, process in tools.items()),
        stations=tuple(
            Station(kind.value, kind, PLACEHOLDER_POSE) for kind in STATION_KINDS
        ),
    )


def transformer_travel_times(cell_settings) -> TravelTimeTable:
    """The env's cell settings plus the inspection time, as the cell's travel-time table (ADR-0012, interim)."""
    return TravelTimeTable(
        {**dataclasses.asdict(cell_settings), "inspection_time": INSPECTION_TIME}
    )


def extract_bolt_and_nut():
    """The env and the extracted (disassembly-only) sequence graph of the bolt-and-nut case, down to its
    disassembled state."""
    env = make_env_from_imported_xml_auto(
        XML,
        observer=make_external_case_observer(debug=False),
        base_parts=(BASE_PART,),
        target_parts=GOAL,
    )
    graph = DisassemblyGraphAgent(env).build_graph(verbose=False)
    return env, graph


def export_case(
    out_dir: Path,
    graph,
    env,
    cell,
    name="bolt_and_nut",
    product_folder: Path = PRODUCT_FOLDER,
    xml: Path = XML,
) -> tuple[Path, Path]:
    """Write the cell file and the sequence graph export (the complete graph) of a case; return both paths.
    `product_folder` and `xml` name the case's product model (default: the bolt-and-nut benchmark)."""
    cell_path = Path(out_dir) / "cell.json"
    save_cell_file(
        CellFile(
            cell,
            {"robot_1": env.initial_mounted_tool, "robot_2": None},
            travel_times=transformer_travel_times(env.cell_settings),
        ),
        cell_path,
    )
    provenance = make_provenance(
        name,
        product_folder,
        str(xml.relative_to(product_folder)),
        cell_file=cell_path,
        export_folder=out_dir,
        extraction_depth=graph.graph.get("extraction_depth"),
    )
    graph_path = Path(out_dir) / "sequence_graph.json"
    export_graph(graph, graph_path, provenance)
    return graph_path, cell_path


def run_bolt_and_nut(
    out_dir: Path,
    graph_file: Path | None = None,
    cell_file: Path | None = None,
    cell_simulation=None,
):
    """Run the disassembly from an exported graph; without one, extract and export it first.
    `cell_simulation` replaces the model-level stand-in (e.g. a `RealCellSimulation`)."""
    if graph_file is None:
        env, graph = extract_bolt_and_nut()
        graph_file, cell_file = export_case(
            out_dir, graph, env, transformer_cell(graph)
        )
    if (
        cell_file is None
    ):  # the cell file named in the provenance, relative to the export folder
        cell_file = Path(graph_file).parent / read_export(graph_file)[1]["cell_file"]
    export = load_export(graph_file, product_folder=PRODUCT_FOLDER, cell_file=cell_file)
    cell = load_cell_file(cell_file)
    record = run_from_export(
        export,
        cell.cell,
        RunSettings(goal=Goal(absent=GOAL)),
        make_v1_planner(removal_plan_functions()),
        cell_simulation or ModelLevelStandIn(),
        mounted_tools=cell.mounted_tools,
        simulation_only=cell.simulation_only,
        travel_times=cell.travel_times,
    )
    record.save_json(Path(out_dir) / "run_record.json")
    return record, export.graph


def main() -> None:
    out_dir = ROOT / "examples" / "runs" / "mrtamp_walking_skeleton"
    args = sys.argv[1:]
    simulation = None
    if "--sim" in args:
        from mrtamp.sim_client import RealCellSimulation

        host, port = args.pop(args.index("--sim") + 1).rsplit(":", 1)
        args.remove("--sim")
        simulation = RealCellSimulation.at(host, int(port))
        if "--planner" in args:
            simulation.configure_planner(args.pop(args.index("--planner") + 1))
            args.remove("--planner")
    record, _ = run_bolt_and_nut(
        out_dir, *[Path(a) for a in args], cell_simulation=simulation
    )
    for warning in record.warnings:
        print(f"WARNING: {warning}")
    print(f"Run {record.status}; run record in {out_dir / 'run_record.json'}")
    for plan in record.operation_plans:
        for op in plan.operations:
            print(
                f"  {op.id:<14} {op.process.value:<22} {op.target or '':<20} on {', '.join(op.robots)}"
            )
        m = plan.metrics
        print(
            f"  makespan {m.makespan:.1f} s, busy {m.busy_time}, tool changes {m.tool_changes}, "
            f"inspections {m.inspections}"
        )


if __name__ == "__main__":
    main()
