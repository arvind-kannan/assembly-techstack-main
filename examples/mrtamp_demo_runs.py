# examples/mrtamp_demo_runs.py
"""
Assembly runs of the two demo products (angle grinder, battery replacement product) with B1 and the real cell
simulation's `straight_line_checked` planner (#151), targeted at the simulation.

Reads the sequence graph exports of `examples/export_demo_products.py` and the cell simulation's cell file
(`cases/transformer_cell_simulation.json`: both robots may screw in, each from its own screw nest plate). Each run starts at the product's base parts
alone and ends with all parts present. Writes per product the run record (every operation plan and simulation
report) to `examples/runs/mrtamp_demo_runs/<product>/run_record.json` and prints a summary.

Each product then gets a cell rehearsal: the same run with the real cell's cell file (`cases/transformer_cell.json`),
targeted at the cell but sent to the simulation, whose reports carry the NC programs. Writes its run record and NC
programs to `<product>/cell_rehearsal/` (`round_<n>/MRTAMP_CH<channel>.MPF`). They rest on placeholder slot and
fixture poses: not for the cell.

The cell simulation runs as its own process (its venv, see `crc_disassembly_sim/README.md`), headless:

    cd crc_disassembly_sim
    CRC_SIM_HEADLESS=1 PYTHONPATH=../crc_diss_assembly_dataclasses/src python -c \
        "import DisassemblySim as D; D.WSKAgiloDrive(server_active=False).start_server(port=12345)"

Usage:
    python examples/export_demo_products.py                      # once, about 2 minutes
    python examples/mrtamp_demo_runs.py [--sim HOST:PORT] [--priority shortest|critical_path] [product ...]
"""

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "crc_diss_assembly_dataclasses" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diss_assembly_dataclasses.process import Process  # noqa: E402

from motion_planning.assembly_plan_functions import assembly_plan_functions  # noqa: E402
from mrtamp.b1_planner import make_b1_planner  # noqa: E402
from mrtamp.cell_file import load_cell_file, with_station  # noqa: E402
from mrtamp.exported_graph import (  # noqa: E402
    assembly_from_scratch_start,
    load_export,
    run_from_export,
)
from mrtamp.planning import Goal, RunSettings  # noqa: E402
from mrtamp.screw_groups import part_positions  # noqa: E402
from mrtamp.sim_client import RealCellSimulation  # noqa: E402
from mrtamp.supply import slots_in_cell_frame  # noqa: E402

PRODUCTS = ("angle_grinder", "battery_replacement")
EXPORTS = ROOT / "examples" / "runs" / "demo_products"
CELL_FILE = ROOT / "cases" / "transformer_cell_simulation.json"
REAL_CELL_FILE = ROOT / "cases" / "transformer_cell.json"
NC_MARKER = "; CELL REHEARSAL ONLY (examples/mrtamp_demo_runs.py): placeholder slot and fixture poses, not for the cell\n"
OUT_DIR = ROOT / "examples" / "runs" / "mrtamp_demo_runs"
# Where the product's reference part (its first base part) sits in the cell frame (x, y, z, rx, ry, rz; metres,
# radians): a placeholder in the middle of what both robots reach in the simulation, until the fixture is measured.
FIXTURE_POSE = [-1.2, -1.6, 0.7, 0.0, 0.0, 0.0]
# The TCP orientation of poses given as positions only: the robots' home orientation, tool pointing down.
TOOL_ORIENTATION = [0.0, 0.0, 0.0]


def run_demo_product(
    product: str,
    simulation: RealCellSimulation,
    priority: str = "shortest",
    target: str = "simulation",
):
    """The run record of one demo product's assembly run in the cell simulation.

    target "cell" is the cell rehearsal: the real cell's cell file (Robot 2 alone screws), run against the
    simulation, so that its reports carry the NC programs. Its supply slot poses are all zero, which no robot
    reaches; the rehearsal takes the simulation's supply station instead (nominal placeholder poses, not for the
    cell), so Robot 2 fetches its screws from `screw_nest_plate_h5_robot_2`, a nest plate the real cell lacks.
    """
    file = REAL_CELL_FILE if target == "cell" else CELL_FILE
    export = load_export(
        EXPORTS / product / "sequence_graph.json",
        product_folder=ROOT / "cases" / product / "model",
        cell_file=file,
    )
    cell_file = load_cell_file(file)
    if target == "cell":
        supply = load_cell_file(CELL_FILE).cell.station("supply")
        cell_file = replace(cell_file, cell=with_station(cell_file.cell, supply))
    setup = json.loads(
        (ROOT / "cases" / product / "model" / "product_setup.json").read_text(
            encoding="utf-8"
        )
    )
    reference_part = setup["base_parts"][0]
    simulation.configure_planner(
        "straight_line_checked",
        {
            "reference_poses": {reference_part: FIXTURE_POSE},
            "orientation": TOOL_ORIENTATION,
        },
    )
    plan_functions = assembly_plan_functions(
        slots_in_cell_frame(cell_file.cell), reference_part=reference_part
    )
    all_parts = tuple(sorted(max(export.graph.nodes, key=len)))
    positions = part_positions(export.product_model_xml, export.product_folder)
    return run_from_export(
        export,
        cell_file.cell,
        RunSettings(goal=Goal(present=all_parts), target=target),
        make_b1_planner(plan_functions, priority=priority, part_positions=positions),
        simulation,
        start_state=assembly_from_scratch_start(export.graph, setup["base_parts"]),
        mounted_tools=cell_file.mounted_tools,
        supply_slots=slots_in_cell_frame(cell_file.cell),
        simulation_only=cell_file.simulation_only,
        travel_times=cell_file.travel_times,
    )


def summary(record) -> list[str]:
    lines = [
        f"status {record.status}, {len(record.entries)} round(s), {len(record.operation_plans)} of them planned"
    ]
    for warning in record.warnings:
        lines.append(f"WARNING: {warning}")
    for number, entry in enumerate(record.entries, 1):
        report = entry.simulation_report
        resimulated = (
            f" (re-simulation, segment {entry.segment})" if entry.resimulation else ""
        )
        lines.append(f"round {number}{resimulated}: {entry.decision}")
        if report is None:
            lines.append(
                f"  goal unreachable: {getattr(entry.plan_result, 'reason', '')}"
            )
            continue
        if report.rejection is not None:
            lines.append(
                f"  rejected: {report.rejection.reason_code.value}: {report.rejection.text}"
            )
        if report.infeasible is not None:
            i = report.infeasible
            lines.append(
                f"  infeasible {i.operation_id} on {i.robot}: {i.reason_code.value}: {i.text}"
            )
        m = report.metrics
        lines.append(
            f"  simulated makespan {m.makespan:.1f} s, busy {m.busy_time}, tool changes {m.tool_changes}"
        )
    for plan in record.operation_plans[-1:]:
        screwing = {}
        for op in plan.operations:
            if op.process is Process.SCREWING_IN:
                for robot in op.robots:
                    screwing[robot] = screwing.get(robot, 0) + 1
        lines.append(
            f"last plan: {len(plan.operations)} operations, screwing in per robot {screwing}"
        )
    return lines


class FeasibilityCount:
    """The cell simulation, counting the feasibility queries B1 sends it and those it refuses."""

    def __init__(self, simulation: RealCellSimulation):
        self.simulation, self.queries, self.refused = simulation, 0, 0

    def __getattr__(self, name):
        return getattr(self.simulation, name)

    def __setattr__(self, name, value):
        if name in ("simulation", "queries", "refused"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.simulation, name, value)

    def check_feasibility(self, group, snapshot, cell, *, product_model, travel_times):
        answer = self.simulation.check_feasibility(
            group,
            snapshot,
            cell,
            product_model=product_model,
            travel_times=travel_times,
        )
        self.queries += 1
        self.refused += not answer.feasible
        return answer


def save_nc_programs(record, folder: Path) -> list[str]:
    """Writes the NC programs of each round's simulation report to `folder/round_<n>/<file name>`, marked as
    rehearsal output; removes the rounds of an earlier run first."""
    for old in folder.glob("round_*"):
        shutil.rmtree(old)
    lines = []
    for number, entry in enumerate(record.entries, 1):
        report = entry.simulation_report
        programs = report.nc_programs if report and report.nc_programs else {}
        for program in programs.values():
            out = folder / f"round_{number}" / program["file_name"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(NC_MARKER + program["text"], encoding="utf-8")
            lines.append(f"NC program {out}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("products", nargs="*", choices=PRODUCTS)
    parser.add_argument(
        "--sim", default="127.0.0.1:12345", help="HOST:PORT of the cell simulation"
    )
    parser.add_argument(
        "--priority", choices=("shortest", "critical_path"), default="shortest"
    )
    args = parser.parse_args()
    host, port = args.sim.rsplit(":", 1)
    simulation = RealCellSimulation.at(host, int(port), keep_open=True)
    try:
        for product in args.products or PRODUCTS:
            for target, folder in (
                ("simulation", OUT_DIR / product),
                ("cell", OUT_DIR / product / "cell_rehearsal"),
            ):
                counting = FeasibilityCount(simulation)
                record = run_demo_product(product, counting, args.priority, target)
                record.save_json(folder / "run_record.json")
                print(f"{product}, target {target}: run record in {folder}")
                for line in summary(record) + save_nc_programs(record, folder):
                    print(f"  {line}")
                print(
                    f"  feasibility queries {counting.queries}, {counting.refused} of them refused"
                )
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
