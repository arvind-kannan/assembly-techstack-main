# examples/mrtamp_assembly_run.py
"""
Assembly run of the bolt-and-nut benchmark case with the v1 planner and the model-level stand-in.

The sequence graph is extracted in the same process and completed by reversal (Sequence graph context),
so the exported graph the run controller reads is complete and it never reverses it. The goal (all parts present) decides
the direction: the run starts at the graph's one disassembled state (the lid, the only target part, out;
the nut locks bolt and nut in place) and inserts the lid from the supply. Writes a run record as JSON.

Usage:
    python examples/mrtamp_assembly_run.py
"""

import json
from pathlib import Path

from diss_assembly_dataclasses.resource import Station
from mrtamp_walking_skeleton import (  # also sets up sys.path
    PRODUCT_FOLDER,
    ROOT,
    export_case,
    extract_bolt_and_nut,
    transformer_cell,
)

from disassembly_pomdp.graph_reversal import complete_by_reversal
from mrtamp.assembly_cell import with_assembly_capabilities
from mrtamp.cell_file import load_cell_file, with_station
from mrtamp.exported_graph import disassembled_state, load_export, run_from_export
from mrtamp.planning import Goal, RunSettings
from mrtamp.stand_in import ModelLevelStandIn
from mrtamp.supply import slots_in_cell_frame
from mrtamp.v1_planner import plan as v1_plan

LID = "part_l_cover_lid"
# The supply station of a cell file (not a whole cell file); its pose and the lid slot's are placeholders until the
# supply is measured on the cell.
SUPPLY_STATION_FILE = Path(__file__).with_name("bolt_and_nut_supply_slots.json")


def bolt_and_nut_supply() -> Station:
    return Station.from_dict(
        json.loads(SUPPLY_STATION_FILE.read_text(encoding="utf-8"))
    )


def assemble_bolt_and_nut(out_dir: Path):
    env, disassembly_graph = extract_bolt_and_nut()
    # the graph's one disassembled state (the lid out); the nut locks the bolt, so it is not the base part alone
    start = disassembled_state(disassembly_graph)
    graph = complete_by_reversal(
        disassembly_graph
    )  # reversal happens before the export
    cell = with_station(
        with_assembly_capabilities(transformer_cell(graph)),
        bolt_and_nut_supply(),
    )
    graph_file, cell_file = export_case(
        out_dir, graph, env, cell, name="bolt_and_nut_assembly"
    )

    export, cell_data = (
        load_export(graph_file, product_folder=PRODUCT_FOLDER, cell_file=cell_file),
        load_cell_file(cell_file),
    )
    full = tuple(sorted(p.name for p in env.base_assembly.parts.values()))
    record = run_from_export(
        export,
        cell_data.cell,
        RunSettings(goal=Goal(present=full)),
        v1_plan,
        ModelLevelStandIn(),
        start_state=start,
        mounted_tools=cell_data.mounted_tools,
        supply_slots=slots_in_cell_frame(cell_data.cell),
        simulation_only=cell_data.simulation_only,
        travel_times=cell_data.travel_times,
    )
    record.save_json(Path(out_dir) / "run_record.json")
    return record, export.graph


def main() -> None:
    out_dir = ROOT / "examples" / "runs" / "mrtamp_assembly_run"
    record, _ = assemble_bolt_and_nut(out_dir)
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
