"""Union product model (#284, decided in #268): a product that gains a part the old model lacks is one
product folder with the old parts plus the new one, extracted once with the new part among the target parts.
The old product is then a product state of the union graph, and one run goes from it to the new product.

The fixture is the bolt-and-nut product plus a cap on the pipe's free end that comes out first."""

import os
import sys
from pathlib import Path

import pytest
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Station

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from mrtamp_walking_skeleton import (  # noqa: E402  (also sets up sys.path)
    BASE_PART,
    GOAL,
    XML,
    export_case,
    make_env_from_imported_xml_auto,
    make_external_case_observer,
    transformer_cell,
)

from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent  # noqa: E402
from disassembly_pomdp.graph_reversal import complete_by_reversal  # noqa: E402
from mrtamp.assembly_cell import with_assembly_capabilities  # noqa: E402
from mrtamp.cell_file import load_cell_file, with_station  # noqa: E402
from mrtamp.exported_graph import load_export, run_from_export  # noqa: E402
from mrtamp.planning import Goal, RunSettings  # noqa: E402
from mrtamp.stand_in import ModelLevelStandIn  # noqa: E402
from mrtamp.supply import slots_in_cell_frame  # noqa: E402
from mrtamp.v1_planner import plan as v1_plan  # noqa: E402

UNION_FOLDER = ROOT / "tests" / "test_products" / "bolt_and_nut_with_cap"
UNION_XML = UNION_FOLDER / "xml" / XML.name
CAP = "part_m_cap"

pytestmark = [pytest.mark.integration, pytest.mark.geom]


def _extract(xml: Path, target_parts):
    """The env and the disassembly-only sequence graph of a bolt-and-nut product, built the same way for the
    old and the union product: only the XML and the target parts differ."""
    cwd = Path.cwd()
    os.chdir(xml.parent)  # mesh and contact paths in the XML are relative to its folder
    try:
        env = make_env_from_imported_xml_auto(
            xml,
            observer=make_external_case_observer(debug=False),
            base_parts=(BASE_PART,),
            target_parts=target_parts,
        )
        return env, DisassemblyGraphAgent(env).build_graph(verbose=False)
    finally:
        os.chdir(cwd)


def _full(graph):
    """The full product model: the graph's largest product state."""
    return max(graph.nodes, key=len)


def _without_cap(state):
    return tuple(sorted(set(state) - {CAP}))


def _steps(graph, among=None):
    """The steps as (from, to, process, target part, tool, direction): what must agree between the graphs.
    Node attributes (`is_goal`, `depth`, `parts_removed`) differ legitimately: the targets differ."""
    return sorted(
        (u, v, s["process"], s["target_part"], s["tool"], tuple(s["direction"]))
        for u, v, s in graph.edges(data=True)
        if among is None or (u in among and v in among)
    )


@pytest.fixture(scope="module")
def old():
    return _extract(XML, GOAL)


@pytest.fixture(scope="module")
def union():
    return _extract(UNION_XML, (*GOAL, CAP))


def test_the_old_product_is_the_union_graphs_state_without_the_extra_part(old, union):
    _, old_graph = old
    _, union_graph = union
    old_root = _full(old_graph)

    assert old_root in union_graph.nodes
    assert CAP not in old_root
    assert old_root == _without_cap(_full(union_graph))


def test_the_union_graph_without_the_extra_part_is_the_old_graph(old, union):
    _, old_graph = old
    _, union_graph = union
    without_cap = {n for n in union_graph.nodes if CAP not in n}

    assert without_cap == set(old_graph.nodes)
    assert _steps(union_graph, among=without_cap) == _steps(old_graph)


def test_the_extra_part_comes_out_of_the_full_union_product_in_one_step(union):
    _, union_graph = union
    full = _full(union_graph)

    steps = [s for _, v, s in union_graph.out_edges(full, data=True) if CAP not in v]

    assert [(s["process"], s["target_part"]) for s in steps] == [
        (Process.GRIPPING_AND_REMOVING.value, CAP)
    ]


def test_a_run_from_the_old_product_to_old_targets_and_extra_part_present_finds_a_plan(
    old, union, tmp_path
):
    env, sequence_graph = union
    old_root = _full(old[1])  # the old product, as the old graph has it
    assert old_root in sequence_graph.nodes
    graph = complete_by_reversal(sequence_graph)
    supply = Station.from_dict(
        {
            "name": "supply",
            "kind": "supply",
            "pose": [0.0, 0.5, 0.1, 0.0, 0.0, 0.0],
            "processes": [],
            "coupling_places": [],
            "slots": [
                {
                    "name": "cap_slot",
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "part": CAP,
                    "screw_type": None,
                    "robots": [],
                }
            ],
        }
    )
    cell = with_station(with_assembly_capabilities(transformer_cell(graph)), supply)
    graph_file, cell_file = export_case(
        tmp_path,
        graph,
        env,
        cell,
        name="bolt_and_nut_with_cap",
        product_folder=UNION_FOLDER,
        xml=UNION_XML,
    )
    export = load_export(graph_file, product_folder=UNION_FOLDER, cell_file=cell_file)
    cell_data = load_cell_file(cell_file)

    record = run_from_export(
        export,
        cell_data.cell,
        RunSettings(goal=Goal(present=(*GOAL, CAP))),
        v1_plan,
        ModelLevelStandIn(),
        start_state=old_root,
        mounted_tools=cell_data.mounted_tools,
        supply_slots=slots_in_cell_frame(cell_data.cell),
        simulation_only=cell_data.simulation_only,
        travel_times=cell_data.travel_times,
    )

    assert record.status == "goal_reached"
    plan = record.operation_plans[-1]
    assert (Process.INSERTING, CAP) in [
        (op.process, op.target) for op in plan.operations
    ]
    assert CAP in record.final_snapshot.product_state_key
    assert not export.warnings
