"""The run controller's input: an exported sequence graph and the product folder named in its provenance.

`load_export` refuses an unknown `schema_version`, compares the fingerprints in the provenance header with
the files now in the product folder and collects a warning for each mismatch (a stale export warns, it
does not block). The assembly of any product state is the base product model minus the removed parts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState
from diss_assembly_dataclasses.resource import Cell, SupplySlot

from disassembly_pomdp.graph_export import fingerprint, fingerprint_of_cell, read_export
from disassembly_pomdp.product_model import ProductModel
from disassembly_pomdp.travel_time_table import TravelTimeTable
from mrtamp.planning import Planner, RunSettings
from mrtamp.run_controller import RunRecord, run


@dataclass(frozen=True)
class GraphExport:
    """The graph, its provenance header, the product folder to use and the fingerprint warnings."""

    graph: nx.MultiDiGraph
    provenance: dict
    product_folder: str
    warnings: tuple[str, ...] = ()

    @property
    def product_model(self) -> ProductModel:
        """The base product model: the XML its provenance names, in the product folder in use."""
        return ProductModel.find(
            self.product_folder, provenance_xml=self.provenance["product_model_xml"]
        )

    @property
    def product_model_xml(self) -> str:
        """The base product model: the XML text of the whole product."""
        return self.product_model.xml

    def snapshot_for(
        self,
        product_state: tuple[str, ...],
        mounted_tools: dict | None = None,
        holds: dict | None = None,
    ) -> Cell_Assembly_Snapshot:
        """The snapshot at `product_state`: the base product model minus the removed parts, with this cell state."""
        return self.product_model.snapshot_at(
            product_state,
            CellState(holds=dict(holds or {}), mounted_tools=dict(mounted_tools or {})),
        )


def load_export(
    path: str | Path,
    product_folder: str | Path | None = None,
    cell_file: str | Path | None = None,
) -> GraphExport:
    """Read an export. `product_folder` and `cell_file` override the paths in the provenance (moved files)."""
    graph, provenance = read_export(path)
    folder = Path(
        product_folder if product_folder is not None else provenance["product_folder"]
    )
    cell_path = cell_file if cell_file is not None else provenance.get("cell_file")
    model = ProductModel.find(folder, provenance_xml=provenance["product_model_xml"])
    checks = (
        (
            "product model",
            provenance["product_model_xml"],
            model.xml_path,
            "product_model_xml",
            fingerprint,
        ),
        (
            "product setup",
            provenance.get("product_setup"),
            None
            if provenance.get("product_setup") is None
            else folder / provenance["product_setup"],
            "product_setup",
            fingerprint,
        ),
        (
            "cell file",
            cell_path,
            None if cell_path is None else Path(cell_path),
            "cell_file",
            fingerprint_of_cell,  # the cell file and its travel-time table
        ),
    )
    warnings = []
    for label, name, file, key, fingerprint_of in checks:
        recorded = provenance["fingerprints"].get(key)
        if recorded is None or file is None:
            continue
        current = fingerprint_of(file)
        if current is None:
            warnings.append(
                f"{label} {name} not found at {file}; the export belongs to a product folder that is not here"
            )
        elif current != recorded:
            warnings.append(
                f"{label} {name} changed since the export (fingerprint mismatch); the sequence graph may be stale"
            )
    return GraphExport(graph, provenance, str(folder), tuple(warnings))


def assembly_from_scratch_start(
    graph: nx.MultiDiGraph, base_parts: Iterable[str]
) -> tuple[str, ...]:
    """The start of an assembly from scratch: the base parts alone (sorted, as the graph names product
    states). Only a graph extracted for a full teardown is sure to have it; it must be a node of the graph."""
    state = tuple(sorted(base_parts))
    if state not in graph:
        raise ValueError(
            f"the base parts {list(state)} are not a product state of the sequence graph; "
            "an assembly from scratch needs a graph extracted for a full teardown"
        )
    return state


def disassembled_state(graph: nx.MultiDiGraph) -> tuple[str, ...]:
    """The start of a reassembly: the graph's one disassembled state (every target part removed, the
    `is_goal` mark). Raises if the graph has none or several."""
    states = [n for n, d in graph.nodes(data=True) if d.get("is_goal")]
    if len(states) != 1:
        raise ValueError(
            f"the sequence graph has {len(states)} disassembled states ({sorted(states)}); "
            "a reassembly needs exactly one to start from"
        )
    return states[0]


def run_from_export(
    export: GraphExport,
    cell: Cell,
    settings: RunSettings,
    planner: Planner,
    cell_simulation,
    *,
    start_state: tuple[str, ...] | None = None,
    mounted_tools: dict | None = None,
    supply_slots: tuple[SupplySlot, ...] = (),
    max_rounds: int = 20,
    simulation_only: bool = False,
    travel_times: TravelTimeTable,
) -> RunRecord:
    """Run from an exported graph. The run starts at `start_state` (default: the whole product model).
    `simulation_only` and `travel_times` come from the cell file (see `mrtamp.cell_file.CellFile`)."""
    if start_state is None:
        start_state = tuple(sorted(n for n in max(export.graph.nodes, key=len)))
    return run(
        sequence_graph=export.graph,
        product_model=export.product_model,
        cell=cell,
        start_snapshot=export.snapshot_for(start_state, mounted_tools),
        settings=settings,
        planner=planner,
        cell_simulation=cell_simulation,
        supply_slots=supply_slots,
        max_rounds=max_rounds,
        provenance=export.provenance,
        warnings=export.warnings,
        simulation_only=simulation_only,
        travel_times=travel_times,
    )
