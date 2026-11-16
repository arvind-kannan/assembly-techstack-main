# The sequence graph is extracted from the POMDP env

Two graphs of disassembly states exist: the legacy planner search graph (cheapest-first search in `HierarchicalDisassemblySequencePlanner`, nodes are full assemblies) and the graph extracted by `graph_extraction.py` (exhaustive breadth-first search over the env's capabilities with EOL sampling and execution noise switched off). The extracted graph is the canonical sequence graph: it uses exactly the capabilities, tools and timings of the POMDP, so graph-based analyses (line balancing, inspection, parallel operations) and the POMDP never disagree about which actions exist. The planner search graph is legacy.

**Amended (AT 269):** the extraction is exhaustive down to the disassembled states by default (product states with every target part removed are not expanded), and exhaustive down to a full teardown on request. The export's provenance records the extraction depth.

## Consequences

- Product states (formerly disassembly states) are identified by the set of parts still present; two states with the same parts but different connections or damage collapse into one. Extensions that need that distinction must change the state key, not add a second graph.
- Assembly uses the same graph by reversal, not a second graph (#31): every removing step has an inverse inserting step, every unscrewing step an inverse screwing-in step; milling steps are one-way. Reversal does not check assembly feasibility (press fits, gravity, gripping at the supply); the cell simulation reports such problems as infeasible operations ([ADR-0008](0008-cell-simulation-sits-before-the-real-cell.md)).
- The extraction depth is an option with two values (AT 269): down to the disassembled states (the default) or a full teardown. A reassembly starts from a disassembled state; an assembly from scratch starts from the base parts alone and needs a full-teardown graph. Precedence extraction and line balancing do not check the depth.
- Reversal happens in the Sequence graph context **before the export** (#89): the exported sequence graph holds the removing steps and their inverse assembly steps, and no consumer (MRTAMP app, Product explorer) reverses the graph itself. Inverse steps' nominal times are an initial guess: reversed path duration plus assembly constants from the cell settings (#90).
