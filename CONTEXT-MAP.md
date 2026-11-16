# Context Map

The assembly techstack plans and simulates the robotic disassembly of end-of-life products under uncertainty, and names the assembly operations the cell runs on its demonstrator products. It is split into six bounded contexts, each with its own glossary. Terms shared across contexts are defined once, in the context that owns them, and referenced elsewhere.

## Contexts

- [Core model](./docs/contexts/core-model/CONTEXT.md): the product as parts, connections and free directions; grows into the PPR model, whose process and resource types are named in the contexts that use them ([ADR-0006](./docs/adr/0006-core-grows-into-the-ppr-model.md)). Lives in `crc_diss_assembly_dataclasses/` (subtree, protected, see [ADR-0003](./docs/adr/0003-core-model-change-process.md)).
- [Sequence graph](./docs/contexts/sequence-graph/CONTEXT.md): all product states reachable from a product and the actions between them. Lives in `src/disassembly_pomdp/graph_extraction.py`; its consumers are the precedence relation (`graph_precedence.py`) and line balancing (`line_balancing.py`, `line_balance_report.py`, all in `src/disassembly_pomdp/`) and the scripts in `examples/`.
- [Motion planning](./docs/contexts/motion-planning/CONTEXT.md): turning an action into a geometric, executable operation and checking that it is feasible. Lives in `crc_disassembly_sequence_generator/` (`operations.py`, `plan_*` in `planning.py`), in the plan functions wired in `examples/build_env.py` and in `src/motion_planning/` (the assembly plan functions: inserting, screwing in, gripping at the supply).
- [Decision making](./docs/contexts/decision-making/CONTEXT.md): the POMDP: capabilities, actions, belief over hidden EOL conditions, observations, reward; also the assembly capabilities and the skills and NC programs that execute capabilities on the cell. Lives in `src/disassembly_pomdp/`.
- [Product explorer](./docs/contexts/product-explorer/CONTEXT.md): the app in which a product author builds and checks a product model and a researcher inspects one. Replaces the retired DASG app ([ADR-0011](./docs/adr/0011-dasg-app-retired-product-explorer-replaces-it.md)). Lives in `src/product_explorer/` (outside the subtrees); v1 is specified in #222.
- [MRTAMP app](./docs/contexts/mrtamp-app/CONTEXT.md): the app in which a cell operator or researcher plans a run from a sequence graph with an MRTAMP planner, executes it on the real cell or a cell simulation, and replans after a failed operation. Code location is decided in its spec (outside the subtrees). The cell simulation itself lives in `crc_disassembly_sim/` (subtree, [ADR-0002](./docs/adr/0002-sub-repos-as-git-subtrees.md)).

## Relationships

- **Core model ← all**: every other context reads and copies `Assembly` objects. Only the core model defines what a part, connection or free direction is.
- **Decision making → Motion planning**: each capability hands an operation to its plan function and gets back a planned operation, optionally with the result assembly ([ADR-0004](./docs/adr/0004-plan-function-is-the-motion-planning-seam.md)).
- **Sequence graph → Decision making**: the sequence graph is extracted by running the POMDP's own capabilities with all uncertainty switched off, so graph and POMDP always agree on which actions exist ([ADR-0005](./docs/adr/0005-sequence-graph-is-extracted-from-the-env.md)).
- **Line balancing → Sequence graph**: reads the complete exported graph and uses the disassembly direction only (inverse steps are dropped first); it assigns steps to the stations of a line and changes nothing in the graph.
- **Product explorer → Core model**: edits, loads and saves assemblies only through core calls ([ADR-0003](./docs/adr/0003-core-model-change-process.md)).
- **Product explorer → Sequence graph**: extracts the sequence graph, shows it, and marks it stale after an edit to the product model.
- **Product explorer → MRTAMP app**: reads the cell file with the MRTAMP app's cell file loader (`mrtamp.cell_file`) to choose the tools of the available actions, and the MRTAMP app's rule for a part's position (`mrtamp.screw_groups`, a mesh with a finite vertex mean) to flag parts without a usable mesh.
- **Product explorer → Decision making**: finds the available actions with the POMDP's own capabilities; later also reads belief for a POMDP view.
- **Decision making ↔ Core model**: robot holds (which robot holds which part) currently live on `Assembly` as a provisional concept; they move to the cell state type in the core, and the hold API on `Assembly` is deprecated, then removed (#15, decided in #95, [ADR-0006](./docs/adr/0006-core-grows-into-the-ppr-model.md)).
- **MRTAMP app → Sequence graph**: reads the complete exported sequence graph (removing and inverse assembly steps) and the product folder its provenance names, and chooses the path through it itself.
- **MRTAMP app → Motion planning**: its MRTAMP planner fills in operations through plan functions ([ADR-0004](./docs/adr/0004-plan-function-is-the-motion-planning-seam.md)); MRTAMP planners themselves plug in through the stateless MRTAMP seam ([ADR-0007](./docs/adr/0007-mrtamp-seam-is-a-stateless-plan-call.md)).
- **MRTAMP app → cell simulation → real cell**: every operation plan reaches the real cell only through the cell simulation, which plans all robot motion and generates the NC programs ([ADR-0008](./docs/adr/0008-cell-simulation-sits-before-the-real-cell.md)).
- **MRTAMP app → Decision making**: uses the cell, its cell state, skills and NC programs; the POMDP policy stays out.

## CRC glossary alignment

The CRC 1574 glossary ([circular_factory/glossary](https://gitlab.kit.edu/kit/ifl/opensource/circular_factory/glossary)) is the shared vocabulary of the SFB. Where its terms differ from ours:

- CRC *component* ≈ our *part*; CRC *object* is not used. Both stay under _Avoid_ here. CRC *virgin component* = a virgin part.
- CRC *assembly* = our *assembly*; CRC *product instance* = ours.
- CRC *capability* and *skill* = ours; CRC *movement primitive* = ours.
- CRC *operation* is broader (a task on resources); ours is narrowed to one action.
- CRC *task* ≈ our *action*; CRC *process* = our *process* (the technology, e.g. unscrewing), CRC *resource* and *capability* = ours (a resource's ability to carry out one process).
- CRC *Dis-/Re-/Assembly graph* ≈ our *sequence graph*; its *product states* = ours.
- CRC *action*, *state*, *reward*, *policy* = the usual RL meanings, as ours.

## Legacy (not a context)

Kept for reference and for the real robot cell. Do not extend; ask a core maintainer before changing.

- **DASG app** in `crc_disassembly_sequence_generator/`: retired ([ADR-0011](./docs/adr/0011-dasg-app-retired-product-explorer-replaces-it.md)), replaced by the Product explorer. Its GUI (`dasg_hmi*.py`), Q-learning, demos and data copies leave in a first removal (#170), perception and the Sinumerik OPC UA / G-code link in a second (#171). Its perception (`perceptionModule.py`, `Registration.py`, `Zivid.py`) is replaced for the part-pose pipeline (#45) by `src/perception/`, Hao's perception core ported outside the subtrees (#249): Zivid capture in metres, registration with the #45 gate, System 28 desk and clamp localisation. `operations.py`, the `plan_*` functions, grasp planning and the utilities stay in the subtree.
- **Planner search graph**: the cheapest-first graph search of `HierarchicalDisassemblySequencePlanner` in `crc_disassembly_sequence_generator/planning.py`. Its static `plan_*` functions are still used by motion planning.

## Upstreams

| Folder | Upstream (where changes flow back) | Imported from |
|---|---|---|
| repo root | `github.com/WBK-Robotics/pomdp_disassembly_gym` | `pomdp_observer@dabba42` (squashed, [ADR-0001](./docs/adr/0001-baseline-is-gym-pomdp-observer.md)) |
| `crc_diss_assembly_dataclasses/` | `github.com/WBK-Robotics/crc_diss_assembly_dataclasses` | `hawo2003` fork `@6c87e5f` (subtree, [ADR-0002](./docs/adr/0002-sub-repos-as-git-subtrees.md)) |
| `crc_disassembly_sequence_generator/` | `github.com/WBK-Robotics/crc_disassembly_sequence_generator` | `hawo2003` fork `@ced2ed9` (subtree) |
| `crc_disassembly_sim/` | `gitlab.kit.edu/kit/wbk/map/imk/sfb1574_kreislauffabrik/crc-disassembly-sim` | `main@d81ed0f` (subtree with LFS, re-added in #148) |
