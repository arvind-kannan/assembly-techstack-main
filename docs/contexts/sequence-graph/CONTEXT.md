# Sequence graph

Every product state reachable from a product model, and the actions that lead from one state to the next. Read forwards it describes disassembly, read backwards assembly. Used to inspect, visualise and balance disassembly and assembly processes; the basis for extending them to parallel operations.

## Language

**Sequence graph**:
The directed graph of all product states reachable from a product model with the capabilities of the cell, assuming every feasible action succeeds ([ADR-0005](../../adr/0005-sequence-graph-is-extracted-from-the-env.md)). By default it goes down to its disassembled states and no deeper; on request it goes down to a full teardown, until no step removes a part. The export's provenance records which. It is extracted for disassembly and then completed by reversal: every removing step gets an inverse inserting step, every unscrewing step an inverse screwing-in step, milling steps are one-way. The complete graph, with both kinds of steps, is what consumers receive; none of them reverses it themselves. Corresponds to the CRC *Dis-/Re-/Assembly graph*.
_Avoid_: disassembly graph, assembly graph, MDP graph, search graph, precedence graph

**Product state**:
A node of the sequence graph, identified by the set of parts present and their part conditions (see [Core model](../core-model/CONTEXT.md)).
_Avoid_: disassembly state, assembly state, node state, configuration

**Disassembled state**:
A product state with every target part removed: where a disassembly ends and a reassembly starts. The term is strict: a product whose target parts cannot all come out has no disassembled state, which is a finding about the product or the cell; a dead end is not a disassembled state. A sequence graph extracted without target parts has none marked; its extraction goes on until no step removes a part.
_Avoid_: goal state, empty product state, end state

**Extraction depth**:
How far the extraction of a sequence graph goes: down to the disassembled states (the default; they are not expanded), or a **full teardown**, every state expanded until no step removes a part. Chosen when the graph is extracted, and recorded in the export's provenance. A reassembly starts from a disassembled state; an **assembly from scratch** starts from the base parts alone and needs a full-teardown graph.
_Avoid_: depth (alone, which is a product state's distance from the full product model), level

**Step**:
An edge of the sequence graph: one action, named by its process (see [Decision making](../decision-making/CONTEXT.md)), with its tool, target, direction and nominal execution time. It is an action, not an operation: which robot carries it out, and how, is decided only when it is planned.
_Avoid_: transition, move, operation

**Interchangeable screws**:
Screws that hold the same parts with the same screw head and are removed with the same tool, so they can come out in any order. With them collapsed (the Product explorer always does this), the extraction removes them in one order only (by name) and lists each group in the graph's `collapsed_screw_groups`; 41 lid screws give 42 product states instead of 2^41. A group is a contiguous block of the sequence: the screws may be permuted or split across robots, and the steps before and after the block keep their order. Each planner reads the groups itself: B1 takes the screws of a group in any order, so its product states may lie off the chain (`mrtamp.screw_groups`). The extraction checks the rule and fails loudly, naming the screw or step, if a collapsed screw cannot be removed where another screw of its group can, or if another step can come between the group's screws. Other parallel steps are open.
_Avoid_: symmetric screws, screw pattern

**Disassembly sequence**:
A path through the sequence graph from the full product model to a disassembled state.
_Avoid_: plan, schedule, route

**Assembly sequence**:
A path along inverse steps from a disassembled state to the product state a run's goal names. It follows from the complete graph (reversal), so no one writes it by hand. An assembly from scratch starts from the base parts alone instead, a product state only a full-teardown graph is sure to have.
_Avoid_: assembly plan, build order

**Target part**:
A part whose removal is the goal of the disassembly (defined in [Decision making](../decision-making/CONTEXT.md)).

**Rework**:
Changing or refurbishing the parts of a disassembled product before reassembly, on the cell and its special machines (e.g. rework milling in the mill) or outside it. A reworked part goes back into the product.
_Avoid_: Aufarbeitung, repair, reprocessing

**Remanufacturing sequence**:
One forward flow from an old product to a new one: disassembly into parts, rework, reassembly into the new product assembly. Not yet supported: disassembly and assembly sequences are planned and run separately for now.
_Avoid_: round trip, disassembly-assembly cycle

**Tool change time**:
The time to swap the mounted tool; it depends on the previous step, so it is added along a sequence and never stored on a step. It is looked up in the travel-time table (see [Decision making](../decision-making/CONTEXT.md)).

**Line balancing**:
Assigning the steps of a disassembly or assembly sequence to the stations of a line so their workloads are even. A station is a place in a line, not a robot of a cell (see **Station**); planning the robots of one cell is the MRTAMP planner's job (see [MRTAMP app](../mrtamp-app/CONTEXT.md)).

**Station** (of a line):
A place in a line, with its position in line order, that carries out the steps of one process; a station with no steps stays empty. It is not the cell's own Station (see [Decision making](../decision-making/CONTEXT.md)): that is a named place in a robot cell, this one a place in a line.
_Avoid_: workstation, cell station, line position

**Cycle time**:
In line balancing, the total time of the busiest station: the pace the line can run at; not the length of one operation plan. The *makespan* of an operation plan is a different measure (see [MRTAMP app](../mrtamp-app/CONTEXT.md)).
_Avoid_: makespan, takt time, duration (alone)

**Line efficiency**:
The resource utilisation of a line: the total work of all steps divided by the number of stations offered times the cycle time; 1 when every offered station is busy for the whole cycle.
_Avoid_: throughput, OEE

**SALBP-E**:
The simple assembly line balancing problem of one product on one line, type E: choose the number of stations that minimises stations times cycle time (Scholl and Becker, 2006, European Journal of Operational Research 168(3)). For each number of stations the solver minimises the cycle time, then the stations used, with CP-SAT.
_Avoid_: line balancing (for the problem type), SALBP (without the type)

**Precedence relation**:
For a goal, the pairs of parts (p, q) such that p's step comes before q's step on every path through the sequence graph that reaches the goal. Derived from the sequence graph alone; line balancing and MRTAMP planners may use it to order steps, but it chooses no path.
_Avoid_: constraint graph, precedence graph, dependencies

**Planner search graph** (legacy):
The graph built by the legacy cheapest-first planner, whose nodes are full assemblies and whose edges are planned operations. Superseded by the sequence graph.
_Avoid_: calling it the sequence graph

**Known gap**:
Assembly feasibility is not always the reverse of disassembly feasibility (press fits, gravity, gripping at the supply); the reversal does not check it.
