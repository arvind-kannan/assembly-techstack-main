# Parallel removal geometry screening

`DisassemblyGraphAgent._try_parallel_actions()` now rejects a group before mutating
its assembly unless it passes three checks:

1. Pairwise surface separation using FCL mesh distance. A gap of 50 mm (5 cm) or less
   rejects the group by default. This is a conservative access heuristic, not a
   physical law or a gripper-clearance guarantee.
2. Finite centre rays along each removal direction. Segments passing within
   1 mm reject the group, irrespective of timing. This may reject paths that
   could be scheduled safely; rays do not establish mesh clearance.
3. Sampled FCL checks of the whole moving meshes against each other and the
   stationary assembly. All pulls start together at constant speed, with their
   individual action durations. Completed pulls leave the scene. Moving-pair
   intersections are rejected; stationary contact uses the existing 2.5 mm
   penetration tolerance. Initial stationary contact is expected.

Edges contain directions but do not retain the original removal paths. Screening
therefore constructs straight paths extending by the assembly bounding-box
diagonal plus the part diagonal and one sample step. This ensures clearance of
the initial assembly but can reject groups requiring shorter or different paths.
These are screening paths, not executable robot trajectories. Input meshes are
assumed to be in the shared product coordinate frame, in millimetres.

Configure the heuristics when creating the agent:

```python
from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent
from disassembly_pomdp.parallel_geometry import ParallelGeometrySettings

agent = DisassemblyGraphAgent(
    env,
    parallel_geometry=ParallelGeometrySettings(
        min_separation=50.0,
        ray_clearance=1.0,
        step_size=2.0,
        stationary_penetration=2.5,
        max_samples=10000,
    ),
)
graph = agent.build_graph(parallel=True, n_robots=2)
```

The shared sample spacing bounds relative displacement between two parts to at
most `step_size`; completion times are sampled explicitly too. This is discrete
collision testing, not a continuous collision-free guarantee: thin obstacles
can still be missed. A sample-budget overflow rejects the group rather than
silently coarsening sampling. Missing geometry, directions or valid durations
also reject the group. `agent.last_parallel_check` explains the most recently
screened group's acceptance or rejection. Accepted edges store `geometry_check`
and `feasibility="geometry_screened"`.

The original assembly and its mesh/collision caches are untouched. Single-removal
steps and sequential extraction retain their existing behavior. No robot, tool,
assignment, rotation or alternate-path planning is performed. Screw collapsing
remains unsupported together with parallel extraction.

## Shortest-sequence plot

The extraction CLI also writes `shortest_path.png`, `shortest_path.json` and
`shortest_path.pkl` into the chosen output folder. The separate plot contains
only the fastest synchronized-stage sequence to any reachable goal, selected
with Dijkstra using `action_time`. Each stage lists all its operations and its
duration; the title gives the total. Tool-change times are excluded.

The exact cheapest multiedge is retained at each stage. No reachable goal or
invalid edge durations produce an explanatory message instead of a misleading
plot. The full `graph.*` exports remain available alongside the path exports.
For an already-exported graph, use `plot_shortest_path(load_graph(path), output)`
from `disassembly_pomdp.shortest_path` and `disassembly_pomdp.graph_extraction`.

## Compatible screw set reduction

In parallel mode the generator identifies currently feasible unscrewing actions
using their process and screw connections. It tries full-set choices (one action
per screw) against the geometry screen. If a complete set passes, it groups the
chosen operations into stages of at most `n_robots`, checking each stage against
its updated assembly. Stages are ordered by descending action duration, with
part name as a deterministic tie-breaker. Each stage costs its maximum duration;
the macro-edge costs the sum. Four screws with two robots therefore require two
stages, not four simultaneous robot operations. Individual operation metadata,
stage metadata and the full-set geometry certificate are retained on the edge.

The complete set is still certified before batching. After each internal stage
with screws left to remove, the generator recalculates available actions on the
updated assembly. If a new non-screw removal choice (part, process, capability
or tool) becomes available, the macro stops there. That prefix is one graph edge;
its successor state exposes both the freed-part removal and continuation with
the remaining screws. Metadata (`unlocked_parts`, `batch_stop_reason`) describes
why it stopped, and operations, target parts and time include only executed stages.
The UI lists the newly unlocked parts in the batch tooltip and shortest-path view.

For example, six compatible screws with three robots may stop after the first
three if that frees a cover, leaving three screws in place. With two robots,
checks happen after complete two-operation stages; a cover requiring three
particular screws may therefore be exposed after four screw removals rather
than splitting a simultaneous stage. Non-unscrewing actions on parts recorded
as screws do not count as newly unlocked non-screw removals. A non-screw action
already available at the start does not itself force a split; it remains an
alternative edge. Newly available tool choices for that part do force a split.

Failure of full-set certification or an executed stage keeps the existing
singleton and combination generation. No smaller incompatible subset is silently
collapsed. Sequential extraction and the existing stop-at-goal behavior remain
unchanged.

This reduction still prunes screw permutations, intermediate states inside a
stage, and some mixed screw/part combinations. It preserves newly unlocked part
branches at the chosen stage boundaries, not every possible schedule. Use
`--no-collapse-parallel-screws` (or `collapse_parallel_screws=False` on
`build_graph`) for exhaustive comparisons. It remains separate from the older
sequential `--collapse-screws` option. Intermediate availability rechecks add
work compared with unconditional batching, while avoiding exhaustive screw-order
enumeration when no part becomes newly removable.

## Rejected attempts

Parallel CLI runs also save `rejected_parallel_graph.png`,
`rejected_parallel_graph.svg`, `rejected_parallel_graph.json` and
`rejected_parallel_graph.pkl`. Each failed `_try_parallel_actions()` call has its
own rejected-attempt leaf branching from the evaluated source state. Leaves
include the attempted operations, reason, and available collision diagnostics.
They are not added to the valid sequence graph or its shortest-path search.
Full-set certification failures and failed internal macro stages are included.
Ordinary individually infeasible actions rejected before parallel grouping are
not parallel attempts and are not included.

The PNG is an overview with bounded image dimensions; use the vector SVG and
JSON for large reports. The diagnostic graph is reset on each `build_graph()`;
inspect it directly via `agent.rejected_parallel_graph`. `last_parallel_check`
also records invalid-target and assembly-mutation rejection reasons now.

## Product explorer UI

From the repository root, run:

```bash
uv run python examples/run_product_explorer.py --product cases/angle_grinder/model
```

The browser opens at `http://127.0.0.1:8765/`. For the battery, substitute
`cases/battery_replacement/model`; omit `--product` to choose a folder with Load.
Use `--port 8766` if the default port is occupied, or `--no-browser` to open the
printed address yourself.

The generation controls include Parallel, robot count, minimum surface gap in
millimetres (default 50), compatible screw collapsing, and extraction depth.
The default is still sequential generation. Select Parallel and Generate graph
for the new analysis. The graph selector switches between all transitions,
the minimum-time path with screw sub-stages, and rejected parallel attempts
with reasons. Hover a valid transition for its operation names and timing.
Grouped next steps remove all affected parts together; their 3D state update
is atomic rather than a robot motion animation.

Export → Analysis graphs and plots saves `graph.*`, `shortest_path.*` and
`rejected_parallel_graph.*` to the chosen output directory. Stale graphs must
be rebuilt before export. Analysis export does not reverse parallel batches
into robot-planner instructions. The existing sequence-graph/URDF and recorded
sequence exports remain available for sequential graphs; recording and planner
export are disabled for parallel analysis until those consumers support batches.

## Single removal alternatives

Each source/destination pair now has one single-removal edge: the feasible
alternative with the lowest finite nonnegative `action_time`. This prevents
different tools or capabilities from drawing duplicate lines for the same part
removal. Equal-time choices use the original deterministic enumeration order.
The edge retains the original choices under `alternative_actions` for auditing.

Deduplication happens after parallel candidates are generated. Geometry screening
therefore still considers all tool and direction choices, and distinct grouped
parallel transitions are retained. Existing exported graphs must be regenerated
to show the reduced single-removal edges.
