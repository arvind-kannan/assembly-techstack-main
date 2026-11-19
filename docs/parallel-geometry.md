# Parallel removal geometry screening

`DisassemblyGraphAgent._try_parallel_actions()` now rejects a group before mutating
its assembly unless it passes three checks:

1. Pairwise surface separation using FCL mesh distance. A gap of 5 mm or less
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
        min_separation=5.0,
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
