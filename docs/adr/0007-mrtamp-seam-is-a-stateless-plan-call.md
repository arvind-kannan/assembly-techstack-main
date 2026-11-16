# The MRTAMP seam is a stateless `plan` call

MRTAMP planners plug in at one seam, between the run controller and the planner:

```python
plan(planning_request) -> OperationPlan | GoalUnreachable  # with a reason
```

The planning request carries everything the planner needs: the complete sequence graph and its product folder, the cell and its travel-time table (since #174, [ADR-0012](0012-cell-description-holds-no-times.md)), the start cell-assembly snapshot, the run settings, the raw feedback of the run so far (with the earlier operation plans, whose operation ids are unique across the run, so the feedback can be resolved to its steps), and optionally a cell simulation to send feasibility queries to. The planner keeps nothing between requests that the run controller relies on; replanning is a new request from a new snapshot. The run direction (disassembly or assembly) follows from the goal. "Goal unreachable" stops the run for a person, so target parts are never dropped silently. Every operation plan reports its plan metrics (makespan, each robot's busy time, number of tool changes, number of inspections), so planners can be compared on the same product.

This lets theses swap planners without touching the run controller or the cell simulation, and one contract test suite checks every planner at the same seam.

Decided in #91, building on #85, #89 and #92.

## One owner per planning layer

- **MRTAMP planner**: the task layer: robot assignment, order, task-level sync points, inspections.
- **Plan function**: each share's path (tool poses relative to the parts: approach, process motion, retract) and its estimated duration ([ADR-0004](0004-plan-function-is-the-motion-planning-seam.md)). It plans no robot motion.
- **Cell simulation**: all robot motion (along each path, transits between operations, robot-vs-robot checks), motion-level sync points and the NC programs ([ADR-0008](0008-cell-simulation-sits-before-the-real-cell.md)).

## Amendment: product model and step identity (architecture review 2026-10-19)

- The planning request carries the **product model** (its XML and the product folder it is read from) instead of the product folder alone; every snapshot at a product state is built from it. Built from the start snapshot instead, an assembly run planned against snapshots without the parts inserted so far (#232 Assembly runs plan from snapshots without the inserted parts; one product-model type).
- Every operation names the sequence-graph step it carries out, so feedback resolves to its step without inferring it from neighbouring operations (#228 Step catalogue: admissible steps with their cost, and operations that name their step). The planner still keeps nothing between requests.

## Considered options

- **Stateful planner** that is told about failures and patches its plan: replanning would depend on hidden planner state, and the run record could not reproduce it.
- **Planner plans robot motion too** (integrated TAMP at the seam): every thesis planner would need its own motion planner and collision model of the cell. Integrated TAMP stays possible inside a planner via feasibility queries.
- **Fixed sequence from the export**, with the planner only assigning robots: the planner could not choose a cheaper path through the sequence graph.

## Consequences

- Step expansion (one step into the operations that carry it out), the penalty over the feedback, the step catalogue (the admissible steps from a product state for a robot, with their cost) and the operation-plan builder (records the chosen steps and finishes them into an operation plan with inspections and plan metrics, #233 Operation-plan builder shared by the v1 and B1 planners) are shared helpers next to the seam, not part of it: a planner may use or replace them.
- Feasibility queries go to whatever cell simulation the run was wired with, real or the model-level stand-in, so both implement the same interface.
- The baselines (v1 one-robot cheapest path, then B1 greedy two-robot scheduler) port the idea of the legacy planner search graph, not its code; `HierarchicalDisassemblySequencePlanner` stays untouched.
