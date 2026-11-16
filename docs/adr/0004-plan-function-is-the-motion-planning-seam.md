# The plan function is the motion-planning seam

Motion planners plug into the POMDP only through a capability's plan function; capability and env code stay untouched. This lets motion-planning theses replace planners without forking the decision-making code.

## Contract (baseline)

```python
plan_func(assembly: Assembly, operation: Operation) -> Operation | tuple[Operation, Assembly]
```

- **Input**: the current assembly (read-only; copy before mutating) and an operation shell for the decoded action.
- **Output**: the planned operation, or `(operation, result_assembly)`. The operation's `data` holds at least `direction` (the removal direction, or the insertion direction for the assembly plan functions), `path` and `path_duration`.
- **Result assembly**: if returned, it is the assembly after success and must actually reflect the operation (e.g. the removed part is gone); otherwise the transition changes nothing.
- After planning, the env runs the operation's feasibility check; a failing check makes the action fail without changing the assembly.
- Capabilities without a plan function run in symbolic mode.

Plan functions are passed to the capability constructor (`plan_func=`). The removal plan core in `src/motion_planning/removal_planning.py` wraps the sequence generator's removal planners; `src/disassembly_pomdp/env_builder.py` wires its plan functions into the capabilities directly. Whatever plan function a capability has, the core judges its output once: whether the plan failed and why, the unit removal direction (falling back to the part's degrees of freedom) and whether the result assembly really lost the part. The capability and the env only read that judgement; an operation that reaches the env without it (one no plan function produced) is judged there by the same core, without the degrees-of-freedom fallback (#234).

## Amendment: MRTAMP app (#91, #90)

The contract above stays; the MRTAMP planners call plan functions too, after assigning robots ([ADR-0007](0007-mrtamp-seam-is-a-stateless-plan-call.md)).

- Plan functions take the **cell-assembly snapshot** (assembly plus cell state) and the **robot and tool** as context, not the assembly alone. They also take the **product model** of the planning request: an assembly plan function plans on the result assembly, which holds the part the step inserts, and the snapshot before the step does not (#232 Assembly runs plan from snapshots without the inserted parts; one product-model type).
- An operation has one **share** per robot. A plan function delivers each share's path (tool poses relative to the parts: approach, process motion, retract) and its estimated duration. It plans no robot motion: transits and robot-vs-robot checks belong to the cell simulation ([ADR-0008](0008-cell-simulation-sits-before-the-real-cell.md)).
- There is one plan function per process, including the assembly plan functions *inserting*, *screwing in* and *gripping at the supply*. They live in a Motion planning module in this repo, built for now on the removal planners run on the step's result assembly.
- Operation poses are relative to a **reference part** (usually the base part in the fixture), so a measured reference-part pose places them on the real cell.

## Consequences

- The return order stays `(operation, result_assembly)`. #16 Merge `removing-capability` into the techstack was closed as won't do (2026-10-19): one removal plan core decides failure, removal direction and whether the part was really removed, and the POMDP and MRTAMP adapters only translate its result (#234 One removal plan core: failure, removal direction and part-really-removed decided once).
