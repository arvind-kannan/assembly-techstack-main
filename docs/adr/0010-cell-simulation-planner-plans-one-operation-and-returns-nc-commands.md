# The cell simulation's motion planner plans one operation and returns NC commands

A motion planner plugs into the cell simulation (`register_planner` / `configure_planner`, chosen once per config, [ADR-0008](0008-cell-simulation-sits-before-the-real-cell.md)) and plans **one operation per call**, all of its shares at once. It gets the operation, the start joint configuration of each robot involved (the cell simulation tracks it) and the cell-assembly snapshot, including holds. It returns, per share, **NC commands**: skill calls (preferred), Cartesian TCP moves with the arm posture pinned (`STAT`/`TU`), or, discouraged, joint moves (`RA1..RA6`). The cell simulation computes cell time by executing these commands, running the skills' own parametrized NC subroutines; the planner knows a skill only by its name and the meaning of its parameters.

Coupled motion (two robots moving one held product, as in a coupled insertion or a rigid carry) is **one operation with a share per robot**, like a handover; holding stays its own operation in which the holder does not move. A planner may set phase marks inside a coupled operation; the cell simulation turns them into `WAITM`. Between operations the cell simulation keeps scheduling and adds motion-level sync points where `conflicts` reports a clash. Transits stay per robot. Single-robot planners use a default that plans each share on its own.

Decided in #188, from the inventory in #187.

## Considered options

- **Per share, as before**: coupled planners (Dana's 12-DOF OMPL, Ignacio's rigid carry) cannot be expressed.
- **Per concurrent group**: the planner could couple unrelated operations, but scheduling inside a group would move from the cell simulation into every planner, and holding would have to become a moving operation.
- **Timed joint paths as the return**: one NC writer and exact durations, but joint coordinates transfer badly because the cell simulation and the cell are not perfectly aligned, and parametrized skills, which have proven robust on the cell, could not be used.
- **Skill motion supplied by the planner**: every planner would have to know every skill, and nothing would catch a mismatch with the real subroutine.
- **True synchronous motion (`CC_SYMO`) now**: the target, but it needs controller work first; phase marks with `WAITM` run on the cell today (Dana's "simple" mode).

## Consequences

- The cell simulation must run the skills' parametrized NC subroutines (`CALL`, R-parameters); today it only writes them into the NC as text.
- Measured poses still enter only through inspection and re-simulation (ADR-0008); a skill adapts on the cell only inside its movement primitives, within a range the cell simulation checks.
- Planning with a held product does not wait for the hold decision of #147; the snapshot carries the holds.
- Task-level schedulers from the theses (ScheduleStream, CP-SAT, Dana's `sync/`) are not motion planners at this seam; they belong behind the MRTAMP seam ([ADR-0007](0007-mrtamp-seam-is-a-stateless-plan-call.md)). Of TPG, only its conflict test is reused, as `conflicts`.
- Where the planner's scene comes from for planners in a separate GPU process is decided in #189.
