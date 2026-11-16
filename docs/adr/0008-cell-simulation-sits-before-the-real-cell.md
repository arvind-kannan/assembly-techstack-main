# The cell simulation sits before the real cell

Every operation plan reaches the real cell only through the cell simulation: operation plan → cell simulation → real cell. The cell simulation plans all robot motion, executes the plan virtually and generates both NC programs; nobody writes NC by hand per product, and no operation plan goes to the cell unchecked. A run targets either the cell simulation (and ends there) or the real cell (always through the simulation).

The cell simulation **never falls back silently**. What it cannot plan, it reports as an infeasible operation with a reason code from a closed list; it stops at the first one and produces no NC. Today's fallbacks in `crc_disassembly_sim` (cuMotion "no path" → direct motion, reactive → serial) are removed. Times are cell time computed from the generated motion, not wall clock.

**NC programs end at every inspection.** Each `simulate` call rehearses the whole remaining plan, so a late infeasibility is found before the cell moves, but returns NC only up to and including the next inspection. If the inspection found what was expected, the run controller sends the rest of the same operation plan through the simulation again from the inspected cell-assembly snapshot (re-simulation); the MRTAMP planner is not asked again. Otherwise it replans. The cell operator presses Cycle Start once per segment.

**NC programs also end at every machine operation** (revised by #216). A machine (the DMG DMU 60, the Ponticon pE3D) has its own control, so the robots must not run on while it runs its cycle. After the robot has loaded the machine, the run controller asks the cell operator to start the machine's cycle and to confirm it is done, then sends the rest of the same operation plan through the cell simulation again. Starting the machine over its hardwired link later replaces only the operator's part. The cell simulation books the machine operation as a wait of its cycle time and does not model the cut.

Decided in #92 and #96, machine operations added in #216; revises #92 point 8 and the "applied as an offset at execution" part of #90 (point 6).

## Part-relative operation poses

Operations store poses relative to a **reference part** (usually the base part in the fixture), from CAD. An inspection measures the reference part's pose, and re-simulation places the following operations with it, so the NC always contains measured, collision-checked poses.

## Considered options

- **Cell simulation beside the cell** (a plan goes to one or the other): motion on the cell would be unchecked, and cell and simulation runs would not share one chain.
- **One NC program per run with an `R0` continue/stop handshake at inspections** (#92 point 8): the programs could not use poses measured during the run.
- **Measured reference-part pose as a SINUMERIK frame or R-parameter offset** (#90 point 6): keeps the NC valid without re-simulation, but the shifted motion is never collision-checked.
- **Fall back to direct or serial motion when planning fails**: the run goes on, but the NC is not what was planned and nobody learns why.

## Consequences

- A model-level stand-in with the same two calls (`check_feasibility`, `simulate`) lets the whole chain run before `crc_disassembly_sim` supports operation plans. It produces no NC, so the run controller refuses a cell-targeted run with it.
- The run controller, not the simulation, compares the observed snapshot at each inspection with the expected one.
- Assembly feasibility problems the sequence graph's reversal does not check (press fits, gravity, gripping at the supply) surface here as infeasible operations.
- The server changes (wrapped handlers, `invalid_request` / `internal_error`, length-prefixed framing) start from `crc_disassembly_sim` branch `thesis/final-sim-snapshot-2026-09-25`, the only branch with cuMotion, TPG, `WAITM` and dry/physics.
