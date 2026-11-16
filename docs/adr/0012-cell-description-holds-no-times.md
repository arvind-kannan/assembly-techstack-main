# The cell description holds no times

The cell description lists the cell's robots, stations, tools and other resources with their capabilities and poses; details such as joint speeds come from the robot and station descriptions. It holds no times. Travel and tool-change times are a **travel-time table**: a cheap look-up (station to station, and the robot's home), kept in its own file next to the cell file under the same fingerprint, set once per cell by hand, by a Cartesian estimate, or recorded later from cell simulation runs. Process parameters that are not planned (feed rates, start offset, pullout, inspection and screw-in durations) are default parameters of the skills, which the cell simulation runs ([ADR-0010](0010-cell-simulation-planner-plans-one-operation-and-returns-nc-commands.md)). The POMDP, the sequence graph and the MRTAMP planners ask one module for their time estimates; the cell simulation computes its own times and does not read the table.

*Interim note (2026-10-28):* until #235 Travel-time table instead of times in the cell settings reshapes the table, #174 Build: Station and supply slots in the core Cell moves today's flat `timing` keys unchanged (feed rates and offsets included) into the travel-time table file, and the run passes that file to the cell simulation as a separate field, which reads its tool-change and inspection times from it. So, until #235, the cell simulation reads the table and process parameters sit in it. Decided by Max Goebels (Maintainer) on 2026-10-28 (AT 174 card 20261028-133901-157c), to break cell files only once.

Decided in the architecture review of 2026-10-19, see #235 Travel-time table instead of times in the cell settings.

## Considered options

- **Times as cell settings, grouped** (station-to-station legs, tool change and per-process parameters in the cell file): one file, but times are not a property of the cell; they follow from where the stations are and how fast the robots move, and the cell simulation already computes them.
- **Times computed from poses on demand** (inverse kinematics once per cell, joint moves over the URDF speed limits): closer to the simulated times, but adds a kinematics dependency to every caller of a look-up that the POMDP makes on every step.
- **Process parameters on the resources** (e.g. feed rates on the tool): two copies, because the skill subroutine that the cell simulation and the cell run has the same parameters.

## Consequences

- `CellSettings`, the gym's own `Cell` wrapper and the gym's use of `cell_settings.json` go away; `build_env` takes the core `Cell`. Rewards and graph times change, and existing graph exports go stale through their fingerprints.
- `Cell.timing` leaves the core; the cell-file break goes through #174 Build: Station and supply slots in the core Cell, so cell files break once.
- `milling_penalty` is a reward term and moves to the env's reward settings.
- #117 Assembly constants from the cell settings is superseded: its constants are skill parameters or travel-time table entries.
