# Milestone sim runs, 2026-10-21

Setup: `crc_disassembly_sim` (this subtree) started headless (`CRC_SIM_HEADLESS=1`) in `D:\sfb-assembly-cases\.venv-sim`,
`start_server(port=12451)`; driver `examples/mrtamp_demo_runs.py --sim 127.0.0.1:12451` in the techstack venv.
Planner `straight_line_checked`, `orientation` [0, 0, 0], `reference_poses` {first base part: [-1.2, -1.6, 0.7, 0, 0, 0]}.
Cell file `cases/transformer_cell_simulation.json`, target `simulation`, B1 (`shortest`; `critical_path` gives the same battery plan).
Run records (sequence graph copy removed): `angle_grinder_run_record.json`, `battery_replacement_run_record.json`.

| Product | Status | Rounds | Operations | Simulated makespan | Screwing in per robot |
|---|---|---|---|---|---|
| angle grinder | goal_reached | 1 | 12 | 179.3 s | robot_1: 4 |
| battery | goal_reached | 1 | 86 | 1030.5 s | robot_1: 41 |

## Infeasible operations met on the way, with reason codes

1. All runs, supply slots at [0, 0, 0, ...]: `out_of_reach`, "transit to op_1: robot_1 cannot reach (0, -1188, 1584) mm (no IK solution)".
   Fixed: nominal reachable slot poses (park 20261021-015208-1e02).
   B1 then replanned the same op 20 times (`replan_limit_reached`): ticketed as AT 250.
2. Plan function paths in mm: every pose of a share was about 1000× too far. Fixed: paths in metres (park 20261021-015205-6ff6).
3. Planner default orientation [π, 0, 0]: no grid point of the cell reachable by both robots; identity orientation (the home TCP orientation) reaches 445 points. Set in the driver.
4. Battery, fixture at [-1.2, -1.75, 0.5]: `collision`, "op_30: robot_1 at (-763, -2167, 740) mm: example_robot/cable_bar touches spanntisch (-1 mm)", then the same for robot_2; then `goal_unreachable`. Fixed by moving the placeholder fixture to [-1.2, -1.6, 0.7].
5. Battery, two robots screwing together: `check_feasibility` refuses every pair, e.g. "op_24:transit of robot_2 conflicts with op_4:transit of robot_1" (both fetch from the one H5 nest plate). Robot 2 alone reaches all 41 screws. Parked (M, 20261021-022420-ebc8).

## Cell rehearsal, 2026-10-21 (park 20261021-014951-03ca, option B)

Same simulation and driver; per product a second run with the real cell's file `cases/transformer_cell.json` (Robot 2 alone screws),
target `cell`. Its all-zero supply slot poses are swapped in memory for the simulation file's nominal ones (park 20261021-105907-7b81, S).
Records and NC programs of this run were replaced by the re-run of 2026-10-23 (below).

| Product | Status | Rounds | Operations | Simulated makespan | Screwing in per robot |
|---|---|---|---|---|---|
| angle grinder | goal_reached | 1 | 13 | 111.2 s | robot_2: 4 |
| battery | goal_reached | 1 | 87 | 962.2 s | robot_2: 41 |

The NC programs hold no motion: each is `; ---- inspection_1: inspecting`, the segment end and `M30`.
A cell-targeted plan opens with an inspection of the start state (`src/mrtamp/inspections.py`), and the simulation ends
the NC at the first inspection (`crc_disassembly_sim/cell_simulation.py`, `_nc_programs`). Reported as AT 252.

# Re-run on the box, 2026-10-23

After AT 252 (merged, !96): the cell rehearsal now runs segment by segment, one round per inspection, and each round's
simulation report carries the NC of its segment. The simulation's cell file gives each robot its own H5 screw nest plate
(park 20261021-022420-ebc8). Setup as above, on this box: headless sim in a local sim venv, fresh sequence graph exports
(`examples/export_demo_products.py`), driver passing the product folder to `load_export` (AT 246 portable paths).
Records: `*_run_record.json` (sequence graph copy removed) and `*_cell_rehearsal_run_record.json` (also each round's
observed snapshots removed for size).

| Product | Target | Status | Rounds | Operations (first plan) | Simulated makespan (round 1) | Screwing in per robot |
|---|---|---|---|---|---|---|
| angle grinder | simulation | goal_reached | 1 | 12 | 179.3 s | robot_1: 4 |
| battery | simulation | goal_reached | 1 | 86 | 1030.5 s | robot_1: 41 |
| angle grinder | cell (rehearsal) | goal_reached | 6 | 13 | 123.1 s | robot_2: 4 |
| battery | cell (rehearsal) | goal_reached | 43 | 87 | 1068.6 s | robot_2: 41 |

No round of the four runs has an infeasible operation, a rejection or an unreachable goal.

NC programs with motion, marked as rehearsal output (placeholder slot and fixture poses, not for the cell), in
`nc/<product>/round_<n>_MRTAMP_CH<channel>.MPF`: grinder CH1 round 2 and CH2 rounds 3 to 6; battery CH1 round 2
(lid) and CH2 rounds 3 to 43 (one screw each). The other rounds' programs hold only inspection comments and `M30`.
The rehearsal swaps in the simulation file's whole supply slot list (park 20261021-105907-7b81 swapped the poses), so
its Robot 2 fetches from `screw_nest_plate_h5_robot_2` at (-1.8, -2.4, 0.8); the real cell file has one nest plate
of another name. Harmless while the poses are placeholders and the NC is marked rehearsal-only.

The battery's simulation-targeted plan still screws 41/0 on robot_1: B1 never pairs two screws, because the sequence
graph holds the 41 order-free lid screws as one chain. Decided by the user (park 20261023-004824-3469): AT 151 closes
with this split; B1 learns to split a screw group in AT 264. The per-robot nest plates are in place for it.

## Reach of the battery screws by fixture pose (shares feasible of 41)

| Fixture | robot_1 | robot_2 |
|---|---|---|
| [-1.2, -1.75, 0.5] | 40 | 39 |
| [-1.2, -1.6, 0.7] | 41 | 41 |
| [-1.2, -2.0, 0.7] | 39 | 35 |
| [-1.4, -2.0, 0.6] | 36 | 38 |
