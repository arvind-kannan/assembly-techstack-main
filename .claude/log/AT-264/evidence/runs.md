# Sim runs, 2026-10-23 (AT 264)

Setup as in AT 151's re-run on the box: `crc_disassembly_sim` headless in a local sim venv, port 12464; the AT 151
exports of both demo products (`examples/export_demo_products.py`, schema unchanged); cell file
`cases/transformer_cell_simulation.json` (a screw nest plate per robot); target `simulation`; driver
`examples/mrtamp_demo_runs.py`'s `run_demo_product` with the part positions of the product model (mesh centres, metres)
and the default minimum screw distance 0.2 m. Records: `*_run_record.json` (sequence graph copy and observed
snapshots removed).

| Product | Priority | Status | Rounds | Screwing in per robot | Simulated makespan | Feasibility queries (refused) |
|---|---|---|---|---|---|---|
| battery | shortest | goal_reached | 1 | robot_2: 24, robot_1: 17 | 518.3 s | 75 (0) |
| battery | critical_path | goal_reached | 1 | robot_2: 25, robot_1: 16 | 560.9 s | 76 (0) |
| angle grinder | shortest | goal_reached | 1 | robot_2: 4 | 83.1 s | 0 |

Before (AT 151, same exports and cell file): battery robot_1: 41, 1030.5 s; angle grinder robot_1: 4, 179.3 s.
The grinder's four screws are a collapsed group too: they go to Robot 2, which finishes them first, and none pairs
(closer than 20 cm).

## On the way: B1's feasibility queries were all rejected

First battery run with this branch: robot_2: 41, 762 queries, all `invalid_request` from the simulation:
"operation ids in a concurrent group must be unique" (every operation of a query was `op_1000000`), then, once
numbered apart, "a concurrent group has at most one operation per robot" (the query held every step of the other
robot that ended after this robot's clock). Fixed in this branch (park 20261023-104950-d5ed, S).

Neighbouring battery lid screws are 72 to 110 mm apart (mesh centres); the lid spans about 1.4 m.
