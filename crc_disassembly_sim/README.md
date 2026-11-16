# CRC Disassembly Simulation — Multi-Robot Motion Planning

PyBullet simulation of the CRC transformer disassembly cell with two Comau NJ290
robots, extended with a modular motion planning framework supporting three
interchangeable planners.

## Features

- **Full CRC cell** — transformer cell geometry, two Comau NJ290 robots, tool
  storages, and assembly parts loaded from URDF
- **Modular planners** — switch at runtime between DIRECT, GNN, and cuMotion
- **cuMotion GPU planning** — NVIDIA cuMotion 1.1.0 for collision-aware joint-space
  trajectories, in either of two execution modes (see below)
- **Two execution layers for dual-arm coordination** — `reactive` (live per-tick
  yield arbitration between the arms) and `tpg` (an offline Temporal Plan Graph:
  solo rehearsal, pairwise segment-safety classification, one WAITM rendezvous per
  dependency edge — see `modular_planner/tpg.py`)
- **Frame-transform architecture** — all targets specified in world/cell frame;
  per-robot base transforms (and, where needed, per-robot planning URDFs) handled
  internally by the planner
- **GUI demo** — `training/tests/test_cumotion_gui.py` runs a 4-phase parallel
  movement sequence with PyBullet GUI visualisation

## Planner overview

Two independent axes select the motion-planning behaviour: **which planner** computes
waypoints (`--planner`), and **which execution layer** coordinates the two arms while
running them (`--execution`).

| Planner (`--planner`) | Description | Dependencies |
|---|---|---|
| `direct` | Single G1 waypoint move (original behaviour) | none |
| `gnn` | RoboBallet GNN policy, collision-aware Cartesian waypoints | JAX, RoboBallet checkpoint (see `checkpoints/README.md`) |
| `cumotion` | NVIDIA cuMotion GPU planner, joint-space | CUDA 12.6, cuMotion wheel |

| Execution (`--execution`) | Description |
|---|---|
| `reactive` (default) | Live, per-tick yield arbitration: a priority scheme decides which arm yields when the two arms come within a configurable margin, with progressive deceleration before any hard hold, escape/retrace on contact, and serialization as a livelock breaker. |
| `tpg` | Offline Temporal Plan Graph: each arm's motion is fixed in advance from a solo rehearsal, conflicting segment pairs get a dependency edge, and execution is otherwise fully asynchronous — each arm runs at its own speed and only waits at edges (one `WAITM` per edge, exported into both Sinumerik programs so the real cell reproduces the same ordering as the sim). |

The HMI exposes the combination directly as one dropdown: **"cuMotion - reactive"** and
**"cuMotion - asynchronous"** map to `(planner=cumotion, execution=reactive)` and
`(planner=cumotion, execution=tpg)` respectively; `gnn` and `direct` currently only run
under `reactive`.

## Repository structure

```
crc_disassembly_sim/
├── modular_planner/           # Planner package (DIRECT / GNN / cuMotion) + execution layers
│   ├── __init__.py            # PlannerType enum, public exports
│   ├── cumotion_planner.py    # CuMotionWaypointPlanner
│   ├── gnn_planner.py         # GNNWaypointPlanner, GNNPlannerRegistry (renamed from planner.py)
│   ├── tpg.py                 # Temporal Plan Graph: offline asynchronous execution layer
│   └── scene.py               # ObstacleBox, SceneConstraint
├── robot_descriptions/        # Comau NJ290 URDFs, XRDF, planner config YAML
│                               #   (includes auto-generated *_plan.urdf variants with
│                               #    multi-turn wrist joints clamped, used only for planning)
├── transformer_cell/          # CRC cell geometry URDFs and meshes
├── Objects/                   # Assembly part URDFs
├── Classes/                   # AssemblyPB, depth camera helpers
├── methodes/                  # GCodeProcessor, interpolation
├── cuMotion_wheels/           # install instructions for the cuMotion wheel (wheel not tracked)
├── robo_ballet/               # RoboBallet package used by the GNN planner (--roboballet_root)
├── tpg_cache/                 # Cached rehearsal/planning results for the tpg execution layer
├── training/                  # GNN training docs + interface tests (see training/README.md)
│   └── tests/
│       ├── test_cumotion_gui.py   # GUI demo: 4-phase dual-robot movement
│       └── test_cumotion_sim.py   # Headless integration test
├── tests/
│   ├── test_sim_server_protocol.py  # Server wire protocol and error handling (no pybullet needed)
│   ├── test_cell_simulation.py      # simulate / check_feasibility / configure_planner (no pybullet needed)
│   ├── test_planner_worker.py       # planner_worker client and protocol against a fake worker (no pybullet needed)
│   └── test_straight_line_planner.py # straight_line_checked on the Comau URDF in pybullet DIRECT
├── DisassemblySim.py          # Main simulation class (WSKAgiloDrive)
├── sim_server_protocol.py     # Server framing (legacy + length-prefixed), invalid_request / internal_error
├── cell_simulation.py         # Operation plans for the MRTAMP app: simulate, check_feasibility, planner registry
├── straight_line_planner.py   # straight_line_checked: straight lines, IK + collision check in pybullet
├── planner_worker.py          # Planner workers: the worker side and wire format (imports only sim_server_protocol)
├── planner_worker_client.py   # planner_worker: sends each operation's scene description to a planner worker
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Installation

### Prerequisites

- NVIDIA GPU with CUDA 12.6 driver (driver ≥ 525.85)
- Python 3.14 (native WSL setup, recommended) **or** Python 3.10 (Docker)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (Docker path only)

### cuMotion Installation

cuMotion is distributed by NVIDIA and is **not** on PyPI.  Download the
`cumotion-1.1.0-cuda12.6` package from the NVIDIA developer portal:

> **Download:** https://developer.nvidia.com/isaac/download

The full archive includes wheels for Python 3.10–3.14.  The wheel is licensed by NVIDIA and is
not redistributed here; see `cuMotion_wheels/README.md` for where to put it.

### Native WSL setup (Python 3.14 — recommended)

The development venv is `~/robotics_env/.venv` (Python 3.14).
cuMotion 1.1.0 is already installed there using the cp314 wheel.

```bash
# Re-create from scratch if needed:
python3.14 -m venv ~/robotics_env/.venv
source ~/robotics_env/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install <path-to>/cumotion-1.1.0-cp314-cp314-linux_x86_64.whl
# cumotion_vis is skipped — open3d has no cp314 wheel; PyBullet handles visualisation
pip install -r requirements.txt
```

`pybullet-industrial` is on PyPI and is included in `requirements.txt`.

### Docker (Python 3.14, matches the native venv)

The Dockerfile targets Python 3.14 + the cp314 cuMotion wheel, the same configuration as the
native `.venv` (updated 2026-08-19/20; it previously targeted a stale Python 3.10/cp310 setup —
if you see that mentioned elsewhere, it is out of date).
Use this to run the sim server in a container on a machine without a native WSL Python 3.14 venv.
The HMI always runs on Windows and connects via TCP regardless.

```bash
# Build the image (cuMotion_wheels/ must contain the cp314 wheel, see its README)
docker build -t crc_disassembly_sim .

# Headless test
docker run --gpus all --rm crc_disassembly_sim python3 training/tests/test_cumotion_sim.py

# GUI demo (Linux host with X11)
docker compose up
```

On Windows/WSL, start XLaunch first and set `DISPLAY` before running:

```bash
export DISPLAY=<your-host-ip>:0.0
docker compose up
```

**Status:** the image builds successfully end-to-end (see `PROGRESS_LOG.md` 2026-08-19/20 for the
fix history). It has not yet been run against the battery benchmark end-to-end — that is still
open. `cumotion_vis` is not installed in the image — `open3d` has no cp314 wheel, and it is not
needed since PyBullet handles visualisation.

## Running the demos

```bash
# GUI visualisation — 4-phase dual-robot sequence in the full CRC cell
python training/tests/test_cumotion_gui.py

# Headless integration test — verifies cuMotion + PyBullet round-trip
python training/tests/test_cumotion_sim.py
```

Five additional lightweight interface tests (no GPU or checkpoint required) live in
`training/tests/test_interface_*.py` — see `training/README.md`'s "Interface tests" section.

## Running the simulator

```bash
python DisassemblySim.py --planner cumotion --execution tpg --assembly path/to/assembly.xml
```

The full CLI (`--planner {direct,gnn,cumotion}`, `--execution {reactive,tpg}`,
`--checkpoint_dir_independent`, `--checkpoint_dir_cooperative`, `--roboballet_root`,
`--assembly`, `--server_active`) is defined at the bottom of `DisassemblySim.py`; run with
`--help` for the current, authoritative list. Most day-to-day usage goes through the HMI
(`dasg_hmi.py`, in the sibling `submodule_crc_disassembly_sequence_generator/`), which builds
this command line for you from the Motion planner dropdown.

## Using the planner programmatically

```python
from DisassemblySim import WSKAgiloDrive
from modular_planner import PlannerType

# server_active=False so planner setup happens before the blocking TCP server loop starts
sim = WSKAgiloDrive(server_active=False)

# GPU motion planning with cuMotion
sim.set_planner(PlannerType.CUMOTION)

# grip() takes the part name, a Cartesian path, and an approach direction —
# see DisassemblySim.py for the exact signature and how the HMI calls it.
sim.grip(part_name="03_rotor_short", path=[...], direction=...)

# Or switch back to the original single-waypoint move
sim.set_planner(PlannerType.DIRECT)
```

## License

Project code: see [LICENSE](LICENSE).  
cuMotion: [NVIDIA Isaac ROS Software License](https://developer.nvidia.com/isaac/download).


## Operation plans (MRTAMP app)

`cell_simulation.py` (no pybullet) implements the three requests the techstack's run controller sends, as plain dicts
over the framed protocol: `configure_planner` (choose the motion planner once, by name), `simulate` and
`check_feasibility`. Motion planners register with `CellSimulationService.register_planner(name, factory)` and implement
`MotionPlanner`: `plan_operation` plans one operation, all of its shares at once, from each robot's start pose and
joint configuration and the cell-assembly snapshot before it (holds included), and `conflicts`; a single-robot planner
implements `plan_share`, `plan_transit` (optionally `path_start`) instead, which the default `plan_operation` calls per
share. `DisassemblySim` registers
`straight_line_checked` (`straight_line_planner.py`): each share and transit is a straight line, sampled every 1 cm,
with the arm placed by IK on its URDF and every coupled body (tools, camera) carried along; any contact with the
scene is `collision`, an unreachable pose `out_of_reach`, and two robots' segments that come within
`conflict_margin` get a motion-level sync point. Poses relative to a reference part are placed by the
`reference_poses` option of `configure_planner`. There is no fallback: with no planner
configured, or when a planner finds nothing, an operation with a path is reported as an infeasible operation
(`no_path_share` / `no_path_transit`) and the simulation stops there with no NC. Cell time is computed from the planned
motion, dwells and tool change constants; every sync point becomes a `WAITM` mark in both NC programs; for a
cell-targeted run both programs end at the first inspection while the whole plan is rehearsed. *Physics* execution
needs a registered executor and is otherwise refused.

Planners that need a GPU or Linux (cuRobo, cuMotion, OMPL) run as **planner workers**, each in its own process and
venv (techstack ADR-0011). `DisassemblySim` registers `planner_worker` (`planner_worker_client.py`), configured with
the worker's `address` (`"host:port"`), and every `name = host:port` line of `[planner_workers]` in `config.ini` as a
planner of that name. Per operation it sends the worker a scene description (the operation, each robot's start pose
and joint configuration, mounted tool and held part, the parts present, the reference part's pose, static obstacles)
and takes back each share's transit and share motion as NC command dicts; it asks the worker too whether two
robots' segments conflict. A worker that cannot be reached or does not answer within `timeout` makes the operation
infeasible (`no_path_share`), never direct motion; a worker error is an `internal_error`. A worker subclasses
`planner_worker.PlannerWorker` and calls `planner_worker.serve(worker, host, port)`; it needs only
`planner_worker.py` and `sim_server_protocol.py`. The wire format is in `planner_worker.py`.
