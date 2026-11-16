# GNN Planner — Training and Testing

This folder contains scripts for training the RoboBallet GNN used by
`modular_planner/gnn_planner.py`, along with interface tests that verify the
planner contract without requiring a live GPU or checkpoint.

---

## Background

The GNN planner is based on **RoboBallet**, a multi-robot motion planning
framework that trains a graph neural network policy via reinforcement learning.

- **Paper:** Scheikl et al., *"Robot Ballet: Robotic Loco-manipulation for
  Surgical Tasks,"* Science Robotics, 2025.
  [doi:10.1126/scirobotics.ads1204](https://www.science.org/doi/10.1126/scirobotics.ads1204#supplementary-materials)

- **Zenodo (code + weights):**
  [zenodo.org/records/16546403](https://zenodo.org/records/16546403)

The original RoboBallet implementation targets Franka Panda arms in MuJoCo.
Our adaptation replaces this with two Comau NJ290 arms in PyBullet, but the
policy network and training algorithm are unchanged.

---

## What we changed from the original

| Item | Original | This work |
|---|---|---|
| Robot | Franka Panda (7-DOF) | Comau NJ290 (6-DOF, Comau cell) |
| Simulator | MuJoCo | PyBullet |
| `num_robots` | 4 (Panda quad-arm) | **2** (Comau dual-arm) |
| Training stages | 1-stage | **7-stage curriculum** (see below) |
| Curriculum manager | manual | `stage_manager.py` (automated) |
| Training visualisation | TensorBoard only | `plot_training.py` (multi-stage overlay) |
| Checkpoint restore | optional flag | auto-detects latest step |
| Cooperative task support | none | Stage 5–6 synchronized transport (velocity sync, 6D rigid-body pose maintenance) |
| Robot→robot graph edges | constant base pose | dynamic tip-to-tip pose (`--robot_tip_edges`, Stage 5+) |

The interface in `modular_planner/gnn_planner.py` translates between the CRC
world frame (metres, Comau kinematics) and the RoboBallet virtual workspace,
so the trained policy does not need to be retrained when the physical cell
layout changes — only the coordinate transform needs updating.

---

## Curriculum stages

Training uses a 7-stage curriculum of increasing difficulty:

| Stage | Targets | Obstacles | Orientation | Distance threshold | Advance when |
|---|---|---|---|---|---|
| 1 | 3 | 5 | no | 15 cm | ≥ 90 % targets done (3 evals) |
| 2 | 3 | 10 | no | 5 cm | ≥ 50 % targets done (3 evals) |
| 3 | 5 | 15 | **yes** | 5 cm | ≥ 70 % targets done (3 evals) |
| 4 | 10 | 30 | **yes** | 2.5 cm | ≥ 70 % targets done (5 evals) |
| 5 | 2 | 10 | **yes** | 2.5 cm | ≥ 80 % targets done (5 evals) — parallel movement |
| 6 | 1 | 10 | **yes** | 2.5 cm | ≥ 80 % targets done (5 evals) — synchronized arrival |
| 7 | 2 | 10 | **yes** | 2.5 cm | final — rigid body transport, runs until stopped |

Each stage restores the previous stage's checkpoint and continues training.
All stages use `num_robots=2`.

**Stage 5** — parallel movement, two independent targets:

- **Distance maintenance** (weight 0.4): robots are rewarded each step for
  keeping their tip-to-tip distance close to the value at episode start (±5 cm).
- **Velocity synchrony** (weight 0.3): bonus when both robots are near their
  targets and moving at matched speeds.
- **Tip-to-tip graph edges**: robot→robot GNN edges encode dynamic tip pose
  instead of constant base pose.
- No hold penalty — robots move freely after reaching their target.

**Stage 6** — synchronized arrival at geometrically consistent targets:

- `num_targets=1` so the simultaneous bonus is unambiguous: there is exactly one
  pair of targets and the episode ends when both are reached. With multiple
  targets and independent per-robot advancement the bonus can fire for mismatched
  target indices (e.g. robot A on target 1, robot B on target 0); the
  single-target design eliminates this.
- **Derived targets**: robot 1's target is computed from robot 0's target using
  `T_rel = inv(tcp_A) @ tcp_B`, same as Stage 7. This means Stage 6 already
  trains with grasp-consistent goal geometry — Stage 7 then only adds the pose
  maintenance reward rather than changing the target structure.
- **Hold reward** (weight 0.1): the first robot to reach its target is rewarded
  for staying near it while the partner catches up.
- **Simultaneous bonus** (weight 0.1): one-time reward when both robots finish
  within 20 steps of each other.
- **Distance maintenance** (weight 0.3) continues as the parallel-movement
  signal.

**Stage 7** — rigid body transport, full 6D pose maintenance:

- **6D relative TCP pose reward** (weight 0.4): both position error (≤5 cm) and
  rotation angle error (≤0.1 rad) of the inter-arm relative pose are penalised
  each step. Enforces that the piece doesn't twist or shear during transport.
- **Derived targets**: robot 1's target is computed from robot 0's target using
  the initial inter-TCP transform `T_rel = inv(tcp_A) @ tcp_B`, ensuring target
  separation matches starting grasp geometry.
- **Distance maintenance** (weight 0.2) is kept as an auxiliary signal.
- Note: within-episode target advancement remains per-robot (not synchronized).
  This is a known approximation — see thesis limitations section.

All cooperative weights default to 0.0 — Stages 1–4 are unaffected.

---

## Replicating the training

### Prerequisites

1. **Clone or download RoboBallet** from Zenodo:
   [zenodo.org/records/16546403](https://zenodo.org/records/16546403)

2. **Set up the roboballet venv** (JAX + Launchpad + tmux):
   ```bash
   cd robo_ballet
   python3.10 -m venv venv_robo_ballet
   source venv_robo_ballet/bin/activate
   pip install -e .
   # Follow the Zenodo README for JAX/Launchpad install
   ```

3. **Apply the modified upstream files** from `patches/` — these replace three
   files in the original RoboBallet source that we had to extend:

   | File in `patches/` | Destination in `robo_ballet/` | What changed |
   |---|---|---|
   | `patches/compute_features.py` | `roboballet/agents/compute_features.py` | `FeatureConfig.robot_tip_edges` field; robot→robot edges switch from constant base pose to dynamic tip-to-tip pose when set |
   | `patches/planning_env.py` | `roboballet/environments/planning_env.py` | `PlanningEnvConfig` gains `sync_reward_weight`, `simultaneous_bonus_weight`, `hold_reward_weight`, `distance_maintenance_weight/tolerance`, `relative_pose_maintenance_weight`, `relative_pose_maintenance_tolerance_pos/rot` (all 0.0 default — zero cost to Stages 1–4); `_cooperative_reward()` with velocity sync, scalar distance maintenance, and 6D relative TCP pose maintenance; `step()` tracks per-robot done timestamps |
   | `patches/generate_start_states.py` | `roboballet/environments/generate_start_states.py` | `--pair_sep_min/max` flags for target separation constraints; `--derive_second_target` flag: when set, robot 1's target is derived from robot 0's target using the initial inter-TCP transform (ensures rigid-body consistent goal geometry for Stage 7) |
   | `patches/launch.py` | `roboballet/launch/launch.py` | `--robot_tip_edges` absl flag; `--replay_max_size` flag (prevents Reverb OOM on constrained machines); `--train_steps` override; `local_mp` launch type with tmux terminal |

   ```bash
   cp patches/compute_features.py        /path/to/robo_ballet/roboballet/agents/compute_features.py
   cp patches/planning_env.py            /path/to/robo_ballet/roboballet/environments/planning_env.py
   cp patches/generate_start_states.py   /path/to/robo_ballet/roboballet/environments/generate_start_states.py
   cp patches/launch.py                  /path/to/robo_ballet/roboballet/launch/launch.py
   ```

4. **Copy `stage_manager.py`** into `robo_ballet/roboballet/launch/`
   (it expects to live there; `REPO_ROOT` is resolved two directories up from
   the script's own location, i.e. the `robo_ballet/` root).

5. **Copy `plot_training.py`** into `robo_ballet/roboballet/launch/`
   (same convention — run from the robo_ballet root).

### Run the full curriculum

```bash
# Inside robo_ballet/, with venv_robo_ballet active:
python roboballet/launch/stage_manager.py
```

To resume after an interruption:
```bash
python roboballet/launch/stage_manager.py --start_stage 3 --resume
```

`stage_manager.py` patches `hparams.py` automatically for each stage,
generates start states if they don't exist, launches Launchpad training in a
tmux session, monitors TensorBoard, and advances to the next stage when the
success threshold is met.

### Visualise training progress

```bash
# After one or more stages have produced tensorboard logs:
python roboballet/launch/plot_training.py \
    --checkpoint_base . \
    --output training_progress.png \
    --smooth_window 5
```

Expects directories `checkpoints_stage1/tensorboard_log` …
`checkpoints_stage7/tensorboard_log` in the current directory. Stages that
haven't run yet are silently skipped.

### Use a trained checkpoint

Because the cooperative fine-tuning (Stages 5–7) specialises the network and
degrades independent-reaching performance, **`GNNPlannerRegistry` maintains two
separate checkpoints side by side**, selected automatically by task:

| Task | Best known step | Training-output location (`robo_ballet/`) | Deployed location (used by `DisassemblySim.py`) |
|---|---|---|---|
| Independent multi-target reaching (screws, single moves) | **1 730 000** | `checkpoints_stage4/model_checkpoints/1730000/` | `../../checkpoints/stage4_parallel/1730000/` |
| Cooperative rigid-body transport (battery cover, dual-arm carry) | TBD | `checkpoints_stage7/model_checkpoints/` | not yet trained — `../../checkpoints/stage7/` |

Note these are two *different* directories: `stage_manager.py` writes training output under
`robo_ballet/checkpoints_stageN/model_checkpoints/<step>/`; to deploy a checkpoint for
`DisassemblySim.py` to actually use, copy (or rename) that step folder into this repo's
`checkpoints/<task-folder>/<step>/` (see `checkpoints/README.md`). `GNNPlannerRegistry` scans
whichever folder it's pointed at for the highest-numbered step subdirectory itself — no
step number is ever passed on the command line.

The Stage 4 step-1 730 000 checkpoint was trained on ProKI-desktop (August 2026)
and is the best independent-reaching result to date. Unzip `1730000.zip` (kept in
`~/Downloads/` or the cloud drive) into `checkpoints/stage4_parallel/1730000/` under the
`probabilistic_disassembly_planner` repo root (**not** inside `robo_ballet/`).

```bash
# From probabilistic_disassembly_planner/submodule_crc_disassembly_sim/, with defaults
# already pointing at ../checkpoints/stage4_parallel and ../checkpoints/stage7:
python DisassemblySim.py --planner gnn --roboballet_root <path-to>/submodule_crc_disassembly_sim/robo_ballet

# Or override the checkpoint locations explicitly:
python DisassemblySim.py \
    --planner gnn \
    --checkpoint_dir_independent ~/robotics_env/probabilistic_disassembly_planner/checkpoints/stage4_parallel \
    --checkpoint_dir_cooperative ~/robotics_env/probabilistic_disassembly_planner/checkpoints/stage7 \
    --roboballet_root <path-to>/submodule_crc_disassembly_sim/robo_ballet
```

There is no `--num_robots` flag on `DisassemblySim.py` — the sim is fixed to the two-arm Comau
cell, so `num_robots=2` is implicit throughout; only `stage_manager.py`/`launch.py` (on the
training side) take an explicit `--num_robots`. A Stage 3 semi-trained checkpoint can be used
for end-to-end integration testing before Stage 4 completes, by pointing
`--checkpoint_dir_independent` at its deployed location the same way.

### cuMotion as an alternative to GNN for approach planning

`PlannerType.CUMOTION` is the third option alongside `DIRECT` and `GNN`. It uses NVIDIA
cuMotion GPU planning for joint-space trajectories. No checkpoint file is needed — this is the
planner axis; how the two arms are *coordinated* while running is a separate, orthogonal
choice (`--execution {reactive,tpg}`, see `../README.md`'s "Planner overview" for both modes).
The `reactive` default replans each robot every ~10 sim steps, treating the other robot's live
AABB as a dynamic obstacle; `tpg` instead rehearses each arm's motion offline once, classifies
conflicting segment pairs, and executes both arms asynchronously with `WAITM` rendezvous only
where a real conflict exists.

**Python environment:** cuMotion 1.1.0 is installed in `.venv` (Python 3.14)
via the `cp314` wheel.  DisassemblySim runs in `.venv` — no separate Python 3.10
environment is needed.  The `schedulestream_env` (Python 3.10) venv that was
previously used for cuMotion is now redundant and has been deleted.

---

## Known limitations and future work

### Catastrophic forgetting — no task-conditioned policy

Sequential fine-tuning through Stages 5–7 specialises the network for
cooperative transport and degrades the independent-reaching behavior learned in
Stage 4. After Stage 7, the network will tend to move in parallel even when not
required. The workaround — separate checkpoints per task type — is functional
but not architecturally clean.

**The proper solution is a task-conditioned policy.** A task embedding vector
(e.g. a small one-hot or learned vector) would be appended to each robot's node
features, and training would sample both independent and cooperative episodes
every batch. At inference, passing a different embedding switches behavior
without changing weights.

What this would require:
1. Extend each robot's node feature vector in `compute_features.py` with the
   task embedding dimension.
2. Adjust the GNN node encoder input size accordingly.
3. Change the training sampler to draw from both task distributions per batch.
4. The observation-space change makes existing checkpoints incompatible — full
   retraining from Stage 1 is needed (~2 weeks on ProKI hardware).
5. Lighter alternative without architecture changes: **Elastic Weight
   Consolidation (EWC)**, which adds a regularisation term penalising weight
   drift from the Stage 4 solution, reducing forgetting during Stages 5–7.

This was scoped out in August 2026 due to time constraints. It is a recommended
direction for follow-up work.

### Approximate synchronized target advancement

With `num_targets > 1` and independent per-robot advancement, the two robots
can be working on different target indices simultaneously (robot A on target 1
while robot B is still on target 0). True synchronized advancement — where the
group only moves to the next target pair after all robots have finished the
current one — would require modifying `step()` in `planning_env.py` to block
individual advancement, and exposing the partner's done-state in the
observation. The current approximation uses `hold_reward_weight > 0` to
encourage lingering near finished targets, which produces roughly synchronized
arrivals but does not enforce strict pairing across waypoints.

---

## Interface tests

`tests/` contains five lightweight tests that verify `modular_planner` works
correctly without a GPU or checkpoint. Run them from the repo root:

```bash
python training/tests/test_interface_1_obstacles.py   # ObstacleBox construction
python training/tests/test_interface_2_pose.py        # 4×4 pose handling
python training/tests/test_interface_3_gcode.py       # G-code waypoint format
python training/tests/test_interface_4_transforms.py  # frame transforms
python training/tests/test_interface_5_cumotion.py    # cuMotion lazy-import guard
```

All five should print `Interface N OK` and exit 0 with no GPU required.
