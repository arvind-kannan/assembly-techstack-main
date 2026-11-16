# RoboBallet — Comau CRC Cell Fork (cuMotion / GNN Motion Planning)

This is a fork of DeepMind's **RoboBallet**, adapted to train a multi-robot motion-planning
policy for the two-arm Comau NJ290 disassembly cell used in
[`probabilistic_disassembly_planner`](../probabilistic_disassembly_planner), developed as part
of a Master's thesis at the Institute of Production Science (wbk), KIT. The trained GNN policy
is one of two interchangeable motion planners wired into that project's simulator
(`DisassemblySim.py`), alongside NVIDIA cuMotion — see
[`submodule_crc_disassembly_sim/README.md`](../probabilistic_disassembly_planner/submodule_crc_disassembly_sim/README.md)
for how a checkpoint trained here is actually consumed, and
[`submodule_crc_disassembly_sim/training/README.md`](../probabilistic_disassembly_planner/submodule_crc_disassembly_sim/training/README.md)
for the full, up-to-date curriculum design, replication steps, and known limitations — that
document is the authoritative training reference for this project; treat this file as an
overview and provenance record for the fork itself, not the primary source.

**Upstream:** Lai, Go, Li, Kröger, Schaal, Allen, Scholz — *"RoboBallet: Planning for
Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning,"* Science Robotics,
2025 ([doi:10.1126/scirobotics.ads1204](https://www.science.org/doi/10.1126/scirobotics.ads1204)).
Code + weights archived at [zenodo.org/records/16546403](https://zenodo.org/records/16546403).

## What this fork changes vs. upstream

| Item | Upstream | This fork |
|---|---|---|
| Robot | Franka Panda (7-DOF), `num_robots=4` | Comau NJ290 (6-DOF), `num_robots=2` |
| Simulator | MuJoCo | Deployed via PyBullet (disassembly cell side); the training env's own kinematics/observation build still goes through `mujoco` (3.12.0 — needed at inference time too, not just training) |
| IK solver | **Stub** — `environments/ik_solver.py` raised `NotImplementedError` | Fully implemented: damped least-squares Jacobian pseudoinverse via MuJoCo's `mj_jacSite` |
| Training procedure | Single stage, full difficulty from scratch | **7-stage curriculum** (`launch/stage_manager.py`): Stages 1-4 independent reaching (increasing targets/obstacles/precision) → Stages 5-7 cooperative dual-arm transport |
| Process launch | Threaded, single process | `launch_type='local_mp'` — actor/learner/replay/evaluator run as isolated OS processes in separate tmux windows |
| Checkpoint handling | Single checkpoint, manual restore path | Consuming side (`GNNPlannerRegistry` in `submodule_crc_disassembly_sim`) lazy-loads two task-specific checkpoints (`independent` / `cooperative`) side by side |
| Robot→robot graph edges | constant base pose | optional dynamic tip-to-tip pose (`--robot_tip_edges`, Stage 5+) |
| Reward shaping | fixed | per-stage `hparams.py` patching (`stage_manager.py`) plus cooperative reward terms (`distance_maintenance`, `sync_reward`, `simultaneous_bonus`, 6D relative-pose maintenance) added for Stages 5-7 |

## Current training status

As of the last check: **Stage 4** (independent reaching — 10 targets, 30 obstacles, 2.5 cm / 15°
tolerance) is done, best checkpoint at step 1,730,000, deployed to
`probabilistic_disassembly_planner/checkpoints/stage4_parallel/`. **Stage 5** (cooperative
parallel reaching, two independent targets) is solved (`targets_done=2/2`, `final_score=1.0`).
**Stage 6** (synchronized single-target arrival) was reached, but training stopped there:
access to the remote GPU workstation it ran on was lost before Stage 6 could be confirmed solved
or advanced further, and it was not regained before submission. **Stage 7** (full 6D rigid-body
transport for the battery-cover dual-arm carry) was never trained. The delivered checkpoint
remains the Stage 4 one (step 1,730,000, see above); cooperative dual-arm transport is not
available as a trained policy. Resuming Stages 6-7 (or restarting them, e.g. with the alternating
task-type / fewer-steps reward idea noted in the thesis future-work section) is left for whoever
picks this up next. See `training/README.md`'s status section for further detail — this file is
not kept in lock-step with training progress, only with the code.

## Repository layout (this fork)

```
robo_ballet/
├── README.md                    ← this file
└── roboballet/                  ← the Python package
    ├── agents/                  ← GNN policy network, compute_features.py (graph construction)
    ├── environments/            ← planning_env.py, ik_solver.py, generate_start_states.py
    ├── launch/                  ← launch.py, stage_manager.py (curriculum), hparams.py, plot_training.py
    ├── train/                   ← training-loop internals
    └── data/                    ← generated start states (.pb files, per-stage)
```

## Installation

Follow `training/README.md` → "Replicating the training" → "Prerequisites" for the version this
project actually uses (Python 3.10 venv `venv_robo_ballet`, the `patches/` files this project
applies on top of upstream, and `stage_manager.py`/`plot_training.py` placement) — that section
is kept up to date. The block below is the **original, unmodified upstream install procedure**
and still applies as the base layer before any of this project's patches:

```bash
# Python 3.10 is required — strict version pins on JAX/cuDNN/Launchpad below.
python3.10 -m venv venv
source venv/bin/activate
pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install nvidia-cudnn-cu12==8.9.4.*
pip install -r requirements.txt

# Older jaxlib .so files have the executable-stack bit set; glibc 2.41+ rejects
# that even though nothing actually executes the stack — clear the flag:
find venv/lib/python3.10/site-packages/jaxlib -name '*.so' -exec execstack -c {} \;

# protoc + generated accessor library for the environment protobufs:
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.19.4/protoc-3.19.4-linux-x86_64.zip
unzip protoc-3.19.4-linux-x86_64.zip -d /tmp
/tmp/bin/protoc --proto_path=roboballet --python_out=roboballet roboballet/environments/data.proto
```

## Usage

**Don't run `launch.py` directly for real training** — use the curriculum wrapper, which patches
`hparams.py` per stage, generates start states if missing, launches training inside a tmux
session, polls TensorBoard for the success threshold, and auto-advances:

```bash
python roboballet/launch/stage_manager.py --start_stage=1
# resume an interrupted run instead of restarting the stage from scratch:
python roboballet/launch/stage_manager.py --start_stage=5 --resume
```

The raw upstream command below still works — it is literally what `stage_manager.py` invokes
per stage — but running it directly trains one fixed difficulty level from scratch with
whatever is currently in `hparams.py`, not the staged curriculum this project actually relies on:

```bash
python -m roboballet.launch.launch --lp_launch_type=local_mp
```

This saves model checkpoints to `./model_checkpoints` and TensorBoard logs to
`./tensorboard_log` (per curriculum stage, `stage_manager.py` uses `./checkpoints_stageN/`
instead via `--artifacts_prefix`).

**A gotcha worth knowing:** `stage_manager.py`'s module-level constants (`STALL_TIMEOUT_S`,
`REPLAY_MAX_SIZE`, ...) are read once at process start. Editing the file does **not** affect an
already-running `stage_manager` process — it must be killed and restarted. This caused a real
multi-day stall in practice (see the comment at the constants' definition in the file).

## Citing this work

```
@article{lai2025roboballet,
  title={RoboBallet: Planning for Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning},
  author={Lai, Matthew and Go, Keegan and Li, Zhibin and Kroger, Torsten and Schaal, Stefan and Allen, Kelsey and Scholz, Jonathan},
  journal={Science Robotics},
  volume={10},
  number={106},
  year={2025},
  publisher={American Association for the Advancement of Science}
}
```

## Known issues (upstream)

* During the training process, we use an IK solver to filter for tasks that at
  least have one IK solution. We have included pre-generated start states so
  that our results can be reproduced, but for adapting this codebase for future
  work, an IK implementation needs to be provided.
  See ik_solver.py.

  **Resolved in this fork:** `roboballet/environments/ik_solver.py` now contains a full
  `IkSolver` implementation (damped least-squares Jacobian pseudoinverse via MuJoCo's
  `mj_jacSite`) — no IK implementation needs to be supplied to adapt or retrain this codebase.

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product. This fork's modifications (Comau/PyBullet adaptation,
IK solver, curriculum training infrastructure, cooperative-task reward shaping) are original
work for this thesis project and are not endorsed by or affiliated with Google.
