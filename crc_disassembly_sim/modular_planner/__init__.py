"""Modular motion planners for the CRC disassembly cell.

Provides three interchangeable planners for DisassemblySim, selectable at
runtime via set_planner().  All planners share the same G-code waypoint output
interface so the execution layer (GCodeProcessor / PyBullet IK) is unchanged.

Planners
--------
DIRECT    Original single-waypoint G1 move.  Zero extra dependencies; used as
          the default and as a baseline for comparing planner quality.

GNN       RoboBallet GNN policy generates intermediate collision-aware waypoints
          in normalised Cartesian space.  Requires the roboballet venv and a
          trained checkpoint.  See modular_planner/gnn_planner.py for details.

CUMOTION  NVIDIA cuMotion 1.1.0 GPU motion planner.  Plans collision-free joint
          trajectories on GPU; supports simultaneous dual-robot replanning.
          Requires CUDA 12.6 and the cumotion wheel (see cuMotion_wheels/).
          See modular_planner/cumotion_planner.py for details.

Execution layer (orthogonal to the planner choice above)
----------------------------------------------------------
DisassemblySim's `--execution` flag separately selects how the two arms are
coordinated while running: `reactive` (live per-tick yield arbitration,
default) or `tpg` (offline Temporal Plan Graph — solo rehearsal, pairwise
segment-safety classification, one WAITM rendezvous per dependency edge).
See modular_planner/tpg.py for the tpg implementation.

Usage
-----
    from modular_planner import PlannerType

    sim = WSKAgiloDrive(server_active=False)

    # Default — original single G1 move:
    sim.set_planner(PlannerType.DIRECT)

    # GNN collision-aware path planning (task-aware checkpoint selection):
    sim.set_planner(
        PlannerType.GNN,
        checkpoint_dirs={
            'independent': '/path/to/checkpoints/stage4_parallel',
            'cooperative': '/path/to/checkpoints/stage7',  # set when Stage 7 is trained
        },
        roboballet_root='/path/to/robo_ballet',
    )
    # Each task's planner is lazy-loaded on first use.
    # _execute_move_D always uses task='independent'.
    # _execute_move_both uses task='independent' by default;
    # pass cooperative=True for dual-arm battery cover transport.

    # cuMotion GPU motion planning (dual-robot):
    sim.set_planner(PlannerType.CUMOTION)
"""

from enum import Enum


class PlannerType(Enum):
    """Selectable motion planners for DisassemblySim._execute_move_both."""
    DIRECT   = 'direct'    # Original single G1 move — zero extra overhead.
    GNN      = 'gnn'       # RoboBallet GNN collision-aware path planning.
    CUMOTION = 'cumotion'  # NVIDIA cuMotion GPU motion planner.


from modular_planner.gnn_planner import GNNWaypointPlanner, GNNPlannerRegistry
from modular_planner.cumotion_planner import CuMotionWaypointPlanner
from modular_planner.scene import ObstacleBox, SceneConstraint

__all__ = [
    'PlannerType',
    'GNNWaypointPlanner',
    'GNNPlannerRegistry',
    'CuMotionWaypointPlanner',
    'ObstacleBox',
    'SceneConstraint',
]
