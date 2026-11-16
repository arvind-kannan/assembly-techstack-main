"""Real-cell synchronous movement test — cuMotion planner.

Generates two Siemens Sinumerik .mpf G-code files (one per robot) for a
basic simultaneous move test on the physical CRC cell.  Both robots start at
their known safe-home positions and move to conservative mid-workspace targets.

SAFETY NOTES
------------
* Verify START and TARGET positions below against the current robot positions
  reported by the OPC-UA server BEFORE loading these programs on the controller.
* Run at reduced feed rate (F <= 5000) for the first attempt.
* Have an e-stop accessible.
* No end-effectors need to be attached for this test — the paths are TCP-frame
  moves only.

Usage
-----
    source ~/robotics_env/.venv/bin/activate
    cd ~/robotics_env/probabilistic_disassembly_planner/submodule_crc_disassembly_sim
    python test_real_cell_sync.py

Output: test_robot_C.mpf  and  test_robot_D.mpf  in the current directory.
Load these files on the Sinumerik controller via the usual program transfer path.
"""

from __future__ import annotations

import copy
import math
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Make modular_planner importable from this directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Cell geometry (metres, world frame — must match the physical cell calibration)
# ---------------------------------------------------------------------------

# Robot base positions and orientations (from DisassemblySim.__init__)
ROBOT_C_BASE_POS = np.array([0.0,     0.0,       0.0])
ROBOT_C_BASE_YAW = 0.0          # radians

ROBOT_D_BASE_POS = np.array([-1.99, -3.750, 0.0])  # measured at cell 02./03.09.2026; was -3.5153162 (235 mm error)
ROBOT_D_BASE_YAW = math.pi / 2  # radians (90° about Z)

# Known safe home positions (from DisassemblySim.save_point_C / save_point_D)
# These are the positions the sim moves to on startup — verified collision-free.
HOME_C_POS = np.array([0.0,   -1.5, 2.0])   # world frame, metres
HOME_C_ORN = np.array([0.0,    0.0, 0.0])   # Euler xyz, radians

HOME_D_POS = np.array([-1.99, -2.0, 2.0])   # world frame, metres
HOME_D_ORN = np.array([0.0,    0.0, 0.0])   # Euler xyz, radians

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------
# Conservative targets: stay high (Z ≥ 1.5 m), keep robots on their own side,
# modest XY offset from home.  Adjust these once the first run is confirmed safe.

TARGETS_C = [
    # (pos_world_m, orn_euler_xyz_rad)  — robot_C moves through these in order
    (np.array([0.0,  -1.5,  1.8]),  np.array([0.0, 0.0, 0.0])),   # slightly lower
    (np.array([0.2,  -1.5,  1.8]),  np.array([0.0, 0.0, 0.0])),   # small X offset
    (np.array([0.0,  -1.5,  2.0]),  np.array([0.0, 0.0, 0.0])),   # back to home
]

TARGETS_D = [
    # robot_D runs synchronously with robot_C (one target per target of C)
    (np.array([-1.99, -2.0,  1.8]),  np.array([0.0, 0.0, 0.0])),
    (np.array([-2.2,  -2.0,  1.8]),  np.array([0.0, 0.0, 0.0])),
    (np.array([-1.99, -2.0,  2.0]),  np.array([0.0, 0.0, 0.0])),
]

assert len(TARGETS_C) == len(TARGETS_D), \
    "TARGETS_C and TARGETS_D must have the same length (synchronous pairs)"

# ---------------------------------------------------------------------------
# Feed rate (mm/min equivalent in Sinumerik).
# Start conservative; increase after first confirmed run.
# ---------------------------------------------------------------------------
FEED_RATE = 5000.0

# ---------------------------------------------------------------------------
# Output file paths
# ---------------------------------------------------------------------------
OUT_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_robot_C.mpf")
OUT_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_robot_D.mpf")


# ---------------------------------------------------------------------------
# G-code helpers (copied from DisassemblySim — standalone, no sim required)
# ---------------------------------------------------------------------------

def _limit_angle(deg: float) -> float:
    """Clamp angle to [-360, 360]."""
    while deg > 360:
        deg -= 360
    while deg < -360:
        deg += 360
    return deg


def write_g_code(g_code: list, textfile: str, prefix: str = "", postfix: str = ""):
    with open(textfile, 'w') as f:
        f.write(prefix)
        for cmd in g_code:
            order = ['G', 'X', 'Y', 'Z', 'A', 'B', 'C', 'F']
            for key in order:
                val = cmd.get(key)
                if val is None:
                    continue
                if key == 'G':
                    f.write(f'G{val} ' if val != 0 else 'G1 ')
                elif key in {'X', 'Y', 'Z'}:
                    f.write(f'{key}{val * 1000:.3f} ')   # m → mm
                elif key in {'A', 'B', 'C'}:
                    f.write(f'{key}{_limit_angle(math.degrees(val)):.3f} ')
                else:
                    f.write(f'{key}{val} ')
            f.write('\n')
        f.write(postfix)


def transform_eulers(g_code: list) -> list:
    """Convert Euler convention: PyBullet xyz (intrinsic) → Siemens XYZ (extrinsic)."""
    out = copy.deepcopy(g_code)
    for cmd in out:
        if 'A' in cmd and 'B' in cmd and 'C' in cmd:
            rot = Rotation.from_euler('xyz', [cmd['A'], cmd['B'], cmd['C']])
            e = rot.as_euler('XYZ')
            cmd['A'], cmd['B'], cmd['C'] = float(e[0]), float(e[1]), float(e[2])
    return out


def pose_to_gcode_dict(pos_m: np.ndarray, orn_euler_xyz: np.ndarray,
                       feed: float = FEED_RATE) -> dict:
    """Build a G1 waypoint dict (world frame, metres + radians)."""
    return {
        'G': 1,
        'X': float(pos_m[0]),
        'Y': float(pos_m[1]),
        'Z': float(pos_m[2]),
        'A': float(orn_euler_xyz[0]),
        'B': float(orn_euler_xyz[1]),
        'C': float(orn_euler_xyz[2]),
        'F': feed,
    }


# ---------------------------------------------------------------------------
# Plan with cuMotion
# ---------------------------------------------------------------------------

def plan_with_cumotion(
    robot_id: str,
    base_pos: np.ndarray,
    base_yaw: float,
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    targets: list[tuple],
    feed: float,
) -> list[dict]:
    """Use cuMotion to plan a sequence of moves, return G-code waypoints.

    Falls back to direct single-waypoint G1 moves if cuMotion is unavailable
    (e.g. running without GPU).  The fallback is clearly labelled in the output.
    """
    from modular_planner import CuMotionWaypointPlanner, ObstacleBox
    from modular_planner.scene import SceneConstraint

    desc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'robot_descriptions')
    urdf_path   = os.path.join(desc_dir, 'comau_nj290_robotNC_sensor.urdf')
    xrdf_path   = os.path.join(desc_dir, 'comau_nj290.xrdf')
    config_path = os.path.join(desc_dir, 'comau_nj290_planner_config.yaml')

    planner = CuMotionWaypointPlanner(
        urdf_path=urdf_path,
        xrdf_path=xrdf_path,
        planning_config_path=config_path,
    )

    # Build a 4×4 base transform for frame conversion inside the planner.
    base_rot = Rotation.from_euler('z', base_yaw)
    T_base = np.eye(4)
    T_base[:3, :3] = base_rot.as_matrix()
    T_base[:3, 3]  = base_pos

    def make_pose(pos, orn_euler):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_euler('xyz', orn_euler).as_matrix()
        T[:3, 3]  = pos
        return T

    all_waypoints: list[dict] = []
    current_pos = start_pos.copy()
    current_orn = start_orn.copy()

    for i, (tgt_pos, tgt_orn) in enumerate(targets):
        print(f'  [{robot_id}] Planning segment {i + 1}/{len(targets)}: '
              f'{current_pos} → {tgt_pos}')
        try:
            wps = planner.plan(
                current_eef_pose=make_pose(current_pos, current_orn),
                target_eef_pose=make_pose(tgt_pos, tgt_orn),
                obstacles=[],          # no assembly in this basic test
                constraints=[],
                robot_base_pose=T_base,
                robot_name=robot_id,
            )
            for wp in wps:
                wp['F'] = feed
            all_waypoints.extend(wps)
            print(f'  [{robot_id}] Segment {i + 1}: {len(wps)} waypoints')
        except Exception as e:
            print(f'  [{robot_id}] cuMotion failed on segment {i + 1}: {e}')
            print(f'  [{robot_id}] WARNING: falling back to direct G1 move for this segment')
            all_waypoints.append(pose_to_gcode_dict(tgt_pos, tgt_orn, feed))

        current_pos = tgt_pos.copy()
        current_orn = tgt_orn.copy()

    return all_waypoints


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=== CRC Real-Cell Sync Test — G-code Generation ===')
    print(f'Robot C: home {HOME_C_POS} m,  {len(TARGETS_C)} targets')
    print(f'Robot D: home {HOME_D_POS} m,  {len(TARGETS_D)} targets')
    print()

    print('Planning robot_C ...')
    wps_C = plan_with_cumotion(
        robot_id='robot_C',
        base_pos=ROBOT_C_BASE_POS,
        base_yaw=ROBOT_C_BASE_YAW,
        start_pos=HOME_C_POS,
        start_orn=HOME_C_ORN,
        targets=TARGETS_C,
        feed=FEED_RATE,
    )

    print()
    print('Planning robot_D ...')
    wps_D = plan_with_cumotion(
        robot_id='robot_D',
        base_pos=ROBOT_D_BASE_POS,
        base_yaw=ROBOT_D_BASE_YAW,
        start_pos=HOME_D_POS,
        start_orn=HOME_D_ORN,
        targets=TARGETS_D,
        feed=FEED_RATE,
    )

    # Convert Euler convention before writing.
    wps_C = transform_eulers(wps_C)
    wps_D = transform_eulers(wps_D)

    # Sinumerik program prefix/postfix.
    # Channel 1 (robot_C) loads and starts channel 2 (robot_D) via INIT/START so
    # both programs run simultaneously.  Both programs share robot_C's base frame
    # as the coordinate origin — no per-robot coordinate transform is needed.
    # Flange-only moves: NO tool correction may be active.  A selected tool
    # (e.g. FIXTURE, 280 mm) shifts every block by its length — the controller
    # computes with the SELECTED tool, not the mounted one.
    _d_name = os.path.basename(OUT_D).upper()
    PREFIX_C = (
        f'R0=1\n'
        f'INIT(2,"{_d_name}")\n'
        f'START(2)\n'
        f'T0 D0\n'
    )
    PREFIX_D = 'R0=1\nT0 D0\n'
    POSTFIX  = 'STOPRE\nR0=0\nM30\n'

    write_g_code(wps_C, OUT_C, prefix=PREFIX_C, postfix=POSTFIX)
    write_g_code(wps_D, OUT_D, prefix=PREFIX_D, postfix=POSTFIX)

    print()
    print(f'Written: {OUT_C}  ({len(wps_C)} lines)')
    print(f'Written: {OUT_D}  ({len(wps_D)} lines)')
    print()
    print('--- Preview robot_C.mpf ---')
    with open(OUT_C) as f:
        for line in f.readlines()[:12]:
            print(' ', line, end='')
    print('...')
    print()
    print('--- Preview robot_D.mpf ---')
    with open(OUT_D) as f:
        for line in f.readlines()[:12]:
            print(' ', line, end='')
    print('...')
    print()
    print('DONE. Verify target positions, then transfer .mpf files to the controller.')


if __name__ == '__main__':
    main()
