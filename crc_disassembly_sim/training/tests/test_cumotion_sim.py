"""Integration test: cuMotion plan_multi + PyBullet DIRECT execution.

Mirrors the _execute_move_both path from DisassemblySim but runs headless
(p.DIRECT) without OPC-UA or assembly geometry.

Usage:
    source ~/robotics_env/schedulestream_env/bin/activate
    cd ~/robotics_env/crc_disassembly_sim-main
    python test_cumotion_sim.py
"""

import os
import sys
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_industrial as pi
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modular_planner import PlannerType, CuMotionWaypointPlanner
from modular_planner.scene import ObstacleBox
from methodes.g_code_processor import GCodeProcessor

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


# ── helpers ──────────────────────────────────────────────────────────────────

def get_eef_4x4(robot) -> np.ndarray:
    pos, orn_q = robot.get_endeffector_pose()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(orn_q).as_matrix()
    T[:3, 3] = np.array(pos, dtype=np.float64)
    return T

def pose_to_4x4(pos, orn_q) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(orn_q).as_matrix()
    T[:3, 3] = np.array(pos, dtype=np.float64)
    return T

def execute_waypoints(robot, waypoints, step_sims=5):
    """Drive robot through G-code waypoints (no GCodeProcessor dependency)."""
    for wp in waypoints:
        pos = [wp['X'], wp['Y'], wp['Z']]
        orn = p.getQuaternionFromEuler([wp['A'], wp['B'], wp['C']])
        robot.set_endeffector_pose(pos, orn)
        for _ in range(step_sims):
            p.stepSimulation()


# ── PyBullet setup ────────────────────────────────────────────────────────────

print('Connecting PyBullet DIRECT …')
client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

urdf_C = os.path.join(WORKING_DIR, 'robot_descriptions', 'comau_nj290_robotNC.urdf')
urdf_D = os.path.join(WORKING_DIR, 'robot_descriptions', 'comau_nj290_robotNC_sensor.urdf')

start_pos_C = np.array([0.0, 0.0, 0.0])
start_orn_C = p.getQuaternionFromEuler([0, 0, 0])
robot_C = pi.RobotBase(urdf_C, start_pos_C, start_orn_C)

start_pos_D = np.array([-1.99, -3.750, 0.0])  # measured at cell 02./03.09.2026; was -3.5153162 (235 mm error)
start_orn_D = p.getQuaternionFromEuler([0, 0, np.pi / 2])
robot_D = pi.RobotBase(urdf_D, start_pos_D, start_orn_D)

# Move both robots to a safe starting pose (matches DisassemblySim __init__)
robot_C.set_joint_position({'q2': np.deg2rad(0.0), 'q3': np.deg2rad(-140.0), 'q5': np.deg2rad(45.0)})
robot_D.set_joint_position({'q2': np.deg2rad(0.0), 'q3': np.deg2rad(-140.0), 'q5': np.deg2rad(45.0)})
for _ in range(200):
    p.stepSimulation()

pos_C, orn_C = robot_C.get_endeffector_pose()
pos_D, orn_D = robot_D.get_endeffector_pose()
print(f'Robot C EEF start: pos={np.round(pos_C,3)}, orn(euler)={np.round(p.getEulerFromQuaternion(orn_C),3)}')
print(f'Robot D EEF start: pos={np.round(pos_D,3)}, orn(euler)={np.round(p.getEulerFromQuaternion(orn_D),3)}')


# ── cuMotion planner ──────────────────────────────────────────────────────────

print('\nInstantiating CuMotionWaypointPlanner …')
planner = CuMotionWaypointPlanner(replan_interval_steps=5, max_replan_cycles=20)


# ── define targets ────────────────────────────────────────────────────────────
# robot_C moves 0.2 m in -Y (hold robofixture, small shift)
# robot_D moves 0.3 m in +X (approach assembly)

target_pos_C = np.array(pos_C) + np.array([0.0, -0.2, 0.0])
target_orn_q_C = orn_C

target_pos_D = np.array(pos_D) + np.array([0.3, 0.0, 0.0])
target_orn_q_D = orn_D

current_pose_C = get_eef_4x4(robot_C)
current_pose_D = get_eef_4x4(robot_D)
target_pose_C  = pose_to_4x4(target_pos_C,  target_orn_q_C)
target_pose_D  = pose_to_4x4(target_pos_D,  target_orn_q_D)

print(f'Target C: {np.round(target_pos_C,3)}')
print(f'Target D: {np.round(target_pos_D,3)}')


# ── robot base transforms (world pose of each robot's base) ──────────────────
# cuMotion plans in robot-local frame; these transforms map local ↔ world.

base_pose_C = np.eye(4, dtype=np.float64)
base_pose_C[:3, 3] = start_pos_C

base_pose_D = np.eye(4, dtype=np.float64)
base_pose_D[:3, :3] = R.from_euler('z', np.pi / 2).as_matrix()
base_pose_D[:3, 3] = start_pos_D


# ── AABB getters (live from PyBullet) ─────────────────────────────────────────

def _aabb_C():
    return p.getAABB(robot_C.urdf)

def _aabb_D():
    return p.getAABB(robot_D.urdf)


# ── plan_multi ────────────────────────────────────────────────────────────────

print('\nCalling plan_multi …')
waypoints_list = planner.plan_multi(
    current_eef_poses=[current_pose_C, current_pose_D],
    target_eef_poses=[target_pose_C,   target_pose_D],
    obstacles=[],
    constraints=[],
    robot_names=['robot_C', 'robot_D'],
    robot_aabb_getters=[_aabb_C, _aabb_D],
    robot_base_poses=[base_pose_C, base_pose_D],
)

waypoints_C = waypoints_list[0]
waypoints_D = waypoints_list[1]

print(f'\nPlanning result:')
print(f'  robot_C: {len(waypoints_C)} waypoints')
print(f'  robot_D: {len(waypoints_D)} waypoints')

if not waypoints_C and not waypoints_D:
    print('ERROR: no waypoints returned for either robot.')
    p.disconnect()
    sys.exit(1)


# ── interleaved execution (mirrors _execute_move_both) ────────────────────────

print('\nExecuting waypoints (interleaved) …')
max_wps = max(len(waypoints_C), len(waypoints_D))
step_sims = 5

for i in range(max_wps):
    if i < len(waypoints_C):
        wp = waypoints_C[i]
        robot_C.set_endeffector_pose(
            [wp['X'], wp['Y'], wp['Z']],
            p.getQuaternionFromEuler([wp['A'], wp['B'], wp['C']]))
    if i < len(waypoints_D):
        wp = waypoints_D[i]
        robot_D.set_endeffector_pose(
            [wp['X'], wp['Y'], wp['Z']],
            p.getQuaternionFromEuler([wp['A'], wp['B'], wp['C']]),
            endeffector_name='tcp_frame')
    for _ in range(step_sims):
        p.stepSimulation()


# Give the IK controllers time to fully settle before measuring.
for _ in range(200):
    p.stepSimulation()

# ── verify final positions ────────────────────────────────────────────────────

final_pos_C, _ = robot_C.get_endeffector_pose()
final_pos_D, _ = robot_D.get_endeffector_pose()
err_C = np.linalg.norm(np.array(final_pos_C) - target_pos_C)
err_D = np.linalg.norm(np.array(final_pos_D) - target_pos_D)

print(f'\nFinal positions:')
print(f'  robot_C: {np.round(final_pos_C,4)}  (target {np.round(target_pos_C,4)})  error={err_C*1000:.2f} mm')
print(f'  robot_D: {np.round(final_pos_D,4)}  (target {np.round(target_pos_D,4)})  error={err_D*1000:.2f} mm')

tol = 0.02  # 20 mm tolerance — PyBullet IK isn't sub-mm
ok_C = err_C < tol
ok_D = err_D < tol
print(f'\n{"PASS" if ok_C else "FAIL"} robot_C (err {err_C*1000:.1f} mm < {tol*1000:.0f} mm)')
print(f'{"PASS" if ok_D else "FAIL"} robot_D (err {err_D*1000:.1f} mm < {tol*1000:.0f} mm)')

p.disconnect()
sys.exit(0 if (ok_C and ok_D) else 1)
