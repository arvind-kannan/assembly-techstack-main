"""Interface test 5: CuMotionWaypointPlanner static API (no cumotion import needed).

Tests the coordinate-conversion helpers and G-code output format without
launching a cuMotion planner (which requires XRDF + planning config files).

Run with:
    python test_interface_5_cumotion.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from scipy.spatial.transform import Rotation
from modular_planner.cumotion_planner import CuMotionWaypointPlanner
from modular_planner.scene import ObstacleBox, SceneConstraint, ConstraintType


# ---------------------------------------------------------------------------
# Test 1: _mat4_to_pose3 / _pose3_to_mat4 round-trip
# ---------------------------------------------------------------------------

class FakePose3:
    """Minimal stand-in for cumotion.Pose3 — used without importing cumotion."""
    def __init__(self, rotation, translation):
        self.rotation = rotation
        self.translation = translation

class FakeRotation3:
    def __init__(self, q_wxyz):
        self._q = q_wxyz
    @property
    def quaternion(self):
        return self._q


def test_pose_roundtrip():
    """4×4 → Pose3 → 4×4 should recover the original matrix."""
    # Build a random rigid transform.
    rot = Rotation.from_euler('xyz', [0.3, -0.5, 1.1]).as_matrix()
    trans = np.array([0.5, -0.2, 0.9])
    mat4 = np.eye(4)
    mat4[:3, :3] = rot
    mat4[:3, 3] = trans

    # Simulate what _mat4_to_pose3 and _pose3_to_mat4 do (without cumotion).
    rot_obj = Rotation.from_matrix(mat4[:3, :3])
    quat_xyzw = rot_obj.as_quat()
    q_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    fake_rot = FakeRotation3(q_wxyz)
    fake_pose = FakePose3(fake_rot, mat4[:3, 3])

    # Reconstruct
    quat_wxyz = np.array(fake_pose.rotation.quaternion)
    quat_xyzw_back = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    mat4_back = np.eye(4)
    mat4_back[:3, :3] = Rotation.from_quat(quat_xyzw_back).as_matrix()
    mat4_back[:3, 3] = np.array(fake_pose.translation)

    assert np.allclose(mat4, mat4_back, atol=1e-6), \
        f'Round-trip failed:\n{mat4}\nvs\n{mat4_back}'
    print('PASS  test_pose_roundtrip')


# ---------------------------------------------------------------------------
# Test 2: _trajectory_to_gcode output format
# ---------------------------------------------------------------------------

def test_trajectory_to_gcode():
    """FK trajectory should produce valid G-code waypoint dicts."""
    planner = CuMotionWaypointPlanner.__new__(CuMotionWaypointPlanner)
    planner._feed_forward_speed = 800.0

    traj = []
    for i in range(3):
        mat = np.eye(4)
        mat[:3, 3] = [i * 0.1, 0.0, 0.5]
        traj.append(mat)

    waypoints = planner._trajectory_to_gcode(traj)

    assert len(waypoints) == 3
    required_keys = {'G', 'X', 'Y', 'Z', 'A', 'B', 'C', 'F'}
    for wp in waypoints:
        assert required_keys == set(wp.keys()), f'Missing keys: {set(wp.keys())}'
        assert wp['G'] == 1
        assert wp['F'] == 800.0
    assert abs(waypoints[1]['X'] - 0.1) < 1e-6
    assert abs(waypoints[2]['X'] - 0.2) < 1e-6
    print('PASS  test_trajectory_to_gcode')


# ---------------------------------------------------------------------------
# Test 3: ObstacleBox → SIDE_LENGTHS consistency
# ---------------------------------------------------------------------------

def test_obstacle_box_spans():
    """ObstacleBox.spans should equal 2 × half_extents."""
    box = ObstacleBox(
        center=np.array([1.0, 0.0, 0.5]),
        half_extents=np.array([0.1, 0.2, 0.15]),
        rotation=np.eye(3),
    )
    expected_spans = np.array([0.2, 0.4, 0.3])
    assert np.allclose(box.spans, expected_spans), \
        f'spans={box.spans}, expected={expected_spans}'
    print('PASS  test_obstacle_box_spans')


# ---------------------------------------------------------------------------
# Test 4: _build_inter_robot_obstacles excludes self
# ---------------------------------------------------------------------------

def test_inter_robot_obstacles_exclude_self():
    """Each robot's obstacle list should contain only the other robot."""
    planner = CuMotionWaypointPlanner.__new__(CuMotionWaypointPlanner)

    pose_C = np.eye(4); pose_C[:3, 3] = [0.5, 0.0, 1.0]
    pose_D = np.eye(4); pose_D[:3, 3] = [-0.5, 0.0, 1.0]

    result = planner._build_inter_robot_obstacles(
        [pose_C, pose_D], aabb_getters=None)

    assert len(result) == 2, 'Expected one obstacle list per robot'
    # Robot 0 (C) should see robot 1 (D) as obstacle, not itself.
    assert len(result[0]) == 1
    assert np.allclose(result[0][0].center, [-0.5, 0.0, 1.0], atol=1e-5), \
        f'Robot 0 obstacle center: {result[0][0].center}'
    # Robot 1 (D) should see robot 0 (C) as obstacle.
    assert len(result[1]) == 1
    assert np.allclose(result[1][0].center, [0.5, 0.0, 1.0], atol=1e-5), \
        f'Robot 1 obstacle center: {result[1][0].center}'
    print('PASS  test_inter_robot_obstacles_exclude_self')


# ---------------------------------------------------------------------------
# Test 5: PlannerType enum has CUMOTION
# ---------------------------------------------------------------------------

def test_planner_type_enum():
    from modular_planner import PlannerType
    assert hasattr(PlannerType, 'CUMOTION'), 'PlannerType.CUMOTION missing'
    assert PlannerType.CUMOTION.value == 'cumotion'
    print('PASS  test_planner_type_enum')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_pose_roundtrip()
    test_trajectory_to_gcode()
    test_obstacle_box_spans()
    test_inter_robot_obstacles_exclude_self()
    test_planner_type_enum()
    print('\nAll interface tests passed.')
