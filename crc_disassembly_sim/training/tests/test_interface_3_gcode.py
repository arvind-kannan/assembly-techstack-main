import numpy as np
from modular_planner.gnn_planner import GNNWaypointPlanner

def make_fake_trajectory(start, end, n=10):
    trajectory = []
    for i in range(n):
        t = i / (n - 1)
        pos = (1 - t) * start + t * end
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = pos
        trajectory.append(pose)
    return trajectory

planner = object.__new__(GNNWaypointPlanner)
planner._feed_forward_speed = 1000.0

trajectory = make_fake_trajectory(
    np.array([0.5, -1.0, 1.2]),
    np.array([0.3, -2.0, 0.9]),
    n=8,
)
waypoints = planner._trajectory_to_gcode(trajectory)

print(f"Generated {len(waypoints)} waypoints")
required_keys = {'G', 'X', 'Y', 'Z', 'A', 'B', 'C', 'F'}
for i, wp in enumerate(waypoints):
    missing = required_keys - wp.keys()
    assert not missing, f"Waypoint {i} missing keys: {missing}"
    assert wp['G'] == 1, f"Waypoint {i} has G={wp['G']}, expected G=1"
    assert isinstance(wp['X'], float), f"Waypoint {i} X is not float"

print("Sample waypoints:")
for wp in waypoints[:3]:
    print(" ", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in wp.items()})
print("Interface 3 OK")
