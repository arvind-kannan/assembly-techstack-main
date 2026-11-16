import numpy as np
from modular_planner.gnn_planner import GNNWaypointPlanner
from modular_planner.scene import ObstacleBox

planner = object.__new__(GNNWaypointPlanner)
planner._workspace_scale = 1.0

pose_crc = np.eye(4, dtype=np.float32)
pose_crc[:3, 3] = [1.5, -2.0, 0.8]
pose_crc[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)

rb = planner._crc_to_rb(pose_crc)
back = planner._rb_to_crc(rb)

print("Original:\n", pose_crc)
print("After round-trip:\n", back)
assert np.allclose(pose_crc, back, atol=1e-5), \
    f"Round-trip failed:\noriginal:\n{pose_crc}\nafter:\n{back}"
print("Pose round-trip OK")

box = ObstacleBox(center=np.array([1.0, -1.5, 0.5]),
                  half_extents=np.array([0.2, 0.3, 0.4]),
                  rotation=np.eye(3))
rb_box = planner._obstacle_crc_to_rb(box)
print(f"Obstacle center CRC→RB: {box.center} → {rb_box.center}")
assert np.allclose(rb_box.center, box.center * planner._workspace_scale), \
    "obstacle center transform wrong"
print("Interface 4 OK")
