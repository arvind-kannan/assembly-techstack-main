import numpy as np
from modular_planner.scene import ObstacleBox

aabb_min = np.array([0.3, -1.2, 0.5])
aabb_max = np.array([0.7, -0.8, 1.2])

box = ObstacleBox.from_pybullet_aabb(aabb_min, aabb_max)

print("center:     ", box.center)
print("half_extents:", box.half_extents)
print("spans:      ", box.spans)
print("pose_4x4:\n", box.pose_4x4)

assert np.allclose(box.center, (aabb_min + aabb_max) / 2), "center wrong"
assert np.allclose(box.half_extents, (aabb_max - aabb_min) / 2), "half_extents wrong"
assert np.allclose(box.spans, aabb_max - aabb_min), "spans wrong"
assert box.pose_4x4.shape == (4, 4), "pose_4x4 wrong shape"
print("Interface 1 OK")
