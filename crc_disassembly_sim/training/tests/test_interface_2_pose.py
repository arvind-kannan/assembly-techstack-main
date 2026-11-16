import numpy as np
from scipy.spatial.transform import Rotation as R

pos = [0.5, -1.0, 1.2]
orn_q = [0.0, 0.707, 0.0, 0.707]  # 90 deg around Y, scalar-last

rot = R.from_quat(orn_q)
T = np.eye(4, dtype=np.float64)
T[:3, :3] = rot.as_matrix()
T[:3, 3] = np.array(pos)

print("4x4 matrix:\n", T)
print("Position recovered:", T[:3, 3])
print("Det(R):", np.linalg.det(T[:3, :3]))

assert np.allclose(T[:3, 3], pos), "position lost"
assert np.isclose(np.linalg.det(T[:3, :3]), 1.0), "rotation is not orthogonal"
print("Interface 2 OK")
