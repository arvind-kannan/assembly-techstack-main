# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""IK solver for robot arms using damped least-squares Jacobian pseudoinverse."""

from typing import Sequence

import mujoco
import numpy as np


def _quat_error(q_current: np.ndarray, q_desired: np.ndarray) -> np.ndarray:
  """Returns axis-angle error rotating q_current towards q_desired.

  Both quaternions are scalar-first [w, x, y, z].
  """
  # q_err = q_desired * q_current^{-1}
  # q^{-1} for unit quat = [w, -x, -y, -z]
  w0, x0, y0, z0 = q_current
  w1, x1, y1, z1 = q_desired
  # Hamilton product q1 * conj(q0)
  w = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
  x = x1 * w0 - w1 * x0 + z1 * y0 - y1 * z0
  y = y1 * w0 - z1 * x0 - w1 * y0 + x1 * z0
  z = y1 * x0 + z1 * w0 - x1 * y0 - w1 * z0  # noqa: intentional sign
  # Flip to shortest arc
  if w < 0:
    x, y, z, w = -x, -y, -z, -w
  # Axis-angle: err = 2 * axis * sin(theta/2) ≈ 2 * [x,y,z] for small angles
  return 2.0 * np.array([x, y, z])


class IkSolver:
  """Inverse kinematics solver using damped least-squares Jacobian."""

  def __init__(
      self,
      model: mujoco.MjModel,
      controllable_joints_ids: list[int],
      site_name: str,
      collision_pairs: Sequence[tuple[Sequence[str], Sequence[str]]] | None,
      solver_configs: dict[str, float | bool] | None = None,
  ):
    self._model = model
    self._data = mujoco.MjData(model)
    self._controllable_joints_ids = controllable_joints_ids
    self._site_id = model.site(site_name).id
    self._collision_pairs = collision_pairs or []

    # Pre-compute geom id sets for collision checking
    self._collision_geom_pairs: list[tuple[set[int], set[int]]] = []
    for group_a, group_b in self._collision_pairs:
      ids_a = {model.geom(n).id for n in group_a if self._geom_exists(n)}
      ids_b = {model.geom(n).id for n in group_b if self._geom_exists(n)}
      if ids_a and ids_b:
        self._collision_geom_pairs.append((ids_a, ids_b))

    # Damping factor for damped least-squares (lambda^2)
    self._damping = (solver_configs or {}).get('damping', 1e-4)
    # Step size for joint updates
    self._step_size = (solver_configs or {}).get('step_size', 0.5)

  def _geom_exists(self, name: str) -> bool:
    try:
      self._model.geom(name)
      return True
    except (KeyError, ValueError):
      return False

  def _in_collision(self) -> bool:
    """Returns True if any configured collision pair is in contact."""
    if not self._collision_geom_pairs:
      return False
    mujoco.mj_collision(self._model, self._data)
    for contact in self._data.contact[: self._data.ncon]:
      g1, g2 = contact.geom1, contact.geom2
      for ids_a, ids_b in self._collision_geom_pairs:
        if (g1 in ids_a and g2 in ids_b) or (g2 in ids_a and g1 in ids_b):
          return True
    return False

  def _get_joint_qpos_addrs(self) -> list[int]:
    return [self._model.jnt_qposadr[jid] for jid in self._controllable_joints_ids]

  def _get_joint_dof_addrs(self) -> list[int]:
    return [self._model.jnt_dofadr[jid] for jid in self._controllable_joints_ids]

  def _set_joints(self, q: np.ndarray) -> None:
    addrs = self._get_joint_qpos_addrs()
    for addr, val in zip(addrs, q):
      self._data.qpos[addr] = val
    mujoco.mj_kinematics(self._model, self._data)
    mujoco.mj_comPos(self._model, self._data)

  def _get_site_pose(self) -> tuple[np.ndarray, np.ndarray]:
    pos = self._data.site_xpos[self._site_id].copy()
    mat = self._data.site_xmat[self._site_id].reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.ravel())
    return pos, quat  # quat is scalar-first [w,x,y,z]

  def _get_jacobian(self) -> np.ndarray:
    """Returns 6×n Jacobian (pos rows first, then rot) for controlled joints."""
    nv = self._model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._site_id)
    dof_addrs = self._get_joint_dof_addrs()
    J = np.vstack([jacp[:, dof_addrs], jacr[:, dof_addrs]])  # (6, n)
    return J

  def solve(
      self,
      desired_global_pos_tcp: np.ndarray,
      desired_global_quat_tcp: np.ndarray,
      initial_joint_configuration: np.ndarray,
      linear_tol: float = 1e-3,
      angular_tol: float = 1e-3,
      max_steps: int = 100,
      early_stop: bool = False,
  ) -> tuple[np.ndarray | None, list[np.ndarray]]:
    if len(initial_joint_configuration) != len(self._controllable_joints_ids):
      raise ValueError(
          'The provided initial joint configuration does not have the right'
          f' number of elements expected length of'
          f' {len(self._controllable_joints_ids)}. Got'
          f' {initial_joint_configuration}'
      )

    q = initial_joint_configuration.copy().astype(np.float64)
    self._set_joints(q)

    path: list[np.ndarray] = [q.copy()]
    best_q: np.ndarray | None = None

    for _ in range(max_steps):
      pos, quat = self._get_site_pose()
      pos_err = desired_global_pos_tcp - pos
      rot_err = _quat_error(quat, desired_global_quat_tcp)
      err = np.concatenate([pos_err, rot_err])

      pos_norm = np.linalg.norm(pos_err)
      rot_norm = np.linalg.norm(rot_err)
      converged = pos_norm < linear_tol and rot_norm < angular_tol

      if converged:
        best_q = q.copy()
        if early_stop:
          break

      J = self._get_jacobian()
      # Damped least-squares: dq = J^T (J J^T + λ²I)^{-1} err
      JJT = J @ J.T
      dq = J.T @ np.linalg.solve(
          JJT + self._damping * np.eye(6), err
      )
      q = q + self._step_size * dq

      # Clamp to joint limits
      qpos_addrs = self._get_joint_qpos_addrs()
      for i, (jid, addr) in enumerate(
          zip(self._controllable_joints_ids, qpos_addrs)
      ):
        lo = self._model.jnt_range[jid, 0]
        hi = self._model.jnt_range[jid, 1]
        if self._model.jnt_limited[jid]:
          q[i] = np.clip(q[i], lo, hi)

      self._set_joints(q)
      path.append(q.copy())

    # Final convergence check if early_stop was False
    if best_q is None:
      pos, quat = self._get_site_pose()
      pos_err = desired_global_pos_tcp - pos
      rot_err = _quat_error(quat, desired_global_quat_tcp)
      if (np.linalg.norm(pos_err) < linear_tol
          and np.linalg.norm(rot_err) < angular_tol):
        best_q = q.copy()

    return best_q, path
