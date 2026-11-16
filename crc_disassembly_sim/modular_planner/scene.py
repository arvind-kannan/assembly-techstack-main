"""Scene representation shared by all planners in modular_planner.

ObstacleBox  — an axis-aligned or oriented bounding box extracted from PyBullet.
SceneConstraint — extensible dataclass for hard / soft constraints.

Constraints are kept separate from obstacles so each planner can encode them
in its own way.  Adding a new constraint type only requires:
  1. Adding a new ConstraintType enum value here.
  2. Handling it in the relevant planner (planner.py for GNN,
     cumotion_planner.py for cuMotion).
"""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import Optional

import numpy as np


@dataclasses.dataclass
class ObstacleBox:
    """An oriented bounding box obstacle extracted from the PyBullet scene.

    Attributes:
        center:        3D center position in world frame (metres).
        half_extents:  Half-lengths along each local axis (metres).
        rotation:      3×3 rotation matrix (world_R_box).  Identity = axis-aligned.
    """
    center: np.ndarray        # shape (3,)
    half_extents: np.ndarray  # shape (3,)
    rotation: np.ndarray      # shape (3, 3)

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float32)
        self.half_extents = np.asarray(self.half_extents, dtype=np.float32)
        self.rotation = np.asarray(self.rotation, dtype=np.float32)

    @property
    def pose_4x4(self) -> np.ndarray:
        """4×4 homogeneous transform with box rotation and center as origin."""
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.center
        return T

    @property
    def spans(self) -> np.ndarray:
        """Full side-lengths (2 × half_extents), as expected by cuMotion/GNN."""
        return 2.0 * self.half_extents

    def in_frame(self, T_inv: np.ndarray) -> 'ObstacleBox':
        """Return a copy of this box expressed in a different frame.

        T_inv is the inverse of the target frame's world transform (4×4).
        The box extents are unchanged; only the pose is transformed.
        """
        T_new = T_inv @ self.pose_4x4.astype(np.float64)
        return ObstacleBox(
            center=T_new[:3, 3].astype(np.float32),
            half_extents=self.half_extents.copy(),
            rotation=T_new[:3, :3].astype(np.float32),
        )

    @classmethod
    def from_pybullet_aabb(cls, aabb_min: np.ndarray,
                           aabb_max: np.ndarray) -> 'ObstacleBox':
        """Construct from PyBullet getAABB() output (axis-aligned)."""
        aabb_min = np.asarray(aabb_min, dtype=np.float32)
        aabb_max = np.asarray(aabb_max, dtype=np.float32)
        center = 0.5 * (aabb_min + aabb_max)
        half_extents = 0.5 * (aabb_max - aabb_min)
        return cls(center=center, half_extents=half_extents,
                   rotation=np.eye(3, dtype=np.float32))


class ConstraintType(Enum):
    """Extensible catalogue of constraint types.

    When you add a new type here, also add handling in the relevant planner:
      GNN    — graph-node encoder in planner.py::_constraint_node_features()
      cuMotion — obstacle/config logic in cumotion_planner.py::_build_world()
    """
    NO_GO_ZONE = auto()       # Volume the EEF must never enter.
    FORCE_LIMIT = auto()      # Maximum contact force along an axis.
    ORIENTATION_LOCK = auto() # EEF must maintain a given orientation.
    COLLABORATIVE_ZONE = auto()  # Slow down / stop when human is nearby.
    SEQUENCE_DEPENDENCY = auto() # Target B cannot start before target A is done.


@dataclasses.dataclass
class SceneConstraint:
    """A constraint to be encoded as an extra graph node.

    Attributes:
        constraint_type: The ConstraintType enum value.
        pose:            4×4 world-frame pose of the constraint (centre/frame).
        half_extents:    For volumetric constraints (NO_GO_ZONE, COLLABORATIVE_ZONE).
        axis:            Unit vector for directional constraints (FORCE_LIMIT,
                         ORIENTATION_LOCK).
        scalar_value:    Limit value (force [N], angle [rad], etc.)
        target_name:     For SEQUENCE_DEPENDENCY: name of prerequisite target.
        metadata:        Arbitrary dict for future fields.
    """
    constraint_type: ConstraintType
    pose: np.ndarray = dataclasses.field(
        default_factory=lambda: np.eye(4, dtype=np.float32))
    half_extents: Optional[np.ndarray] = None
    axis: Optional[np.ndarray] = None
    scalar_value: float = 0.0
    target_name: Optional[str] = None
    metadata: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.pose = np.asarray(self.pose, dtype=np.float32)
        if self.half_extents is not None:
            self.half_extents = np.asarray(self.half_extents, dtype=np.float32)
        if self.axis is not None:
            self.axis = np.asarray(self.axis, dtype=np.float32)
