"""Organized point clouds in metres, pinhole fitting and raycast rendering.

Every cloud here is in the camera frame with the Zivid axis convention
(x right, y down, z forward) and in metres. Invalid pixels are NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class Pinhole:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class OrganizedCloud:
    """An H x W x 3 cloud in metres (NaN = no point) with an optional H x W x 3 uint8 colour image."""

    xyz_m: np.ndarray
    rgb: np.ndarray | None = None

    def __post_init__(self):
        self.xyz_m = np.asarray(self.xyz_m, dtype=np.float64)
        if self.xyz_m.ndim != 3 or self.xyz_m.shape[2] != 3:
            raise ValueError(
                f"organized cloud must be H x W x 3, got {self.xyz_m.shape}"
            )
        if (
            self.rgb is not None
            and np.asarray(self.rgb).shape[:2] != self.xyz_m.shape[:2]
        ):
            raise ValueError("rgb image and cloud differ in size")

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.xyz_m).all(axis=2)

    def points(self, mask: np.ndarray | None = None) -> np.ndarray:
        keep = self.valid if mask is None else self.valid & mask
        return self.xyz_m[keep]


def fit_pinhole(cloud: OrganizedCloud) -> Pinhole:
    """Least-squares pinhole from the organized XYZ (Hao's ``fit_projection_pinhole``)."""
    rows, cols = np.nonzero(
        cloud.valid & (np.abs(np.nan_to_num(cloud.xyz_m[:, :, 2])) > 1e-9)
    )
    stride = max(1, len(rows) // 100_000)
    rows, cols = rows[::stride], cols[::stride]
    if len(rows) < 100:
        raise ValueError("too few valid pixels to fit a pinhole")
    points = cloud.xyz_m[rows, cols]
    ax = np.column_stack([points[:, 0] / points[:, 2], np.ones(len(points))])
    ay = np.column_stack([points[:, 1] / points[:, 2], np.ones(len(points))])
    fx, cx = np.linalg.lstsq(ax, cols.astype(np.float64), rcond=None)[0]
    fy, cy = np.linalg.lstsq(ay, rows.astype(np.float64), rcond=None)[0]
    height, width = cloud.xyz_m.shape[:2]
    return Pinhole(float(fx), float(fy), float(cx), float(cy), width, height)


def raycasting_scene(
    meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> o3d.t.geometry.RaycastingScene:
    """A scene of (vertices_m, triangles, T_camera_mesh) meshes."""
    scene = o3d.t.geometry.RaycastingScene()
    for vertices, triangles, transform in meshes:
        placed = transform_points(transform, vertices)
        scene.add_triangles(
            o3d.core.Tensor(placed.astype(np.float32)),
            o3d.core.Tensor(np.asarray(triangles, dtype=np.uint32)),
        )
    return scene


def render(scene: o3d.t.geometry.RaycastingScene, pinhole: Pinhole) -> OrganizedCloud:
    """Raycast a pinhole camera at the origin of the scene frame; no colour."""
    cols, rows = np.meshgrid(np.arange(pinhole.width), np.arange(pinhole.height))
    directions = np.stack(
        [
            (cols - pinhole.cx) / pinhole.fx,
            (rows - pinhole.cy) / pinhole.fy,
            np.ones(cols.shape),
        ],
        axis=2,
    )
    rays = np.concatenate([np.zeros_like(directions), directions], axis=2).astype(
        np.float32
    )
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    t_hit = np.where(np.isfinite(t_hit), t_hit, np.nan)
    xyz = directions * t_hit[:, :, None]
    return OrganizedCloud(xyz)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def invert(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def pose_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(translation in m, rotation in deg) between two poses."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    relative = a[:3, :3] @ b[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(
        np.degrees(np.arccos(cosine))
    )


def fit_table_plane(
    points: np.ndarray, *, distance_m: float = 0.003, seed: int = 0
) -> tuple[np.ndarray, float]:
    """The dominant plane ``n . p + d = 0``, with ``n`` pointing towards the camera at the origin."""
    o3d.utility.random.seed(seed)
    model, _ = to_o3d(points).segment_plane(distance_m, 3, 1000)
    normal, offset = np.asarray(model[:3], dtype=np.float64), float(model[3])
    scale = np.linalg.norm(normal)
    normal, offset = normal / scale, offset / scale
    if offset < 0.0:  # camera origin on the negative side: flip n and d together
        normal, offset = -normal, -offset
    return normal, offset


def above_plane(
    cloud: OrganizedCloud, normal: np.ndarray, offset: float, band_m: float = 0.003
) -> np.ndarray:
    """Pixels more than ``band_m`` in front of the plane (towards the camera): what stands
    on the table (Hao's ``segment_product_above_dominant_plane``, without the ROI)."""
    signed = np.nan_to_num(cloud.xyz_m) @ np.asarray(normal) + offset
    return cloud.valid & (signed > band_m)


def plane_distance(
    cloud: OrganizedCloud, normal: np.ndarray, offset: float
) -> np.ndarray:
    """Per-pixel distance to the plane ``n . p + d = 0``; invalid pixels are far away."""
    return np.abs(np.nan_to_num(cloud.xyz_m, nan=1e3) @ np.asarray(normal) + offset)


def to_o3d(
    points: np.ndarray, normals: np.ndarray | None = None
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    )
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(
            np.asarray(normals, dtype=np.float64)
        )
    return cloud
