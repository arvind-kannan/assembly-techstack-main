"""The System 28 clamping desk (1 200 x 1 200 mm, 265 holes) and its localisation.

The desk frame is the frame of the sim mesh
``crc_disassembly_sim/fixtures/Spanntisch/spanntisch_low_res.stl`` (in mm) and of
Hao's ``platform_root``: x and y along the hole rows, z up, top face at z = 0.300 m.
``localize_desk`` is Hao's ``localize_system28``: Hough circles on a hole image,
rays to the table plane, a grid fit, and four 90-degree yaw hypotheses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from perception.cloud import (
    OrganizedCloud,
    Pinhole,
    fit_pinhole,
    plane_distance,
    pose_difference,
)
from perception.registrar import MeshModel

DESK_MESH = (
    Path(__file__).resolve().parents[2]
    / "crc_disassembly_sim"
    / "fixtures"
    / "Spanntisch"
    / "spanntisch_low_res.stl"
)
TOP_Z_M = 0.300
TOP_BOUNDS_XY_M = ((-0.095, -0.095), (1.105, 1.105))
SUPPORT_CENTER_M = (0.505, 0.505, TOP_Z_M)
HOLE_DIAMETER_M = 0.028
HOLE_PITCH_M = 0.100
LATTICE_STEP_M = 0.050  # base and diagonal grid together repeat every (50, 50) mm


def desk_model() -> MeshModel:
    return MeshModel.from_file(DESK_MESH, name="system28_desk")


def hole_catalogue() -> tuple[np.ndarray, list[str], list[str]]:
    """The 265 hole centres in the desk frame: a 12 x 12 base grid at 100 mm and an
    11 x 11 diagonal grid between its holes. Returns (centres, ids, pattern types)."""
    base = HOLE_PITCH_M * np.arange(12) - 0.045
    diagonal = HOLE_PITCH_M * np.arange(11) + 0.005
    centres, ids, patterns = [], [], []
    for pattern, axis in (("base_grid", base), ("diagonal", diagonal)):
        for i, x in enumerate(axis):
            for j, y in enumerate(axis):
                centres.append((x, y, TOP_Z_M))
                ids.append(f"{pattern}_{i:02d}_{j:02d}")
                patterns.append(pattern)
    return np.round(np.asarray(centres), 6), ids, patterns


class DeskLocalizationError(RuntimeError):
    """The observation does not show enough of the desk to localise it."""


@dataclass(frozen=True)
class HoleDetection:
    """Hough parameters from metric hole sizes and the current table depth (Hao's ``depth_adaptive``)."""

    min_radius_m: float = 0.010
    max_radius_m: float = 0.016
    min_center_distance_m: float = 0.060
    hough_dp: float = 1.2
    edge_threshold: float = 90.0
    accumulator_threshold: float = 15.0
    table_band_m: float = 0.003
    depth_percentiles: tuple[float, float] = (2.0, 98.0)
    gaussian_kernel_radius_ratio: float = 0.35
    gaussian_sigma_radius_ratio: float = 0.10


@dataclass(frozen=True)
class GridFit:
    """Hao's ``grid_fit`` profile."""

    nominal_pair_min_m: float = 0.092
    nominal_pair_max_m: float = 0.108
    initial_match_threshold_m: float = 0.020
    final_match_threshold_m: float = 0.012
    center_search_half_extent_m: float = 0.100
    center_search_step_m: float = 0.002
    axis_lateral_tolerance_m: float = 0.008
    axis_pitch_tolerance_m: float = 0.012
    lattice_shift_steps: int = (
        4  # shifts of up to 4 x 50 mm checked against the desk edges
    )


@dataclass
class DeskHypotheses:
    """``T_camera_desk`` poses the observation cannot tell apart: four 90-degree yaws
    (square hole pattern) times the lattice shifts that agree with the seen desk edges."""

    poses: list[np.ndarray]
    yaw_deg: list[int]
    shift_m: list[tuple[float, float]]
    matched_holes: int
    residual_rmse_m: float
    detected_circles: int


@dataclass(frozen=True)
class DeskGate:
    """Accept a grid fit only with enough matched holes and a small hole residual."""

    min_matched_holes: int = 20
    max_residual_rmse_m: float = 0.004

    def passes(self, hypotheses: DeskHypotheses) -> bool:
        return (
            hypotheses.matched_holes >= self.min_matched_holes
            and hypotheses.residual_rmse_m <= self.max_residual_rmse_m
        )


def hole_image(cloud: OrganizedCloud, normal, offset, band_m: float) -> np.ndarray:
    """Grey image of the table: on-plane pixels bright, everything else (holes) dark.

    Works on sim renders without colour as well as on Zivid captures."""
    distance = plane_distance(cloud, normal, offset)
    return np.where(distance <= band_m, 255, 0).astype(np.uint8)


def localize_desk(
    cloud: OrganizedCloud,
    normal: np.ndarray,
    offset: float,
    *,
    pinhole: Pinhole | None = None,
    image: np.ndarray | None = None,
    detection: HoleDetection = HoleDetection(),
    grid: GridFit = GridFit(),
) -> DeskHypotheses:
    """Localise the desk from its holes; ``normal``/``offset`` is the table top plane
    with the normal towards the camera (``cloud.fit_table_plane``)."""
    normal = np.asarray(normal, dtype=np.float64)
    pinhole = pinhole or fit_pinhole(cloud)
    if image is None:
        image = hole_image(cloud, normal, offset, detection.table_band_m)
    circles = _hough(cloud, normal, offset, pinhole, image, detection)
    rays = np.column_stack(
        [
            (circles[:, 0] - pinhole.cx) / pinhole.fx,
            (circles[:, 1] - pinhole.cy) / pinhole.fy,
            np.ones(len(circles)),
        ]
    )
    denominator = rays @ normal
    keep = np.abs(denominator) > 1e-9
    centres_camera = rays[keep] * (-offset / denominator[keep])[:, None]
    u, v = _plane_basis(normal)
    observed = np.column_stack([centres_camera @ u, centres_camera @ v])

    model_holes, _, _ = hole_catalogue()
    support = np.asarray(SUPPORT_CENTER_M)
    model = model_holes[:, :2] - support[:2]
    rotation, centre = _grid_axis_and_centre(observed, model, grid)
    rows = cols = accepted = None
    for iteration in range(5):
        predicted = model @ rotation.T + centre
        cost = np.linalg.norm(observed[:, None, :] - predicted[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        threshold = (
            grid.initial_match_threshold_m
            if iteration < 3
            else grid.final_match_threshold_m
        )
        accepted = cost[rows, cols] < threshold
        if int(accepted.sum()) < 4:
            raise DeskLocalizationError("too few holes match the System 28 grid")
        rotation, centre = _rigid_fit_2d(
            model[cols[accepted]], observed[rows[accepted]]
        )
    residuals = np.linalg.norm(
        observed[rows[accepted]] - (model[cols[accepted]] @ rotation.T + centre), axis=1
    )

    x_axis = u * rotation[0, 0] + v * rotation[1, 0]
    y_axis = u * rotation[0, 1] + v * rotation[1, 1]
    base = np.eye(4)
    base[:3, :3] = np.column_stack([x_axis, y_axis, normal])
    top_centre = u * centre[0] + v * centre[1] - offset * normal
    base[:3, 3] = top_centre - base[:3, :3] @ support

    # The hole lattice repeats every (50, 50) mm, so a partial view fits shifted grids
    # as well; only the desk edges, where seen, tell them apart.
    table = cloud.points(
        plane_distance(cloud, normal, offset) <= detection.table_band_m
    )
    table = table[:: max(1, len(table) // 20000)]
    poses, yaws, shifts, inside = [], [], [], []
    steps = range(-grid.lattice_shift_steps, grid.lattice_shift_steps + 1)
    for i in steps:
        for j in steps:
            if (i + j) % 2:
                continue
            shifted = base.copy()
            shifted[:3, 3] += base[:3, :3] @ (LATTICE_STEP_M * np.array([i, j, 0.0]))
            fraction = _inside_top(shifted, table)
            for yaw in (0, 90, 180, 270):
                poses.append(shifted @ _yaw_about(support, math.radians(yaw)))
                yaws.append(yaw)
                shifts.append((LATTICE_STEP_M * i, LATTICE_STEP_M * j))
                inside.append(fraction)
    best = max(inside)
    keep = [k for k, fraction in enumerate(inside) if fraction >= best - 0.005]
    return DeskHypotheses(
        [poses[k] for k in keep],
        [yaws[k] for k in keep],
        [shifts[k] for k in keep],
        int(accepted.sum()),
        float(np.sqrt(np.mean(residuals**2))),
        int(len(circles)),
    )


def _inside_top(pose: np.ndarray, table_camera: np.ndarray) -> float:
    """Fraction of the observed table points that lie on the desk top at this pose."""
    rotation, translation = pose[:3, :3], pose[:3, 3]
    xy = ((table_camera - translation) @ rotation)[:, :2]
    (x0, y0), (x1, y1) = TOP_BOUNDS_XY_M
    margin = 0.005
    return float(
        np.mean(
            (xy[:, 0] >= x0 - margin)
            & (xy[:, 0] <= x1 + margin)
            & (xy[:, 1] >= y0 - margin)
            & (xy[:, 1] <= y1 + margin)
        )
    )


def select_nearest_to_nominal(
    hypotheses: DeskHypotheses, nominal: np.ndarray
) -> tuple[int, np.ndarray]:
    """Resolve the ambiguity without markers: the hypothesis closest to the nominal pose
    (1 deg weighs as 10 mm). Right only while the nominal pose is within half a lattice
    step (35 mm) and 45 deg of the real one."""
    distances = [
        translation + 0.01 * rotation
        for translation, rotation in (
            pose_difference(pose, nominal) for pose in hypotheses.poses
        )
    ]
    index = int(np.argmin(distances))
    return index, hypotheses.poses[index]


def _hough(cloud, normal, offset, pinhole, image, detection) -> np.ndarray:
    distance = plane_distance(cloud, normal, offset)
    depths = cloud.xyz_m[:, :, 2][distance <= detection.table_band_m]
    if len(depths) < 1000:
        raise DeskLocalizationError("too few table points")
    near, far = np.percentile(depths, detection.depth_percentiles)
    focal = (abs(pinhole.fx) + abs(pinhole.fy)) / 2.0
    min_radius = max(2, int(math.floor(focal * detection.min_radius_m / far)))
    max_radius = int(math.ceil(focal * detection.max_radius_m / near))
    min_distance = max(
        4, int(math.floor(focal * detection.min_center_distance_m / far))
    )
    kernel = max(3, int(round(min_radius * detection.gaussian_kernel_radius_ratio)) | 1)
    sigma = max(0.5, min_radius * detection.gaussian_sigma_radius_ratio)
    grey = cv2.GaussianBlur(image, (kernel, kernel), sigma)
    found = cv2.HoughCircles(
        grey,
        cv2.HOUGH_GRADIENT,
        dp=detection.hough_dp,
        minDist=float(min_distance),
        param1=detection.edge_threshold,
        param2=detection.accumulator_threshold,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if found is None:
        raise DeskLocalizationError("no System 28 holes detected")
    return np.asarray(found[0][:, :2], dtype=np.float64)


def _grid_axis_and_centre(observed, model, grid: GridFit):
    vectors = observed[:, None, :] - observed[None, :, :]
    distances = np.linalg.norm(vectors, axis=2)
    left, right = np.where(
        (distances > grid.nominal_pair_min_m) & (distances < grid.nominal_pair_max_m)
    )
    pairs = vectors[left[left < right], right[left < right]]
    if len(pairs) < 20:
        raise DeskLocalizationError("too few hole pairs at the 100 mm pitch")
    angles = np.mod(np.arctan2(pairs[:, 1], pairs[:, 0]), np.pi / 2.0)
    theta = float((np.angle(np.mean(np.exp(4j * angles))) / 4.0) % (np.pi / 2.0))
    for _ in range(3):
        axis = np.array([math.cos(theta), math.sin(theta)])
        parallel, lateral = pairs @ axis, pairs @ np.array([-axis[1], axis[0]])
        selected = (np.abs(lateral) < grid.axis_lateral_tolerance_m) & (
            np.abs(np.abs(parallel) - HOLE_PITCH_M) < grid.axis_pitch_tolerance_m
        )
        if int(selected.sum()) < 2:
            raise DeskLocalizationError("System 28 axis support collapsed")
        directions = pairs[selected] / np.linalg.norm(pairs[selected], axis=1)[:, None]
        directions *= np.sign(directions @ axis)[:, None]
        axis = directions.mean(axis=0)
        theta = float(math.atan2(axis[1], axis[0]) % (np.pi / 2.0))
    x_axis = np.array([math.cos(theta), math.sin(theta)])
    rotation = np.column_stack([x_axis, [-x_axis[1], x_axis[0]]])
    aligned = observed @ rotation
    approximate = np.median(aligned, axis=0)
    half, step = grid.center_search_half_extent_m, grid.center_search_step_m
    best, tree = None, cKDTree(model)
    for dx in np.arange(-half, half + step / 2.0, step):
        for dy in np.arange(-half, half + step / 2.0, step):
            centre = approximate + [dx, dy]
            residual, _ = tree.query(aligned - centre)
            inlier = residual < grid.final_match_threshold_m
            score = (
                int(inlier.sum()),
                -float(np.median(residual[inlier])) if inlier.any() else -9.0,
                -float(np.mean(np.minimum(residual, 0.03))),
            )
            if best is None or score > best[0]:
                best = score, centre
    return rotation, rotation @ best[1]


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.eye(3)[int(np.argmin(np.abs(np.eye(3) @ normal)))]
    u = reference - normal * float(reference @ normal)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def _rigid_fit_2d(source: np.ndarray, target: np.ndarray):
    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    left, _, right = np.linalg.svd(
        (source - source_centre).T @ (target - target_centre)
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    return rotation, target_centre - rotation @ source_centre


def _yaw_about(point: np.ndarray, yaw: float) -> np.ndarray:
    around, local, back = np.eye(4), np.eye(4), np.eye(4)
    around[:3, 3], back[:3, 3] = point, -point
    local[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    return around @ local @ back
