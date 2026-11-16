"""Desk markers that pick one of the desk hypotheses (yaw and lattice shift).

Markers never estimate the desk pose; the hole grid stays the pose authority.
Two kinds: ArUco markers (DICT_4X4_50) and Hao's white corner shapes
(``platform_shape_markers.py``). Each declares where it sits on the desk top, in
desk-frame x/y; the hypothesis that puts every seen marker there is chosen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from perception.cloud import OrganizedCloud, Pinhole, fit_pinhole, invert
from perception.system28 import TOP_Z_M, DeskHypotheses


@dataclass(frozen=True)
class MarkerChoice:
    """``index`` into the hypotheses, or None when the markers do not single one out."""

    index: int | None
    seen: dict[
        str, tuple[float, float, float]
    ]  # marker name -> centre in the camera frame
    errors_m: list[float | None] = field(
        default_factory=list
    )  # worst error per hypothesis


def choose(
    hypotheses: DeskHypotheses,
    seen: dict[str, np.ndarray],
    declared_xy: dict[str, tuple[float, float]],
    *,
    max_error_m: float = 0.03,
    min_markers: int = 2,
) -> MarkerChoice:
    """The unique hypothesis that puts every seen marker within ``max_error_m`` of its declared place."""
    errors: list[float | None] = []
    for pose in hypotheses.poses:
        inverse = invert(pose)
        worst = [
            float(
                np.linalg.norm(
                    (inverse[:3, :3] @ point + inverse[:3, 3])[:2]
                    - np.asarray(declared_xy[name])
                )
            )
            for name, point in seen.items()
        ]
        errors.append(max(worst) if worst and len(worst) >= min_markers else None)
    fitting = [i for i, e in enumerate(errors) if e is not None and e <= max_error_m]
    return MarkerChoice(
        fitting[0] if len(fitting) == 1 else None,
        {name: tuple(map(float, point)) for name, point in seen.items()},
        errors,
    )


def _pixel_on_top(
    pixel: np.ndarray, pinhole: Pinhole, desk_pose: np.ndarray
) -> np.ndarray:
    """The point where the pixel's ray meets the desk top plane (all hypotheses share it)."""
    ray = np.array(
        [
            (pixel[0] - pinhole.cx) / pinhole.fx,
            (pixel[1] - pinhole.cy) / pinhole.fy,
            1.0,
        ]
    )
    normal = desk_pose[:3, 2]
    point_on_top = desk_pose[:3, :3] @ np.array([0.0, 0.0, TOP_Z_M]) + desk_pose[:3, 3]
    return ray * float(normal @ point_on_top) / float(normal @ ray)


def choose_by_aruco(
    cloud: OrganizedCloud,
    hypotheses: DeskHypotheses,
    declared_xy: dict[int, tuple[float, float]],
    *,
    pinhole: Pinhole | None = None,
    max_error_m: float = 0.03,
    min_markers: int = 2,
) -> MarkerChoice:
    """Detect DICT_4X4_50 markers in the colour image; ``declared_xy`` maps marker id to desk x/y."""
    import cv2

    if cloud.rgb is None:
        return MarkerChoice(None, {})
    pinhole = pinhole or fit_pinhole(cloud)
    grey = cv2.cvtColor(np.asarray(cloud.rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(grey)
    seen = {}
    for marker_corners, marker_id in zip(corners, [] if ids is None else ids.ravel()):
        if int(marker_id) in declared_xy:
            centre = np.asarray(marker_corners).reshape(-1, 2).mean(axis=0)
            seen[str(int(marker_id))] = _pixel_on_top(
                centre, pinhole, hypotheses.poses[0]
            )
    declared = {str(k): v for k, v in declared_xy.items()}
    return choose(
        hypotheses, seen, declared, max_error_m=max_error_m, min_markers=min_markers
    )


@dataclass(frozen=True)
class DeclaredShape:
    """A white shape on the desk top: where it sits, and how large and round it is."""

    desk_xy_m: tuple[float, float]
    area_m2: float
    circularity: (
        float  # 4 pi A / P^2 in the image: about 0.9 for a disc, 0.785 for a square
    )
    max_score: float = 2.0


@dataclass(frozen=True)
class ShapeProfile:
    max_saturation: int = 40
    min_value: int = 200
    max_height_error_m: float = 0.010
    min_pixels: int = 20
    max_pixels: int = 50_000
    area_weight: float = 1.0
    circularity_weight: float = 2.0


def choose_by_shapes(
    cloud: OrganizedCloud,
    hypotheses: DeskHypotheses,
    shapes: dict[str, DeclaredShape],
    *,
    profile: ShapeProfile = ShapeProfile(),
    max_error_m: float = 0.03,
    min_markers: int = 2,
) -> MarkerChoice:
    """Hao's white-shape selector: white blobs on the desk top, assigned to the declared
    shapes by area and circularity, then scored against each hypothesis."""
    import cv2

    if cloud.rgb is None:
        return MarkerChoice(None, {})
    xyz = cloud.xyz_m
    hsv = cv2.cvtColor(np.asarray(cloud.rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    base = invert(hypotheses.poses[0])
    height = np.nan_to_num(xyz, nan=1e3) @ base[2, :3] + base[2, 3]
    mask = (
        (hsv[:, :, 1] <= profile.max_saturation)
        & (hsv[:, :, 2] >= profile.min_value)
        & (np.abs(height - TOP_Z_M) <= profile.max_height_error_m)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pinhole = fit_pinhole(cloud)
    blobs = []
    for label in range(1, count):
        if (
            not profile.min_pixels
            <= stats[label, cv2.CC_STAT_AREA]
            <= profile.max_pixels
        ):
            continue
        component = labels == label
        points = xyz[component & cloud.valid]
        if len(points) < 3:
            continue
        contour = max(
            cv2.findContours(
                component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0],
            key=cv2.contourArea,
        )
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * math.pi * cv2.contourArea(contour) / max(perimeter**2, 1.0)
        # metric area: pixel count times the footprint of one pixel at its depth
        depth = float(np.median(points[:, 2]))
        area = int(component.sum()) * depth**2 / (abs(pinhole.fx) * abs(pinhole.fy))
        blobs.append((np.median(points, axis=0), area, circularity))

    seen, used = {}, set()
    for name, shape in shapes.items():
        scored = [
            (
                profile.area_weight * abs(math.log(max(area, 1e-9) / shape.area_m2))
                + profile.circularity_weight * abs(circularity - shape.circularity),
                index,
            )
            for index, (_, area, circularity) in enumerate(blobs)
            if index not in used
        ]
        if scored:
            score, index = min(scored)
            if score <= shape.max_score:
                used.add(index)
                seen[name] = blobs[index][0]
    declared = {name: shape.desk_xy_m for name, shape in shapes.items()}
    return choose(
        hypotheses, seen, declared, max_error_m=max_error_m, min_markers=min_markers
    )
