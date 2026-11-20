"""Conservative straight-pull screening, in product-model millimetres.

This is part geometry only, not robot motion planning. Rays and proximity
are rejection heuristics and can discard otherwise feasible schedules.
"""

from dataclasses import dataclass
from itertools import combinations

import fcl
import numpy as np
from diss_assembly_dataclasses.part import FCLTriMesh
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class ParallelGeometrySettings:
    min_separation: float = 50.0
    ray_clearance: float = 1.0
    step_size: float = 2.0
    stationary_penetration: float = 2.5
    max_samples: int = 10000

    def __post_init__(self):
        for name in ("min_separation", "ray_clearance", "stationary_penetration"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be finite and positive")
        if (
            isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or self.max_samples < 2
        ):
            raise ValueError("max_samples must be an integer >= 2")


def _ray_distance(a, da, b, db):
    """Minimum distance between finite centre-ray segments (not infinite lines)."""
    matrix = np.column_stack((da, -db))
    solution = lsq_linear(matrix, b - a, bounds=(0, 1), tol=1e-10)
    return float(np.linalg.norm(a - b + matrix @ solution.x))


def _collision(a, b, tolerance):
    result = fcl.CollisionResult()
    hit = fcl.collide(
        a, b, fcl.CollisionRequest(enable_contact=True, num_max_contacts=100), result
    )
    if not hit:
        return False
    if tolerance == 0 or not result.contacts:
        return True
    return any(
        not np.isfinite(c.penetration_depth) or c.penetration_depth > tolerance
        for c in result.contacts
    )


def check_parallel_geometry(assembly, operations, settings):
    """Return (accepted, diagnostics); never modify assembly meshes or FCL caches.

    Paths follow recorded directions, extending by the assembly bounding-box
    diagonal plus the part diagonal so each part clears the initial assembly.
    Constant-speed pulls share a start time and use action_time as duration.
    Completed parts leave the scene. No geometry is silently skipped.
    """
    try:
        return _check(assembly, operations, settings)
    except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as exc:
        return False, {"reason": "invalid_parallel_geometry", "error": str(exc)}


def _check(assembly, operations, settings):
    objects, vertices = {}, {}
    for part in assembly.parts.values():
        mesh = part.mesh
        v = np.asarray(mesh.vertices, dtype=float)
        t = np.asarray(mesh.triangles, dtype=np.int32)
        if (
            v.ndim != 2
            or v.shape[1] != 3
            or len(v) < 3
            or not np.isfinite(v).all()
            or t.ndim != 2
            or t.shape[1] != 3
            or len(t) == 0
        ):
            raise ValueError(f"missing or invalid mesh for {part.name}")
        if np.any(t < 0) or np.any(t >= len(v)):
            raise ValueError(f"invalid triangle indices for {part.name}")
        vertices[part.name] = v
        objects[part.name] = FCLTriMesh(v, t).make_object()
    all_vertices = np.concatenate(list(vertices.values()))
    diagonal = float(np.linalg.norm(np.ptp(all_vertices, axis=0)))
    paths = {}
    for attrs in operations:
        name = attrs["target_part"]
        direction = np.asarray(attrs["direction"], dtype=float)
        duration = float(attrs["action_time"])
        if (
            direction.shape != (3,)
            or not np.isfinite(direction).all()
            or np.linalg.norm(direction) < 1e-9
            or not np.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(f"invalid direction or duration for {name}")
        length = (
            diagonal
            + float(np.linalg.norm(np.ptp(vertices[name], axis=0)))
            + settings.step_size
        )
        paths[name] = (direction / np.linalg.norm(direction) * length, duration)
    pairs = list(combinations(paths, 2))
    for a, b in pairs:
        distance = fcl.distance(
            objects[a], objects[b], fcl.DistanceRequest(), fcl.DistanceResult()
        )
        if not np.isfinite(distance) or distance <= settings.min_separation:
            return False, {
                "reason": "parts_too_close",
                "parts": [a, b],
                "distance": float(distance),
            }
    for a, b in pairs:
        ca, cb = vertices[a].mean(axis=0), vertices[b].mean(axis=0)
        if _ray_distance(ca, paths[a][0], cb, paths[b][0]) <= settings.ray_clearance:
            return False, {"reason": "removal_rays_intersect", "parts": [a, b]}
    stage_duration = max(duration for _, duration in paths.values())
    # Bound relative displacement by step_size between shared samples.
    speed = max(np.linalg.norm(delta) / duration for delta, duration in paths.values())
    count = max(1, int(np.ceil(2 * speed * stage_duration / settings.step_size)))
    if count + len(paths) + 1 > settings.max_samples:
        return False, {"reason": "parallel_sample_limit"}
    times = sorted(
        set(np.linspace(0, stage_duration, count + 1)) | {d for _, d in paths.values()}
    )
    stationary = set(objects) - set(paths)
    for time in times:
        active = [name for name, (_, duration) in paths.items() if time <= duration]
        for name in active:
            delta, duration = paths[name]
            objects[name].setTransform(
                fcl.Transform(np.eye(3), delta * (time / duration))
            )
            if hasattr(objects[name], "computeAABB"):
                objects[name].computeAABB()
        for a, b in combinations(active, 2):
            if _collision(objects[a], objects[b], 0):
                return False, {
                    "reason": "moving_parts_collision",
                    "parts": [a, b],
                    "time": float(time),
                }
        # Initial contact with the assembly is expected; check after movement.
        if time > 0:
            for a in active:
                for b in stationary:
                    if _collision(
                        objects[a], objects[b], settings.stationary_penetration
                    ):
                        return False, {
                            "reason": "stationary_part_collision",
                            "parts": [a, b],
                            "time": float(time),
                        }
    return True, {
        "reason": None,
        "samples": len(times),
        "path_model": "straight_clearance_pull",
        "min_separation": settings.min_separation,
        "ray_clearance": settings.ray_clearance,
        "step_size": settings.step_size,
        "stationary_penetration": settings.stationary_penetration,
    }
