"""Model-to-observation registration, ported from Hao's ``product_registrar.py``.

One registrar serves the product and the clamps: fresh multi-seed FPFH/RANSAC
hypotheses (plus the nominal pose as one more seed, for #45), robust
point-to-plane ICP, then the #45 gate on the model as the camera would see it
at each candidate pose. Hao's output-face and feature-agreement gates are not
ported: acceptance is the legacy ICP gate, the plausibility gate when a nominal
pose is given, and optionally Hao's extra geometry gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from perception.cloud import (
    OrganizedCloud,
    Pinhole,
    fit_pinhole,
    invert,
    pose_difference,
    raycasting_scene,
    render,
    to_o3d,
    transform_points,
)
from perception.quality_gate import ExtraGates, IcpGate, PlausibilityGate

UNIQUE_PASS = "UNIQUE_PASS"
AMBIGUOUS = "AMBIGUOUS"
REJECTED = "REJECTED"


@dataclass(frozen=True)
class MeshModel:
    """A rigid model (product, part or clamp) in metres."""

    name: str
    vertices_m: np.ndarray
    triangles: np.ndarray

    @classmethod
    def from_file(
        cls, path: str | Path, *, scale_to_m: float = 0.001, name: str | None = None
    ) -> "MeshModel":
        """Load a mesh file; ``scale_to_m`` is the file unit in metres (the sim meshes are in mm)."""
        mesh = trimesh.load(str(path), force="mesh")
        return cls(
            name or Path(path).stem,
            np.asarray(mesh.vertices) * scale_to_m,
            np.asarray(mesh.faces),
        )

    def surface(self, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
        """Points and normals sampled about ``spacing_m`` apart on the surface."""
        mesh = trimesh.Trimesh(self.vertices_m, self.triangles, process=False)
        points, faces = trimesh.sample.sample_surface(
            mesh, max(2000, int(mesh.area / spacing_m**2)), seed=0
        )
        return np.asarray(points), np.asarray(mesh.face_normals[faces])


@dataclass(frozen=True)
class RegistrationProfile:
    """Search parameters; values from Hao's ``stage92jc`` registration profile."""

    random_seeds: tuple[int, ...] = (
        920501,
        920502,
        920503,
        920504,
        920505,
        920506,
        920507,
        920508,
    )
    voxel_size_m: float = 0.005
    normal_radius_m: float = 0.012
    normal_max_nn: int = 50
    feature_radius_m: float = 0.030
    feature_max_nn: int = 100
    ransac_distance_threshold_m: float = 0.012
    ransac_n: int = 4
    ransac_edge_length_checker: float = 0.85
    ransac_max_iterations: int = 80000
    ransac_confidence: float = 0.999
    icp_distance_threshold_m: float = 0.010
    icp_max_iterations: int = 120
    icp_tukey_scale_m: float = 0.003
    fine_icp_distance_threshold_m: float = 0.003
    fine_rounds: int = 3
    occlusion_tolerance_m: float = 0.005
    dedup_translation_m: float = 0.015
    dedup_rotation_deg: float = 12.0


@dataclass
class Candidate:
    source: str
    pose: np.ndarray
    gate: dict
    plausibility: dict | None = None
    extra: dict | None = None

    @property
    def passed(self) -> bool:
        return bool(
            self.gate["pass"]
            and (self.plausibility is None or self.plausibility["pass"])
            and (self.extra is None or self.extra["pass"])
        )


@dataclass
class Registration:
    """``pose`` is ``T_camera_model`` and is set only for a unique pass."""

    model: str
    status: str
    pose: np.ndarray | None
    best: Candidate | None
    candidates: list[Candidate] = field(default_factory=list)


@dataclass(frozen=True)
class Registrar:
    profile: RegistrationProfile = RegistrationProfile()
    gate: IcpGate = IcpGate()
    plausibility: PlausibilityGate = PlausibilityGate()
    extra: ExtraGates | None = None

    def register(
        self,
        model: MeshModel,
        cloud: OrganizedCloud,
        *,
        mask: np.ndarray | None = None,
        nominal: np.ndarray | None = None,
        pinhole: Pinhole | None = None,
    ) -> Registration:
        """Register ``model`` to the observed pixels in ``mask``; ``nominal`` is ``T_camera_model``."""
        p = self.profile
        observation = cloud.points(mask)
        if len(observation) < 100:
            return Registration(model.name, REJECTED, None, None)
        pinhole = pinhole or fit_pinhole(cloud)
        model_points, model_normals = model.surface(p.voxel_size_m / 3.0)
        normal_search = o3d.geometry.KDTreeSearchParamHybrid(
            radius=p.normal_radius_m, max_nn=p.normal_max_nn
        )
        source, target = to_o3d(model_points, model_normals), to_o3d(observation)
        target.estimate_normals(normal_search)
        target.orient_normals_towards_camera_location(np.zeros(3))

        seeds = [
            (f"ransac_seed_{seed}", pose)
            for seed, pose in zip(p.random_seeds, self._ransac(source, target))
        ]
        if nominal is not None:
            seeds.append(("nominal", np.asarray(nominal, dtype=np.float64)))
        candidates = [
            self._evaluate(
                model, cloud, pinhole, mask, source, target, name, initial, nominal
            )
            for name, initial in seeds
        ]
        return _select(model.name, candidates, p)

    def _ransac(self, source, target) -> list[np.ndarray]:
        p = self.profile
        reg = o3d.pipelines.registration
        normal_search = o3d.geometry.KDTreeSearchParamHybrid(
            radius=p.normal_radius_m, max_nn=p.normal_max_nn
        )
        feature_search = o3d.geometry.KDTreeSearchParamHybrid(
            radius=p.feature_radius_m, max_nn=p.feature_max_nn
        )
        source_down, target_down = (
            source.voxel_down_sample(p.voxel_size_m),
            target.voxel_down_sample(p.voxel_size_m),
        )
        for down in (source_down, target_down):
            down.estimate_normals(normal_search)
        source_feature = reg.compute_fpfh_feature(source_down, feature_search)
        target_feature = reg.compute_fpfh_feature(target_down, feature_search)
        poses = []
        for seed in p.random_seeds:
            o3d.utility.random.seed(int(seed))
            result = reg.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_feature,
                target_feature,
                True,
                p.ransac_distance_threshold_m,
                reg.TransformationEstimationPointToPoint(False),
                p.ransac_n,
                [
                    reg.CorrespondenceCheckerBasedOnEdgeLength(
                        p.ransac_edge_length_checker
                    ),
                    reg.CorrespondenceCheckerBasedOnDistance(
                        p.ransac_distance_threshold_m
                    ),
                ],
                reg.RANSACConvergenceCriteria(
                    p.ransac_max_iterations, p.ransac_confidence
                ),
            )
            poses.append(np.asarray(result.transformation))
        return poses

    def _evaluate(
        self, model, cloud, pinhole, mask, source, target, name, initial, nominal
    ) -> Candidate:
        p = self.profile
        reg = o3d.pipelines.registration
        robust = reg.registration_icp(
            source,
            target,
            p.icp_distance_threshold_m,
            initial,
            reg.TransformationEstimationPointToPlane(
                reg.TukeyLoss(k=p.icp_tukey_scale_m)
            ),
            reg.ICPConvergenceCriteria(max_iteration=p.icp_max_iterations),
        )
        # Fine ICP on the visible model only, as the legacy pipeline registers the sim-rendered segment:
        # the full surface would pull thin walls onto their inner faces.
        pose = np.asarray(robust.transformation)
        for _ in range(p.fine_rounds):
            visible = self.visible_model(model, pose, cloud, pinhole, mask)
            if len(visible) < 100:
                break
            fine = reg.registration_icp(
                to_o3d(transform_points(invert(pose), visible)),
                target,
                p.fine_icp_distance_threshold_m,
                pose,
                reg.TransformationEstimationPointToPlane(),
                reg.ICPConvergenceCriteria(max_iteration=p.icp_max_iterations),
            )
            pose = np.asarray(fine.transformation)
        visible = self.visible_model(model, pose, cloud, pinhole, mask)
        observation = np.asarray(target.points)
        candidate = Candidate(name, pose, self.gate.evaluate(visible, observation))
        if nominal is not None:
            candidate.plausibility = self.plausibility.evaluate(pose, nominal)
        if self.extra is not None:
            moved = to_o3d(visible)
            moved.estimate_normals()
            candidate.extra = self.extra.evaluate(
                visible,
                np.asarray(moved.normals),
                observation,
                np.asarray(target.normals),
            )
        return candidate

    def visible_model(
        self,
        model: MeshModel,
        pose: np.ndarray,
        cloud: OrganizedCloud,
        pinhole: Pinhole,
        mask=None,
    ) -> np.ndarray:
        """The model at ``pose`` as the camera sees it: rendered on the observation's pixel grid,
        without pixels where the observation lies clearly in front of it (occluded) and,
        like the observation, only inside ``mask`` (so the table band cuts both alike)."""
        rendered = render(
            raycasting_scene([(model.vertices_m, model.triangles, pose)]), pinhole
        ).xyz_m
        observed_z = cloud.xyz_m[:, :, 2]
        keep = np.isfinite(rendered[:, :, 2])
        occluded = np.isfinite(observed_z) & (
            observed_z < rendered[:, :, 2] - self.profile.occlusion_tolerance_m
        )
        keep &= ~np.nan_to_num(occluded, nan=False).astype(bool)
        if mask is not None:
            keep &= mask
        return rendered[keep]


def _select(
    name: str, candidates: list[Candidate], profile: RegistrationProfile
) -> Registration:
    ordered = sorted(
        candidates,
        key=lambda c: (c.passed, c.gate["fitness"], -c.gate["rmse_m"]),
        reverse=True,
    )
    representatives: list[Candidate] = []
    for candidate in ordered:
        if not any(
            t < profile.dedup_translation_m and r < profile.dedup_rotation_deg
            for t, r in (
                pose_difference(candidate.pose, kept.pose) for kept in representatives
            )
        ):
            representatives.append(candidate)
    passing = [c for c in representatives if c.passed]
    best = ordered[0] if ordered else None
    if len(passing) == 1:
        return Registration(name, UNIQUE_PASS, passing[0].pose, passing[0], ordered)
    return Registration(name, AMBIGUOUS if passing else REJECTED, None, best, ordered)
