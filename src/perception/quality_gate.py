"""Acceptance gates for a registered pose.

The defaults are #45's acceptance values: the legacy ``check_icp_match`` of
``crc_disassembly_sequence_generator/perceptionModule.py`` (judged there in mm
with a one-step ICP at 1.0 mm) and the ~5 mm / 3 deg plausibility gate against
the nominal pose. Hao's extra geometry gates can run in addition; they are off
by default and never replace the legacy gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from perception.cloud import pose_difference, to_o3d


@dataclass(frozen=True)
class IcpGate:
    """Legacy ``check_icp_match``: fitness > 0.7, RMSE < 1.0 mm, fitness / RMSE[mm] > 1.5."""

    correspondence_m: float = 0.001
    min_fitness: float = 0.7
    max_rmse_m: float = 0.001
    min_fitness_per_mm: float = 1.5

    def evaluate(self, source_m: np.ndarray, target_m: np.ndarray) -> dict:
        """Source: the model points as the camera would see them at the pose; target: the observation."""
        result = o3d.pipelines.registration.evaluate_registration(
            to_o3d(source_m), to_o3d(target_m), self.correspondence_m, np.eye(4)
        )
        fitness, rmse_m = float(result.fitness), float(result.inlier_rmse)
        ratio = fitness / max(rmse_m * 1000.0, 1e-5)
        passed = (
            fitness > self.min_fitness
            and rmse_m < self.max_rmse_m
            and ratio > self.min_fitness_per_mm
        )
        return {
            "fitness": fitness,
            "rmse_m": rmse_m,
            "fitness_per_mm": ratio,
            "pass": bool(passed),
        }


@dataclass(frozen=True)
class PlausibilityGate:
    """The pose may differ from the nominal pose by at most this much."""

    max_translation_m: float = 0.005
    max_rotation_deg: float = 3.0

    def evaluate(self, pose: np.ndarray, nominal: np.ndarray) -> dict:
        translation, rotation = pose_difference(pose, nominal)
        passed = (
            translation <= self.max_translation_m and rotation <= self.max_rotation_deg
        )
        return {
            "translation_m": translation,
            "rotation_deg": rotation,
            "pass": bool(passed),
        }


# Hao's frozen Stage 92I values for the gates that do not need product features or semantic labels.
HAO_EXTRA_THRESHOLDS = {
    "inlier_threshold_m": 0.010,
    "minimum_observation_inlier_ratio": 0.30,
    "maximum_observation_rmse_m": 0.012,
    "minimum_cad_visible_coverage": 0.20,
    "minimum_normal_consistency": 0.45,
    "maximum_symmetric_rmse_m": 0.020,
    "minimum_bidirectional_overlap": 0.20,
}


@dataclass(frozen=True)
class ExtraGates:
    """Hao's coverage, normal consistency, symmetric RMSE and overlap gates (opt-in)."""

    thresholds: dict = field(default_factory=lambda: dict(HAO_EXTRA_THRESHOLDS))

    def evaluate(
        self,
        visible_model_m: np.ndarray,
        model_normals: np.ndarray,
        observation_m: np.ndarray,
        observation_normals,
    ) -> dict:
        t = self.thresholds
        if len(visible_model_m) == 0 or len(observation_m) == 0:
            return {"pass": False, "reason": "empty input"}
        observation_distance, nearest = cKDTree(visible_model_m).query(observation_m)
        model_distance, _ = cKDTree(observation_m).query(visible_model_m)
        inlier = observation_distance <= t["inlier_threshold_m"]
        coverage = float((model_distance <= t["inlier_threshold_m"]).mean())
        dots = np.abs(
            np.sum(
                np.asarray(observation_normals) * np.asarray(model_normals)[nearest],
                axis=1,
            )
        )
        observation_rmse = float(np.sqrt(np.mean(observation_distance**2)))
        model_rmse = float(np.sqrt(np.mean(model_distance**2)))
        metrics = {
            "observation_inlier_ratio": float(inlier.mean()),
            "observation_rmse_m": observation_rmse,
            "cad_visible_coverage": coverage,
            "normal_consistency": float(dots[inlier].mean()) if inlier.any() else 0.0,
            "symmetric_rmse_m": float(
                np.sqrt((observation_rmse**2 + model_rmse**2) / 2.0)
            ),
            "bidirectional_overlap": float(min(inlier.mean(), coverage)),
        }
        gates = {
            "observation_inlier_ratio": metrics["observation_inlier_ratio"]
            >= t["minimum_observation_inlier_ratio"],
            "observation_rmse_m": metrics["observation_rmse_m"]
            <= t["maximum_observation_rmse_m"],
            "cad_visible_coverage": metrics["cad_visible_coverage"]
            >= t["minimum_cad_visible_coverage"],
            "normal_consistency": metrics["normal_consistency"]
            >= t["minimum_normal_consistency"],
            "symmetric_rmse_m": metrics["symmetric_rmse_m"]
            <= t["maximum_symmetric_rmse_m"],
            "bidirectional_overlap": metrics["bidirectional_overlap"]
            >= t["minimum_bidirectional_overlap"],
        }
        return {**metrics, "gates": gates, "pass": bool(all(gates.values()))}
