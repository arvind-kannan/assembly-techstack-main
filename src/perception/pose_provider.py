"""World-frame poses of the product, the System 28 desk and its clamps from one capture.

This is the import surface for #45: give it an organized cloud (from
``capture.zivid_frame_to_cloud``, ``capture.sim_frame_to_cloud`` or a render),
``T_world_camera`` and the nominal poses from the cell description, and it
returns ``T_world_product``, ``T_world_desk`` and ``T_world_clamp`` per clamp,
each with its gate report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from perception.cloud import (
    OrganizedCloud,
    Pinhole,
    above_plane,
    fit_pinhole,
    fit_table_plane,
    invert,
    raycasting_scene,
    render,
)
from perception.desk_markers import DeclaredShape, choose_by_aruco, choose_by_shapes
from perception.registrar import REJECTED, MeshModel, Registrar, Registration
from perception.system28 import (
    DeskGate,
    DeskHypotheses,
    DeskLocalizationError,
    localize_desk,
    select_nearest_to_nominal,
)


@dataclass
class DeskResult:
    """``pose`` is ``T_world_desk``, None when the hypotheses could not be told apart.
    ``resolved_by`` is "unique", "aruco", "shapes", "clamps" or "nominal"."""

    pose: np.ndarray | None
    resolved_by: str | None
    hypotheses: DeskHypotheses | None
    error: str | None = None


@dataclass
class ScenePoses:
    product: Registration
    desk: DeskResult
    clamps: dict[str, Registration] = field(default_factory=dict)

    @property
    def T_world_product(self) -> np.ndarray | None:
        return self.product.pose

    @property
    def T_world_desk(self) -> np.ndarray | None:
        return self.desk.pose

    @property
    def T_world_clamps(self) -> dict[str, np.ndarray | None]:
        return {name: result.pose for name, result in self.clamps.items()}


@dataclass(frozen=True)
class PoseProvider:
    registrar: Registrar = Registrar()
    aruco_markers: dict[int, tuple[float, float]] | None = None  # marker id -> desk x/y
    shape_markers: dict[str, DeclaredShape] | None = None
    desk_gate: DeskGate = DeskGate()
    table_band_m: float = 0.003
    crop_scale: float = (
        1.1  # the legacy pipeline inflates the nominal bounding box by 1.1
    )
    crop_margin_m: float = 0.010
    # The clamp step decides only when the best desk hypothesis scores this much higher (on a
    # scale where a fully supported layout scores about 1, see _clamp_scores) than the second best. The tolerance is looser than the
    # acceptance gate's, because the clamps stand at their nominal poses, not at registered ones.
    clamp_margin: float = 0.5
    clamp_correspondence_m: float = 0.005

    def locate(
        self,
        cloud: OrganizedCloud,
        T_world_camera: np.ndarray,
        *,
        product: MeshModel,
        nominal_world_product: np.ndarray | None = None,
        nominal_world_desk: np.ndarray | None = None,
        clamps: dict[str, tuple[MeshModel, np.ndarray]] | None = None,
    ) -> ScenePoses:
        """``clamps`` maps a clamp name to its model and its nominal ``T_desk_clamp``."""
        T_world_camera = np.asarray(T_world_camera, dtype=np.float64)
        T_camera_world = invert(T_world_camera)
        pinhole = fit_pinhole(cloud)
        normal, offset = fit_table_plane(cloud.points())
        off_table = above_plane(cloud, normal, offset, self.table_band_m)
        desk = self._desk(
            cloud,
            normal,
            offset,
            T_world_camera,
            nominal_world_desk,
            pinhole,
            off_table,
            clamps,
        )

        def register(model, nominal_world):
            nominal = None if nominal_world is None else T_camera_world @ nominal_world
            mask = off_table & (
                self._crop(cloud, model, nominal) if nominal is not None else True
            )
            result = self.registrar.register(
                model, cloud, mask=mask, nominal=nominal, pinhole=pinhole
            )
            return _to_world(result, T_world_camera)

        scene = ScenePoses(register(product, nominal_world_product), desk)
        for name, (model, T_desk_clamp) in (clamps or {}).items():
            if desk.pose is None:
                scene.clamps[name] = Registration(model.name, REJECTED, None, None)
            else:
                scene.clamps[name] = register(model, desk.pose @ T_desk_clamp)
        return scene

    def _desk(
        self,
        cloud,
        normal,
        offset,
        T_world_camera,
        nominal_world,
        pinhole,
        off_table,
        clamps,
    ) -> DeskResult:
        try:
            hypotheses = localize_desk(cloud, normal, offset)
        except DeskLocalizationError as error:
            return DeskResult(None, None, None, str(error))
        if not self.desk_gate.passes(hypotheses):
            return DeskResult(
                None, None, hypotheses, "desk grid fit rejected at the gate"
            )
        index, how = None, None
        if len(hypotheses.poses) == 1:
            index, how = 0, "unique"
        if index is None and self.aruco_markers:
            index, how = (
                choose_by_aruco(cloud, hypotheses, self.aruco_markers).index,
                "aruco",
            )
        if index is None and self.shape_markers:
            index, how = (
                choose_by_shapes(cloud, hypotheses, self.shape_markers).index,
                "shapes",
            )
        if index is None and clamps:
            index = self._choose_by_clamps(
                cloud, pinhole, off_table, hypotheses, clamps
            )
            if index is not None:
                how = "clamps"
        if index is None and nominal_world is not None:
            nominal = invert(T_world_camera) @ nominal_world
            index, how = select_nearest_to_nominal(hypotheses, nominal)[0], "nominal"
        if index is None:
            return DeskResult(None, None, hypotheses, "desk hypotheses ambiguous")
        return DeskResult(T_world_camera @ hypotheses.poses[index], how, hypotheses)

    def _choose_by_clamps(
        self,
        cloud: OrganizedCloud,
        pinhole: Pinhole,
        off_table: np.ndarray,
        hypotheses: DeskHypotheses,
        clamps: dict[str, tuple[MeshModel, np.ndarray]],
    ) -> int | None:
        """The desk hypothesis whose clamp layout the scan supports clearly best, else None."""
        scores = self._clamp_scores(cloud, pinhole, off_table, hypotheses, clamps)
        order = np.argsort(scores)[::-1]
        if scores[order[0]] - scores[order[1]] < self.clamp_margin:
            return None
        return int(order[0])

    def _clamp_scores(
        self,
        cloud: OrganizedCloud,
        pinhole: Pinhole,
        off_table: np.ndarray,
        hypotheses: DeskHypotheses,
        clamps: dict[str, tuple[MeshModel, np.ndarray]],
    ) -> list[float]:
        """Per desk hypothesis, the mean over the clamps of how well the scan supports each.

        A clamp is placed at its nominal desk pose under the hypothesis. Its score is the
        registrar's fitness (the share of its visible model points with a scan point nearby),
        scaled by the share of the pixels it covers when unobstructed that stay visible: a clamp
        that the hypothesis puts out of view or behind other parts is not supported."""
        gate = replace(
            self.registrar.gate, correspondence_m=self.clamp_correspondence_m
        )
        observation = cloud.points(off_table)
        scores = []
        for desk_pose in hypotheses.poses:
            per_clamp = []
            for model, T_desk_clamp in clamps.values():
                pose = desk_pose @ T_desk_clamp
                visible = self.registrar.visible_model(
                    model, pose, cloud, pinhole, off_table
                )
                if len(visible) == 0:
                    per_clamp.append(0.0)
                    continue
                covered = np.isfinite(
                    render(
                        raycasting_scene([(model.vertices_m, model.triangles, pose)]),
                        pinhole,
                    ).xyz_m[:, :, 2]
                )
                expected = int(np.count_nonzero(covered & off_table))
                fitness = gate.evaluate(visible, observation)["fitness"]
                per_clamp.append(fitness * len(visible) / expected)
            scores.append(float(np.mean(per_clamp)))
        return scores

    def _crop(self, cloud: OrganizedCloud, model: MeshModel, nominal: np.ndarray):
        """Pixels inside the model's bounding box at the nominal pose, inflated."""
        low, high = model.vertices_m.min(axis=0), model.vertices_m.max(axis=0)
        centre, half = (low + high) / 2.0, (high - low) / 2.0 * self.crop_scale
        half = half + self.crop_margin_m
        inverse = invert(nominal)
        local = np.nan_to_num(cloud.xyz_m, nan=1e3) @ inverse[:3, :3].T + inverse[:3, 3]
        return np.all(np.abs(local - centre) <= half, axis=2)


def _to_world(result: Registration, T_world_camera: np.ndarray) -> Registration:
    if result.pose is not None:
        result.pose = T_world_camera @ result.pose
    for candidate in result.candidates:
        candidate.pose = T_world_camera @ candidate.pose
    return result
