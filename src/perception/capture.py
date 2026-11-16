"""Organized clouds from a Zivid camera, a saved .zdf frame, or the cell simulation's RGBDCamera.

Zivid delivers XYZ in mm; ``_mm_to_m`` is the one place that converts it. The
simulation already works in metres and is not scaled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perception.cloud import OrganizedCloud

M60_SETTINGS = Path(__file__).resolve().parent / "config" / "zivid2_plus_m60.yml"


def _mm_to_m(xyz_mm: np.ndarray) -> np.ndarray:
    return np.asarray(xyz_mm, dtype=np.float64) * 0.001


def zivid_frame_to_cloud(frame) -> OrganizedCloud:
    """A ``zivid.Frame`` as an organized cloud in metres (Zivid camera frame)."""
    point_cloud = frame.point_cloud()
    xyz_mm = np.asarray(point_cloud.copy_data("xyz"))
    rgba = np.asarray(point_cloud.copy_data("rgba"))
    rgb = (
        rgba[:, :, :3].astype(np.uint8) if rgba.shape[:2] == xyz_mm.shape[:2] else None
    )
    return OrganizedCloud(_mm_to_m(xyz_mm), rgb)


def _zivid():
    try:
        import zivid
    except ImportError as error:
        raise ImportError(
            "the Zivid SDK is not installed: install it first, then `uv sync --group hardware`"
        ) from error
    return zivid


def load_zdf(path: str | Path) -> OrganizedCloud:
    """A saved .zdf frame, decoded offline (no camera needed, but the Zivid SDK is)."""
    zivid = _zivid()
    application = zivid.Application()  # the SDK needs one before it decodes frames
    cloud = zivid_frame_to_cloud(zivid.Frame(str(path)))
    del application
    return cloud


class ZividCapture:
    """The Zivid 2+ M60 on the cell; connects on the first capture."""

    def __init__(self, settings_file: str | Path | None = M60_SETTINGS):
        self.settings_file = settings_file
        self.application = self.camera = self.settings = self.frame = None

    def connect(self):
        zivid = _zivid()
        self.application = zivid.Application()
        self.camera = self.application.connect_camera()
        if self.settings_file is not None:
            self.settings = zivid.Settings.load(str(self.settings_file))

    def capture(self) -> OrganizedCloud:
        if self.camera is None:
            self.connect()
        self.frame = (
            self.camera.capture(self.settings)
            if self.settings is not None
            else self.camera.capture()
        )
        return zivid_frame_to_cloud(self.frame)

    def save_zdf(self, path: str | Path):
        if self.frame is None:
            raise ValueError("no frame captured yet")
        self.frame.save(str(path))


def sim_frame_to_cloud(last_frame: dict, projection_matrix) -> OrganizedCloud:
    """``RGBDCamera.get_image()`` (its ``last_frame``) as an organized cloud in the
    Zivid axis convention (x right, y down, z forward), in metres.

    ``projection_matrix`` is the camera's pybullet projection matrix (16 values,
    column-major). Pixels without an object (mask < 0) are NaN.
    """
    depth = np.asarray(last_frame["depth_img"], dtype=np.float64)
    height, width = depth.shape
    rows, cols = np.mgrid[0:height, 0:width]
    # The same pixel-to-NDC grid as RGBDCamera.get_image.
    ndc = np.stack(
        [
            -1.0 + 2.0 * cols / width,
            1.0 - 2.0 * rows / height,
            2.0 * depth - 1.0,
            np.ones_like(depth),
        ],
        axis=2,
    )
    inverse = np.linalg.inv(
        np.asarray(projection_matrix, dtype=np.float64).reshape(4, 4, order="F")
    )
    eye = ndc @ inverse.T
    eye = eye[:, :, :3] / eye[:, :, 3:4]  # OpenGL eye frame: x right, y up, z backward
    xyz = eye * np.array([1.0, -1.0, -1.0])
    xyz[np.asarray(last_frame["mask"]) < 0] = np.nan
    rgb = np.asarray(last_frame["img"])[:, :, :3].astype(np.uint8)
    return OrganizedCloud(xyz, rgb)
