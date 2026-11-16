"""Capture adapters: Zivid (mm, converted once) and the sim RGBDCamera (already m) give the same metres."""

import sys
import types

import numpy as np
import pytest

from perception import capture
from perception.capture import (
    M60_SETTINGS,
    ZividCapture,
    load_zdf,
    sim_frame_to_cloud,
    zivid_frame_to_cloud,
)

H, W = 48, 64


def _plane_mm(depth_mm=800.0):
    rows, cols = np.mgrid[0:H, 0:W]
    xyz = np.stack(
        [
            (cols - W / 2) * depth_mm / 50.0,
            (rows - H / 2) * depth_mm / 50.0,
            np.full((H, W), depth_mm),
        ],
        axis=2,
    ).astype(np.float32)
    xyz[0, 0] = np.nan  # Zivid marks missing points as NaN
    return xyz


class FakeFrame:
    def __init__(self, xyz_mm):
        self.xyz_mm, self.saved = xyz_mm, None

    def point_cloud(self):
        frame = self

        class PointCloud:
            def copy_data(self, kind):
                if kind == "xyz":
                    return frame.xyz_mm
                return np.full((*frame.xyz_mm.shape[:2], 4), 200, dtype=np.uint8)

        return PointCloud()

    def save(self, path):
        self.saved = path


@pytest.fixture
def fake_zivid(monkeypatch):
    module = types.SimpleNamespace(loaded_settings=[])

    class Camera:
        def capture(self, settings=None):
            module.capture_settings = settings
            return FakeFrame(_plane_mm())

    class Application:
        def connect_camera(self):
            return Camera()

    module.Application = Application
    module.Frame = lambda path: FakeFrame(_plane_mm())
    module.Settings = types.SimpleNamespace(
        load=lambda path: module.loaded_settings.append(path) or "settings"
    )
    monkeypatch.setitem(sys.modules, "zivid", module)
    return module


def test_zivid_frame_is_converted_from_mm_to_m_once():
    cloud = zivid_frame_to_cloud(FakeFrame(_plane_mm()))
    assert np.nanmax(cloud.xyz_m[:, :, 2]) == pytest.approx(0.8)
    assert np.isnan(cloud.xyz_m[0, 0]).all() and cloud.valid.sum() == H * W - 1
    assert cloud.rgb.shape == (H, W, 3)


def test_zivid_capture_uses_the_m60_settings_and_keeps_the_frame(fake_zivid):
    camera = ZividCapture()
    cloud = camera.capture()
    assert fake_zivid.loaded_settings == [str(M60_SETTINGS)] and M60_SETTINGS.is_file()
    assert fake_zivid.capture_settings == "settings"
    assert np.nanmedian(cloud.xyz_m[:, :, 2]) == pytest.approx(0.8)
    camera.save_zdf("capture.zdf")
    assert camera.frame.saved == "capture.zdf"


def test_zivid_capture_without_settings_uses_camera_defaults(fake_zivid):
    ZividCapture(settings_file=None).capture()
    assert fake_zivid.loaded_settings == [] and fake_zivid.capture_settings is None
    with pytest.raises(ValueError):
        ZividCapture().save_zdf("x.zdf")


def test_load_zdf_decodes_offline(fake_zivid):
    assert np.nanmedian(load_zdf("frame.zdf").xyz_m[:, :, 2]) == pytest.approx(0.8)


def test_missing_zivid_sdk_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "zivid", None)
    with pytest.raises(ImportError, match="Zivid SDK"):
        load_zdf("frame.zdf")


def _projection(fov_deg=60.0, aspect=W / H, near=0.01, far=5.0):
    """pybullet.computeProjectionMatrixFOV, column-major as pybullet returns it."""
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    matrix = np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ]
    )
    return matrix, list(matrix.flatten(order="F"))


def test_sim_frame_is_in_metres_with_zivid_axes():
    matrix, flat = _projection()
    clip = matrix @ [0.0, 0.0, -0.8, 1.0]  # a wall 0.8 m in front of the camera
    depth = np.full((H, W), (clip[2] / clip[3] + 1.0) / 2.0)
    mask = np.zeros((H, W), dtype=int)
    mask[0, :] = -1  # background
    frame = {
        "img": np.zeros((H, W, 4), dtype=np.uint8),
        "depth_img": depth,
        "mask": mask,
    }
    cloud = sim_frame_to_cloud(frame, flat)
    assert (
        np.nanmax(np.abs(cloud.xyz_m[:, :, 2] - 0.8)) < 1e-9
    )  # same metres as the Zivid path
    assert np.isnan(cloud.xyz_m[0]).all()
    assert cloud.xyz_m[10, -1, 0] > 0 > cloud.xyz_m[10, 0, 0]  # x to the right
    assert cloud.xyz_m[-1, 5, 1] > 0 > cloud.xyz_m[1, 5, 1]  # y down


def test_mm_to_m_is_the_only_scaling():
    assert capture._mm_to_m(np.array([1000.0]))[0] == 1.0
