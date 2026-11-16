"""
Headless(-ish) 3D rendering of Assembly states, for use by graph UI builders.

open3d's modern OffscreenRenderer requires EGL and raises
"EGL Headless is not supported on this platform" on macOS. The legacy
`o3d.visualization.Visualizer` with `create_window(visible=False)` works
fine there (proven empirically) since it uses a real, just invisible,
GL context rather than EGL -- so that's what this module is built on.

The renderer fits its camera once to the full base assembly (all parts
present) and reuses those exact camera parameters for every subsequent
render, so removing parts doesn't shift the viewpoint/zoom -- states stay
visually comparable as you navigate the disassembly graph.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import open3d as o3d
from PIL import Image

_PALETTE = [
    (0.85, 0.35, 0.35),
    (0.35, 0.55, 0.85),
    (0.40, 0.75, 0.45),
    (0.90, 0.70, 0.25),
    (0.60, 0.45, 0.80),
    (0.30, 0.75, 0.75),
    (0.90, 0.50, 0.70),
    (0.55, 0.55, 0.55),
    (0.75, 0.60, 0.35),
    (0.35, 0.65, 0.90),
]
HIGHLIGHT_COLOR = (0.95, 0.25, 0.20)


class AssemblyRenderer:
    """
    Renders Assembly states to PNG bytes from a fixed, auto-fit camera.

    Usage:
        renderer = AssemblyRenderer(base_assembly)
        png_bytes = renderer.render_state(some_assembly)
        png_bytes_highlighted = renderer.render_state(some_assembly, highlight_part="some_part")
        renderer.close()
    """

    def __init__(self, base_assembly, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        part_names = sorted(p.name for p in base_assembly.parts.values())
        self.color_by_part = {
            name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(part_names)
        }

        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(visible=False, width=width, height=height)
        opt = self._vis.get_render_option()
        opt.background_color = np.array([0.98, 0.98, 0.99])
        opt.light_on = True

        self._camera = None
        self._fit_camera(base_assembly)

    # ------------------------------------------------------------------
    def _colored_mesh(self, part, color) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh(part.mesh)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh

    def _fit_camera(self, assembly) -> None:
        self._vis.clear_geometries()
        for part in assembly.parts.values():
            mesh = self._colored_mesh(
                part, self.color_by_part.get(part.name, (0.6, 0.6, 0.6))
            )
            self._vis.add_geometry(mesh)
        self._vis.reset_view_point(True)
        self._vis.poll_events()
        self._vis.update_renderer()
        self._camera = (
            self._vis.get_view_control().convert_to_pinhole_camera_parameters()
        )

    def render_state(self, assembly, highlight_part: Optional[str] = None) -> bytes:
        """Render `assembly` from the fixed camera. Returns PNG bytes."""
        self._vis.clear_geometries()
        for part in assembly.parts.values():
            color = self.color_by_part.get(part.name, (0.6, 0.6, 0.6))
            if highlight_part is not None and part.name == highlight_part:
                color = HIGHLIGHT_COLOR
            mesh = self._colored_mesh(part, color)
            self._vis.add_geometry(mesh, reset_bounding_box=False)

        if self._camera is not None:
            self._vis.get_view_control().convert_from_pinhole_camera_parameters(
                self._camera
            )
        self._vis.poll_events()
        self._vis.update_renderer()

        img = self._vis.capture_screen_float_buffer(do_render=True)
        arr = (np.asarray(img) * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def render_state_data_uri(
        self, assembly, highlight_part: Optional[str] = None
    ) -> str:
        png_bytes = self.render_state(assembly, highlight_part=highlight_part)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def close(self) -> None:
        self._vis.destroy_window()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
