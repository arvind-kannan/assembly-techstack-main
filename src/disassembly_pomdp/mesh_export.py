"""
Export Part geometry for a live in-browser WebGL viewer.

Heavy CAD meshes (e.g. a stator with 128k triangles) are decimated down to
a triangle budget first -- quadric decimation preserves the overall shape
well at a fraction of the size (verified visually: 128,927 -> 4,000
triangles is essentially indistinguishable at UI scale) and keeps the page
small enough to embed everything in one self-contained HTML file.

Geometry is encoded as raw base64 binary (Float32 positions/normals,
Uint32 indices) rather than JSON number arrays -- far more compact, and
trivial to decode client-side into typed arrays for WebGL buffers.
"""

from __future__ import annotations

import base64

import numpy as np
import open3d as o3d


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def export_part_mesh(part, max_triangles: int = 4000) -> dict:
    """
    Returns a JSON-serializable dict describing `part`'s (possibly
    decimated) geometry, ready to be decoded into WebGL buffers:

        positions_b64 / normals_b64 : base64 Float32Array (xyz per vertex)
        indices_b64                 : base64 Uint32Array (3 per triangle)
        index_count                 : number of indices (len(indices))
        bbox_center / bbox_diag     : for animation travel-distance scaling
    """
    mesh = o3d.geometry.TriangleMesh(part.mesh)

    if len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=max_triangles
        )
        # simplify_quadric_decimation only shrinks the triangle list -- it
        # leaves the (mostly now-unused) original vertex array untouched,
        # so without this the "decimated" mesh keeps ~all original vertices.
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    positions = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    indices = np.asarray(mesh.triangles, dtype=np.uint32).reshape(-1)

    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=np.float64)
    diag = float(np.linalg.norm(bbox.get_extent()))

    return {
        "positions_b64": _b64(positions),
        "normals_b64": _b64(normals),
        "indices_b64": _b64(indices),
        "index_count": int(indices.shape[0]),
        "bbox_center": center.tolist(),
        "bbox_diag": diag,
    }
