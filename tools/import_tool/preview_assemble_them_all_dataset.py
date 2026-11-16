from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import open3d as o3d


def discover_case_dirs(dataset_dir: Path) -> list[Path]:
    case_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]

    def sort_key(path: Path):
        name = path.name
        return (0, int(name)) if name.isdigit() else (1, name)

    return sorted(case_dirs, key=sort_key)


def discover_obj_files(case_dir: Path) -> list[Path]:
    obj_files = list(case_dir.glob("*.obj"))

    def sort_key(path: Path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(obj_files, key=sort_key)


def parse_case_ids(case_ids_raw: str | None) -> set[str] | None:
    if not case_ids_raw:
        return None
    return {token.strip() for token in case_ids_raw.split(",") if token.strip()}


def load_case_meshes_o3d(case_dir: Path) -> list[o3d.geometry.TriangleMesh]:
    """
    Load all OBJ files in one case directory as Open3D meshes.
    Assumes they are already in assembled pose.
    """
    obj_files = discover_obj_files(case_dir)
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {case_dir}")

    palette = [
        [0.85, 0.45, 0.45],
        [0.45, 0.65, 0.85],
        [0.50, 0.80, 0.50],
        [0.85, 0.70, 0.45],
        [0.70, 0.55, 0.85],
        [0.40, 0.80, 0.75],
    ]

    meshes: list[o3d.geometry.TriangleMesh] = []

    for idx, obj_file in enumerate(obj_files):
        mesh = o3d.io.read_triangle_mesh(str(obj_file))
        if mesh.is_empty():
            raise ValueError(f"Failed to load OBJ or mesh is empty: {obj_file}")
        if len(mesh.triangles) == 0:
            raise ValueError(f"OBJ has no triangles: {obj_file}")

        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(palette[idx % len(palette)])
        meshes.append(mesh)

    return meshes


def compute_scene_bounds(
    meshes: list[o3d.geometry.TriangleMesh],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Return (min_bound, max_bound, center, diagonal_length)
    """
    if not meshes:
        raise ValueError("No meshes to bound")

    mins = []
    maxs = []
    for mesh in meshes:
        aabb = mesh.get_axis_aligned_bounding_box()
        mins.append(np.asarray(aabb.min_bound, dtype=float))
        maxs.append(np.asarray(aabb.max_bound, dtype=float))

    min_bound = np.min(np.vstack(mins), axis=0)
    max_bound = np.max(np.vstack(maxs), axis=0)
    center = 0.5 * (min_bound + max_bound)
    diag = float(np.linalg.norm(max_bound - min_bound))
    diag = max(diag, 1.0)

    return min_bound, max_bound, center, diag


def render_case_preview_o3d(
    case_dir: Path,
    output_png: Path,
    image_size: tuple[int, int] = (800, 600),
) -> int:
    """
    Render one case directory to a PNG preview using Open3D OffscreenRenderer.
    Returns the number of OBJ parts rendered.
    """
    meshes = load_case_meshes_o3d(case_dir)
    _, _, center, diag = compute_scene_bounds(meshes)

    width, height = image_size
    output_png.parent.mkdir(parents=True, exist_ok=True)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])

    # Strong sun light + IBL
    scene.scene.enable_sun_light(True)
    scene.scene.set_sun_light(
        direction=[-0.5, -0.5, -1.0],
        color=[1.0, 1.0, 1.0],
        intensity=75000,
    )
    scene.scene.enable_indirect_light(False)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    for idx, mesh in enumerate(meshes):
        scene.add_geometry(f"mesh_{idx}", mesh, material)

    # Camera: diagonal front-top-right view
    eye = center + np.array([1.7, -1.5, 1.2]) * diag
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    vertical_fov = 50.0

    renderer.setup_camera(vertical_fov, center, eye, up)
    # setup_camera usually suffices; no manual intrinsic needed here

    img = renderer.render_to_image()
    o3d.io.write_image(str(output_png), img)

    # Explicit cleanup helps when batch rendering many cases
    renderer.scene.clear_geometry()
    del renderer

    return len(meshes)


def write_index_html(rows: list[dict], output_html: Path, title: str) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)

    cards = []
    for row in rows:
        case_id = html.escape(row["case_id"])
        image_rel = html.escape(row["image_rel"])
        num_parts = html.escape(str(row["num_parts"]))
        status = html.escape(row["status"])
        note = html.escape(row["note"])

        if status == "success":
            card = f"""
            <div class="card">
                <img src="{image_rel}" alt="{case_id}">
                <div class="meta">
                    <div><strong>{case_id}</strong></div>
                    <div>parts: {num_parts}</div>
                    <div>status: {status}</div>
                </div>
            </div>
            """
        else:
            card = f"""
            <div class="card">
                <div class="error-box">FAILED</div>
                <div class="meta">
                    <div><strong>{case_id}</strong></div>
                    <div>status: {status}</div>
                    <div class="note">{note}</div>
                </div>
            </div>
            """
        cards.append(card)

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 24px;
    background: #f7f7f7;
    color: #222;
}}
h1 {{
    margin-bottom: 8px;
}}
p {{
    margin-top: 0;
    color: #555;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
}}
.card {{
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    overflow: hidden;
    border: 1px solid #e5e5e5;
}}
.card img {{
    width: 100%;
    height: 220px;
    object-fit: contain;
    background: #ffffff;
    display: block;
}}
.meta {{
    padding: 12px;
    font-size: 14px;
    line-height: 1.5;
}}
.error-box {{
    height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #fff1f1;
    color: #a33;
    font-weight: bold;
}}
.note {{
    color: #a33;
    font-size: 12px;
}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p>Generated preview index for manual test-case screening.</p>
<div class="grid">
{"".join(cards)}
</div>
</body>
</html>
"""
    output_html.write_text(html_text, encoding="utf-8")


def preview_dataset(
    dataset_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    case_ids: set[str] | None = None,
    skip_existing: bool = False,
    offset: int = 0,
) -> list[dict]:
    case_dirs = discover_case_dirs(dataset_dir)

    if case_ids:
        case_dirs = [p for p in case_dirs if p.name in case_ids]

    if offset:
        case_dirs = case_dirs[offset:]

    if limit is not None:
        case_dirs = case_dirs[:limit]

    print(f"[INFO] Found {len(case_dirs)} cases in {dataset_dir}")

    preview_dir = output_dir / "previews"
    rows: list[dict] = []

    for idx, case_dir in enumerate(case_dirs, start=1):
        case_id = case_dir.name
        png_path = preview_dir / f"{case_id}.png"

        print("=" * 80)
        print(f"[INFO] [{idx}/{len(case_dirs)}] Rendering {case_id}")

        if skip_existing and png_path.exists():
            print(f"[INFO] Skipping existing preview: {png_path}")
            rows.append(
                {
                    "case_id": case_id,
                    "num_parts": "",
                    "status": "skipped_existing",
                    "note": "",
                    "image_rel": f"previews/{case_id}.png",
                }
            )
            continue

        try:
            num_parts = render_case_preview_o3d(case_dir, png_path)
            rows.append(
                {
                    "case_id": case_id,
                    "num_parts": num_parts,
                    "status": "success",
                    "note": "",
                    "image_rel": f"previews/{case_id}.png",
                }
            )
        except Exception as exc:
            print(f"[ERROR] Failed to render {case_id}: {exc}")
            rows.append(
                {
                    "case_id": case_id,
                    "num_parts": "",
                    "status": "failed",
                    "note": f"{type(exc).__name__}: {exc}",
                    "image_rel": "",
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for preview PNGs and HTML index"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N cases"
    )
    parser.add_argument(
        "--case-ids", default=None, help="Comma-separated case ids, e.g. 00016,00031"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip already rendered PNGs"
    )
    parser.add_argument("--title", default=None, help="Optional gallery title")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N cases")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    case_ids = parse_case_ids(args.case_ids)

    rows = preview_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        limit=args.limit,
        case_ids=case_ids,
        skip_existing=args.skip_existing,
        offset=args.offset,
    )

    title = args.title or f"Preview index: {dataset_dir.name}"
    output_html = output_dir / "index.html"
    write_index_html(rows, output_html, title=title)

    print(f"[DONE] HTML index written to: {output_html}")


if __name__ == "__main__":
    main()
