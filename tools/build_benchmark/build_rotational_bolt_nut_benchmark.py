from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part

# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------

OUT_ROOT = ROOT / "benchmark_cases" / "benchmark_bolt_and_nut_case"

SCAD_DIR = OUT_ROOT / "scad"
STL_DIR = OUT_ROOT / "stl"
XML_DIR = OUT_ROOT / "xml"


@dataclass(frozen=True)
class BenchmarkParams:
    case_name: str = "rotational_bolt_nut_lock_benchmark"

    # Pipe geometry
    pipe_outer_radius: float = 21.0
    pipe_inner_radius: float = 15.0
    bend_centerline_radius: float = 51.0

    # Lid side: short -X pipe segment
    straight_len_x: float = 24.0

    # Bolt / nut side: longer +Y pipe segment
    straight_len_y: float = 70.0

    # Bolt station on the +Y straight pipe segment
    bolt_station_y_offset: float = 36.0
    bolt_axis_clearance: float = 4.0

    # Side hole through the pipe wall
    side_hole_radius: float = 3.2

    # Bolt geometry
    bolt_shaft_radius: float = 2.0
    bolt_head_radius: float = 5.4
    bolt_head_thickness: float = 4.0
    bolt_tail_extra: float = 2.0

    # Nut geometry
    nut_inner_radius: float = 2.15
    nut_outer_radius: float = 5.4
    nut_thickness: float = 4.0

    # Cover lid geometry
    lid_thickness: float = 6.0
    lid_outer_radius: float = 21.0

    # Contact / connection reconstruction
    contact_tollerance: float = 0.5
    dot_product_threshold: float = -0.95

    # Rendering / tessellation
    fn_pipe: int = 96
    fn_part: int = 64

    @property
    def pipe_outer_diameter(self) -> float:
        return 2.0 * self.pipe_outer_radius

    @property
    def pipe_inner_diameter(self) -> float:
        return 2.0 * self.pipe_inner_radius

    @property
    def pipe_wall_thickness(self) -> float:
        return self.pipe_outer_radius - self.pipe_inner_radius

    @property
    def shaft_start_z(self) -> float:
        return -self.pipe_outer_radius - self.bolt_axis_clearance - 0.5

    @property
    def nut_center_z(self) -> float:
        return (
            self.pipe_outer_radius + self.bolt_axis_clearance + 0.5 * self.nut_thickness
        )

    @property
    def shaft_end_z(self) -> float:
        return (
            self.pipe_outer_radius
            + self.bolt_axis_clearance
            + self.nut_thickness
            + self.bolt_tail_extra
        )

    @property
    def shaft_length(self) -> float:
        return self.shaft_end_z - self.shaft_start_z

    @property
    def shaft_center_z(self) -> float:
        return 0.5 * (self.shaft_start_z + self.shaft_end_z)

    @property
    def head_center_z(self) -> float:
        return (
            -self.pipe_outer_radius
            - self.bolt_axis_clearance
            - 0.5 * self.bolt_head_thickness
        )

    @property
    def bolt_station_x(self) -> float:
        # +Y straight segment lies at x = bend_centerline_radius.
        return self.bend_centerline_radius

    @property
    def bolt_station_y(self) -> float:
        return self.bolt_station_y_offset

    @property
    def lid_center_x(self) -> float:
        # Cover lid is mounted at the far -X pipe opening.
        return -self.straight_len_x - 0.5 * self.lid_thickness

    @property
    def lid_center_y(self) -> float:
        # +X straight segment lies at y = -bend_centerline_radius.
        return -self.bend_centerline_radius


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------


def ensure_clean_dir(path: Path, *, clean: bool = False) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# OpenSCAD generation
# ---------------------------------------------------------------------


def make_modules_scad(p: BenchmarkParams) -> str:
    return f"""
// Auto-generated benchmark case.
// Units are millimetres.

pipe_outer_radius = {p.pipe_outer_radius};
pipe_inner_radius = {p.pipe_inner_radius};
bend_centerline_radius = {p.bend_centerline_radius};
straight_len_x = {p.straight_len_x};
straight_len_y = {p.straight_len_y};

bolt_station_x = {p.bolt_station_x};
bolt_station_y = {p.bolt_station_y};

side_hole_radius = {p.side_hole_radius};

bolt_shaft_radius = {p.bolt_shaft_radius};
bolt_head_radius = {p.bolt_head_radius};
bolt_head_thickness = {p.bolt_head_thickness};
shaft_start_z = {p.shaft_start_z};
shaft_end_z = {p.shaft_end_z};
shaft_length = {p.shaft_length};
shaft_center_z = {p.shaft_center_z};
head_center_z = {p.head_center_z};

nut_inner_radius = {p.nut_inner_radius};
nut_outer_radius = {p.nut_outer_radius};
nut_thickness = {p.nut_thickness};
nut_center_z = {p.nut_center_z};

lid_outer_radius = {p.lid_outer_radius};
lid_thickness = {p.lid_thickness};
lid_center_x = {p.lid_center_x};
lid_center_y = {p.lid_center_y};

eps = 0.05;

module quarter_torus(radius_major, radius_minor, fn_major={p.fn_pipe}, fn_minor={p.fn_part}) {{
    rotate_extrude(angle=90, convexity=10, $fn=fn_major)
        translate([radius_major, 0, 0])
            circle(r=radius_minor, $fn=fn_minor);
}}

module pipe_outer_solid() {{
    union() {{
        // Bend centerline: from (0, -R) to (R, 0).
        // This makes the +X and +Y straight segments tangent to the elbow.
        rotate([0, 0, -90])
            quarter_torus(bend_centerline_radius, pipe_outer_radius);

        // -X straight pipe segment.
        translate([-straight_len_x, -bend_centerline_radius, 0])
            rotate([0, 90, 0])
                cylinder(
                    h=straight_len_x,
                    r=pipe_outer_radius,
                    center=false,
                    $fn={p.fn_pipe}
                );

        // +Y straight pipe segment.
        translate([bend_centerline_radius, 0, 0])
            rotate([-90, 0, 0])
                cylinder(
                    h=straight_len_y,
                    r=pipe_outer_radius,
                    center=false,
                    $fn={p.fn_pipe}
                );
    }}
}}

module pipe_inner_void() {{
    union() {{
        rotate([0, 0, -90])
            quarter_torus(bend_centerline_radius, pipe_inner_radius);

        // Inner void for -X straight pipe.
        translate([-straight_len_x - eps, -bend_centerline_radius, 0])
            rotate([0, 90, 0])
                cylinder(
                    h=straight_len_x + 2 * eps,
                    r=pipe_inner_radius,
                    center=false,
                    $fn={p.fn_pipe}
                );

        // Inner void for +Y straight pipe.
        translate([bend_centerline_radius, -eps, 0])
            rotate([-90, 0, 0])
                cylinder(
                    h=straight_len_y + 2 * eps,
                    r=pipe_inner_radius,
                    center=false,
                    $fn={p.fn_pipe}
                );
    }}
}}

module bolt_side_holes() {{
    translate([bolt_station_x, bolt_station_y, 0])
        cylinder(h=4 * pipe_outer_radius, r=side_hole_radius, center=true, $fn={p.fn_part});
}}

module part_j_elbow_pipe() {{
    difference() {{
        pipe_outer_solid();
        pipe_inner_void();
        bolt_side_holes();
    }}
}}

module bolt_thread_marker() {{
    for (i = [0:4]) {{
        translate([
            bolt_station_x,
            bolt_station_y,
            nut_center_z - nut_thickness * 0.45 + i * nut_thickness * 0.18
        ])
            cylinder(
                h=0.22,
                r=bolt_shaft_radius + 0.18,
                center=true,
                $fn={p.fn_part}
            );
    }}
}}

module part_i_bolt() {{
    union() {{
        translate([bolt_station_x, bolt_station_y, shaft_center_z])
            cylinder(
                h=shaft_length,
                r=bolt_shaft_radius,
                center=true,
                $fn={p.fn_part}
            );

        translate([bolt_station_x, bolt_station_y, head_center_z])
            cylinder(
                h=bolt_head_thickness,
                r=bolt_head_radius,
                center=true,
                $fn={p.fn_part}
            );

        bolt_thread_marker();
    }}
}}

module part_k_nut() {{
    translate([bolt_station_x, bolt_station_y, nut_center_z])
        difference() {{
            cylinder(
                h=nut_thickness,
                r=nut_outer_radius,
                center=true,
                $fn=6
            );

            cylinder(
                h=nut_thickness + 2 * eps,
                r=nut_inner_radius,
                center=true,
                $fn={p.fn_part}
            );
        }}
}}

module part_l_cover_lid() {{
    translate([lid_center_x, lid_center_y, 0])
        rotate([0, 90, 0])
            cylinder(
                h=lid_thickness,
                r=lid_outer_radius,
                center=true,
                $fn={p.fn_pipe}
            );
}}

module assembly_colored() {{
    color([0.00, 0.52, 0.56, 0.42])
        part_j_elbow_pipe();

    color([0.93, 0.72, 0.05, 1.00])
        part_i_bolt();

    color([0.70, 0.22, 0.17, 1.00])
        part_k_nut();

    color([0.08, 0.27, 0.40, 0.55])
        part_l_cover_lid();
}}
"""


def make_part_scad(module_name: str) -> str:
    return f"""
include <case_modules.scad>;

{module_name}();
"""


def make_overview_scad() -> str:
    return """
include <case_modules.scad>;

assembly_colored();
"""


def run_openscad(scad_path: Path, stl_path: Path) -> bool:
    exe = shutil.which("openscad")
    if exe is None:
        print(
            "[WARN] OpenSCAD executable not found. SCAD files were generated, but STL export was skipped."
        )
        return False

    cmd = [exe, "-o", str(stl_path), str(scad_path)]
    print("[OpenSCAD]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return True


# ---------------------------------------------------------------------
# Standard Assembly export
# ---------------------------------------------------------------------


def build_parts() -> list[Part]:
    required = [
        STL_DIR / "part_i_bolt.stl",
        STL_DIR / "part_j_elbow_pipe.stl",
        STL_DIR / "part_k_nut.stl",
        STL_DIR / "part_l_cover_lid.stl",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required STL files:\n" + "\n".join(str(p) for p in missing)
        )

    return [
        Part("part_i_bolt", str(STL_DIR / "part_i_bolt.stl")),
        Part("part_j_elbow_pipe", str(STL_DIR / "part_j_elbow_pipe.stl")),
        Part("part_k_nut", str(STL_DIR / "part_k_nut.stl")),
        Part("part_l_cover_lid", str(STL_DIR / "part_l_cover_lid.stl")),
    ]


def build_assembly(params: BenchmarkParams) -> Assembly:
    parts = build_parts()
    asm = Assembly(
        name=params.case_name,
        parts=parts,
        contact_tollerance=params.contact_tollerance,
        dot_product_threshold=params.dot_product_threshold,
    )
    asm.find_contacts()
    return asm


# ---------------------------------------------------------------------
# Validation / metadata
# ---------------------------------------------------------------------


def validate_params(p: BenchmarkParams) -> dict:
    checks = {
        "pipe_wall_thickness": p.pipe_wall_thickness,
        "shaft_radius_smaller_than_side_hole": p.bolt_shaft_radius < p.side_hole_radius,
        "head_radius_larger_than_side_hole": p.bolt_head_radius > p.side_hole_radius,
        "nut_inner_radius_clears_shaft": p.nut_inner_radius > p.bolt_shaft_radius,
        "nut_outer_radius_larger_than_side_hole": p.nut_outer_radius
        > p.side_hole_radius,
        "shaft_length": p.shaft_length,
        "pipe_outer_diameter": p.pipe_outer_diameter,
        "shaft_has_margin_over_pipe_diameter": (
            p.shaft_length > p.pipe_outer_diameter + 2.0 * p.bolt_axis_clearance
        ),
        "lid_matches_pipe_outer_radius": abs(p.lid_outer_radius - p.pipe_outer_radius)
        < 1e-9,
        "bolt_station_inside_y_straight": 0.0
        < p.bolt_station_y_offset
        < p.straight_len_y,
        "bolt_station_on_y_straight_x": abs(p.bolt_station_x - p.bend_centerline_radius)
        < 1e-9,
        "lid_on_x_straight_y": abs(p.lid_center_y + p.bend_centerline_radius) < 1e-9,
        "lid_at_negative_x_opening": (
            abs(p.lid_center_x + p.straight_len_x + 0.5 * p.lid_thickness) < 1e-9
        ),
    }

    hard_fail = [
        key for key, value in checks.items() if isinstance(value, bool) and not value
    ]

    if hard_fail:
        raise ValueError(f"Invalid benchmark parameters: {hard_fail}")

    return checks


def write_metadata(params: BenchmarkParams, checks: dict, xml_path: Path) -> None:
    metadata = {
        "case_name": params.case_name,
        "output_root": ".",
        "parameters": asdict(params),
        "derived": {
            "pipe_outer_diameter": params.pipe_outer_diameter,
            "pipe_inner_diameter": params.pipe_inner_diameter,
            "pipe_wall_thickness": params.pipe_wall_thickness,
            "bolt_station": [params.bolt_station_x, params.bolt_station_y, 0.0],
            "shaft_start_z": params.shaft_start_z,
            "shaft_end_z": params.shaft_end_z,
            "shaft_length": params.shaft_length,
            "head_center_z": params.head_center_z,
            "nut_center_z": params.nut_center_z,
            "lid_center_x": params.lid_center_x,
            "lid_center_y": params.lid_center_y,
        },
        "validation": checks,
        "standard_export": {
            "xml_path": xml_path.relative_to(OUT_ROOT).as_posix(),
            "xml_generated_by": "Assembly.find_contacts() + Assembly.save_as_xml()",
            "stl_dir": STL_DIR.relative_to(OUT_ROOT).as_posix(),
            "scad_dir": SCAD_DIR.relative_to(OUT_ROOT).as_posix(),
        },
        "notes": [
            "part_i_bolt is a shaft plus an enlarged cylindrical head.",
            "part_j_elbow_pipe is a parameterized 90-degree elbow pipe with a transverse bolt hole.",
            "part_k_nut is represented as a ring/hex nut around the bolt tail.",
            "part_l_cover_lid is represented as a solid disk cover at the shorter -X pipe opening.",
            "This benchmark is intended for dependency-aware secondary feasibility analysis.",
            "The bolt thread is approximated by geometric markers; a future version may add a semantic bolt-and-nut capability.",
        ],
    }

    metadata_path = OUT_ROOT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------


def build_case(
    params: BenchmarkParams = BenchmarkParams(),
    *,
    export_stl: bool = True,
    clean: bool = False,
) -> None:
    ensure_clean_dir(OUT_ROOT, clean=clean)
    ensure_clean_dir(SCAD_DIR)
    ensure_clean_dir(STL_DIR)
    ensure_clean_dir(XML_DIR)

    checks = validate_params(params)

    write_text(SCAD_DIR / "case_modules.scad", make_modules_scad(params))
    write_text(SCAD_DIR / "overview.scad", make_overview_scad())

    part_modules = {
        "part_i_bolt": "part_i_bolt",
        "part_j_elbow_pipe": "part_j_elbow_pipe",
        "part_k_nut": "part_k_nut",
        "part_l_cover_lid": "part_l_cover_lid",
    }

    for part_name, module_name in part_modules.items():
        write_text(SCAD_DIR / f"{part_name}.scad", make_part_scad(module_name))

    if export_stl:
        for part_name in part_modules:
            run_openscad(
                SCAD_DIR / f"{part_name}.scad",
                STL_DIR / f"{part_name}.stl",
            )

    asm = build_assembly(params)
    xml_path = XML_DIR / f"{params.case_name}.xml"
    asm.save_as_xml(str(xml_path))

    write_metadata(params, checks, xml_path)

    print("=" * 80)
    print("[DONE] Standard benchmark case generated")
    print(f"case_root: {OUT_ROOT}")
    print(f"scad_dir:  {SCAD_DIR}")
    print(f"stl_dir:   {STL_DIR}")
    print(f"xml_path:  {xml_path}")
    print(f"metadata:  {OUT_ROOT / 'metadata.json'}")
    print("=" * 80)
    print("[KEY CHECKS]")
    for key, value in checks.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    build_case(clean=True)
