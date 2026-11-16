from __future__ import annotations

import copy
import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PROJECT_ROOT = ROOT.parents[2]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part

ASSEMBLY = ROOT / "benchmark_ring_case_assembly"
RESULTS = ROOT / "benchmark_ring_case_results"
SCAD = ASSEMBLY / "scad"
STL = ASSEMBLY / "stl"
XML = ASSEMBLY / "xml"

SCAD.mkdir(parents=True, exist_ok=True)
STL.mkdir(parents=True, exist_ok=True)
XML.mkdir(parents=True, exist_ok=True)


@dataclass
class VisibilityCaseConfig:
    # Opening geometry
    hole_radius: float = 3.0

    # Housing geometry
    cylinder_radius: float = 20.0
    cylinder_height: float = 60.0
    wall_thickness: float = 2.0
    cap_thickness: float = 2.0

    # Rod geometry
    rod_radius: float = 2.0
    rod_height: float = 50.0

    # Ring geometry
    ring_outer_radius: float = 8.0
    ring_inner_radius: float = 4.0
    ring_thickness: float = 3.0
    ring_z: float = 30.0

    # Contact / connection detection
    contact_tollerance: float = 0.5
    dot_product_threshold: float = -0.95

    # Observation setup
    viewing_directions: list[list[float]] = field(
        default_factory=lambda: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )

    @property
    def hole_diameter(self) -> float:
        return 2.0 * self.hole_radius

    @property
    def ring_top_z(self) -> float:
        return self.ring_z + self.ring_thickness

    @property
    def depth_from_opening_plane(self) -> float:
        """
        Distance from the ring top surface to the opening plane at z = cylinder_height.
        Larger value means the ring is deeper inside the housing.
        """
        return self.cylinder_height - self.ring_top_z


def clone_config(cfg: VisibilityCaseConfig) -> VisibilityCaseConfig:
    """
    Deep-copy config so mutable fields like viewing_directions are not shared.
    """
    return copy.deepcopy(cfg)


def run_openscad(scad_path: Path, stl_path: Path) -> None:
    subprocess.run(
        ["openscad", "-o", str(stl_path), str(scad_path)],
        check=True,
    )


def check_required_scad_files() -> None:
    required = [
        SCAD / "outer_shell.scad",
        SCAD / "bottom_cap_with_rod.scad",
        SCAD / "ring.scad",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required SCAD files:\n" + "\n".join(str(p) for p in missing)
        )


def write_common_params_scad(cfg: VisibilityCaseConfig) -> Path:
    scad_path = SCAD / "common_params.scad"
    scad_text = f"""
cylinder_radius = {cfg.cylinder_radius:.6f};
cylinder_height = {cfg.cylinder_height:.6f};
wall_thickness = {cfg.wall_thickness:.6f};

cap_thickness = {cfg.cap_thickness:.6f};

rod_radius = {cfg.rod_radius:.6f};
rod_height = {cfg.rod_height:.6f};

ring_outer_radius = {cfg.ring_outer_radius:.6f};
ring_inner_radius = {cfg.ring_inner_radius:.6f};
ring_thickness = {cfg.ring_thickness:.6f};
ring_z = {cfg.ring_z:.6f};

$fn = 128;
""".strip()
    scad_path.write_text(scad_text)
    return scad_path


def export_static_parts(cfg: VisibilityCaseConfig) -> None:
    """
    Export the parts whose geometry depends on the common parameter file.
    """
    write_common_params_scad(cfg)

    run_openscad(SCAD / "outer_shell.scad", STL / "outer_shell.stl")
    run_openscad(SCAD / "bottom_cap_with_rod.scad", STL / "bottom_cap_with_rod.stl")
    run_openscad(SCAD / "ring.scad", STL / "ring.stl")


def write_top_cap_scad(cfg: VisibilityCaseConfig) -> Path:
    scad_path = SCAD / f"top_cap_with_hole_r{cfg.hole_radius:.2f}.scad"

    scad_text = f"""
include <common_params.scad>;

hole_radius = {cfg.hole_radius:.6f};

translate([0, 0, cylinder_height])
difference() {{
    cylinder(h = cap_thickness, r = cylinder_radius);
    translate([0, 0, -0.1])
        cylinder(h = cap_thickness + 0.2, r = hole_radius);
}}
""".strip()

    scad_path.write_text(scad_text)
    return scad_path


def export_top_cap(cfg: VisibilityCaseConfig) -> Path:
    scad_path = write_top_cap_scad(cfg)
    stl_path = STL / f"top_cap_with_hole_r{cfg.hole_radius:.2f}.stl"
    run_openscad(scad_path, stl_path)
    return stl_path


def build_parts(cfg: VisibilityCaseConfig) -> list[Part]:
    top_stl = STL / f"top_cap_with_hole_r{cfg.hole_radius:.2f}.stl"

    required = [
        STL / "outer_shell.stl",
        STL / "bottom_cap_with_rod.stl",
        STL / "ring.stl",
        top_stl,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required STL files:\n" + "\n".join(str(p) for p in missing)
        )

    return [
        Part("outer_shell", str(STL / "outer_shell.stl")),
        Part("bottom_cap_with_rod", str(STL / "bottom_cap_with_rod.stl")),
        Part("ring", str(STL / "ring.stl")),
        Part("top_cap_with_hole", str(top_stl)),
    ]


def build_assembly(cfg: VisibilityCaseConfig) -> Assembly:
    parts = build_parts(cfg)
    asm = Assembly(
        name=f"visibility_ring_case_r{cfg.hole_radius:.2f}",
        parts=parts,
        contact_tollerance=cfg.contact_tollerance,
        dot_product_threshold=cfg.dot_product_threshold,
    )
    asm.find_contacts()
    return asm


def evaluate_visibility(asm: Assembly, cfg: VisibilityCaseConfig) -> bool:
    ring = asm.get_part_by_name("ring")
    if ring is None:
        raise ValueError("Part 'ring' not found in assembly.")

    return asm.check_visibility(
        ring,
        viewing_directions=cfg.viewing_directions,
        debug_visualization=False,
    )


def build_case(
    cfg: VisibilityCaseConfig, save_xml: bool = True
) -> tuple[Assembly, bool, Path | None]:
    """
    Full pipeline for one case:
    1) write common params
    2) export static parts
    3) export top cap
    4) build assembly
    5) evaluate visibility
    6) optionally save XML
    """
    write_common_params_scad(cfg)
    export_static_parts(cfg)
    export_top_cap(cfg)

    asm = build_assembly(cfg)
    visible = evaluate_visibility(asm, cfg)

    xml_path = None
    if save_xml:
        xml_path = XML / f"visibility_ring_case_r{cfg.hole_radius:.2f}.xml"
        asm.save_as_xml(str(xml_path))

    return asm, visible, xml_path


def find_visibility_threshold(
    base_cfg: VisibilityCaseConfig,
    candidate_hole_radii: list[float],
    save_xml: bool = False,
) -> float | None:
    """
    Returns the smallest hole radius for which the ring becomes visible.
    """
    for hole_radius in candidate_hole_radii:
        cfg = clone_config(base_cfg)
        cfg.hole_radius = hole_radius

        _, visible, _ = build_case(cfg, save_xml=save_xml)
        if visible:
            return hole_radius

    return None


def run_hole_radius_sweep(
    base_cfg: VisibilityCaseConfig,
    hole_radii: list[float],
    csv_path: Path,
    save_xml: bool = True,
) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "hole_radius",
                "hole_diameter",
                "ring_outer_radius",
                "ring_inner_radius",
                "ring_thickness",
                "ring_z",
                "depth_from_opening_plane",
                "ring_visible",
            ]
        )

        for hole_radius in hole_radii:
            cfg = clone_config(base_cfg)
            cfg.hole_radius = hole_radius

            print(
                f"\n=== hole_radius = {cfg.hole_radius:.2f}, "
                f"hole_diameter = {cfg.hole_diameter:.2f} ==="
            )

            _, visible, xml_path = build_case(cfg, save_xml=save_xml)

            writer.writerow(
                [
                    cfg.hole_radius,
                    cfg.hole_diameter,
                    cfg.ring_outer_radius,
                    cfg.ring_inner_radius,
                    cfg.ring_thickness,
                    cfg.ring_z,
                    cfg.depth_from_opening_plane,
                    int(visible),
                ]
            )

            if xml_path is not None:
                print(f"Saved XML to {xml_path}")
            print(f"Ring visible: {visible}")

    print(f"\nSaved sweep results to {csv_path}")


def run_threshold_experiment(
    base_cfg: VisibilityCaseConfig,
    candidate_hole_radii: list[float],
    csv_path: Path,
) -> None:
    """
    Writes one row with the threshold for the given base configuration.
    Useful when you later sweep ring_z, ring size, or view config outside this function.
    """
    threshold = find_visibility_threshold(
        base_cfg, candidate_hole_radii, save_xml=False
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ring_outer_radius",
                "ring_inner_radius",
                "ring_thickness",
                "ring_z",
                "depth_from_opening_plane",
                "threshold_hole_radius",
                "threshold_hole_diameter",
            ]
        )
        writer.writerow(
            [
                base_cfg.ring_outer_radius,
                base_cfg.ring_inner_radius,
                base_cfg.ring_thickness,
                base_cfg.ring_z,
                base_cfg.depth_from_opening_plane,
                "" if threshold is None else threshold,
                "" if threshold is None else 2.0 * threshold,
            ]
        )

    print(f"Saved threshold result to {csv_path}")
    print(f"Threshold hole radius: {threshold}")


def run_threshold_vs_ring_z_experiment(
    base_cfg: VisibilityCaseConfig,
    ring_z_values: list[float],
    candidate_hole_radii: list[float],
    csv_path: Path,
) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ring_z",
                "ring_top_z",
                "depth_from_opening_plane",
                "ring_outer_radius",
                "ring_inner_radius",
                "ring_thickness",
                "threshold_hole_radius",
                "threshold_hole_diameter",
            ]
        )

        for ring_z in ring_z_values:
            cfg = clone_config(base_cfg)
            cfg.ring_z = ring_z

            print(
                f"\n=== ring_z = {cfg.ring_z:.2f}, "
                f"depth = {cfg.depth_from_opening_plane:.2f} ==="
            )

            threshold = find_visibility_threshold(
                base_cfg=cfg,
                candidate_hole_radii=candidate_hole_radii,
                save_xml=False,
            )

            writer.writerow(
                [
                    cfg.ring_z,
                    cfg.ring_top_z,
                    cfg.depth_from_opening_plane,
                    cfg.ring_outer_radius,
                    cfg.ring_inner_radius,
                    cfg.ring_thickness,
                    "" if threshold is None else threshold,
                    "" if threshold is None else 2.0 * threshold,
                ]
            )

            print(f"Threshold hole radius: {threshold}")

    print(f"\nSaved threshold-vs-ring_z results to {csv_path}")


def run_threshold_vs_ring_geometry_experiment(
    base_cfg: VisibilityCaseConfig,
    inner_radius_values: list[float],
    radial_width_values: list[float],
    thickness_values: list[float],
    candidate_hole_radii: list[float],
    target_depth: float,
    csv_path: Path,
) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ring_inner_radius",
                "radial_width",
                "ring_outer_radius",
                "ring_thickness",
                "ring_z",
                "depth_from_opening_plane",
                "threshold_hole_radius",
                "threshold_hole_diameter",
            ]
        )

        for inner_radius in inner_radius_values:
            for radial_width in radial_width_values:
                for thickness in thickness_values:
                    outer_radius = inner_radius + radial_width

                    cfg = clone_config(base_cfg)
                    cfg.ring_inner_radius = inner_radius
                    cfg.ring_outer_radius = outer_radius
                    cfg.ring_thickness = thickness

                    # keep depth fixed
                    cfg.ring_z = cfg.cylinder_height - target_depth - cfg.ring_thickness

                    print(
                        f"\n=== inner = {cfg.ring_inner_radius:.2f}, "
                        f"width = {radial_width:.2f}, "
                        f"outer = {cfg.ring_outer_radius:.2f}, "
                        f"thickness = {cfg.ring_thickness:.2f}, "
                        f"depth = {cfg.depth_from_opening_plane:.2f} ==="
                    )

                    threshold = find_visibility_threshold(
                        base_cfg=cfg,
                        candidate_hole_radii=candidate_hole_radii,
                        save_xml=False,
                    )

                    writer.writerow(
                        [
                            cfg.ring_inner_radius,
                            radial_width,
                            cfg.ring_outer_radius,
                            cfg.ring_thickness,
                            cfg.ring_z,
                            cfg.depth_from_opening_plane,
                            "" if threshold is None else threshold,
                            "" if threshold is None else 2.0 * threshold,
                        ]
                    )

                    print(f"Threshold hole radius: {threshold}")

    print(f"\nSaved threshold-vs-ring-geometry results to {csv_path}")


def run_threshold_vs_view_angle_experiment(
    base_cfg: VisibilityCaseConfig,
    view_configs: dict,
    candidate_hole_radii: list[float],
    csv_path: Path,
) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "view_name",
                "view_angle_deg",
                "viewing_directions",
                "ring_outer_radius",
                "ring_inner_radius",
                "ring_thickness",
                "ring_z",
                "depth_from_opening_plane",
                "threshold_hole_radius",
                "threshold_hole_diameter",
            ]
        )

        for view_name, view_data in view_configs.items():
            cfg = clone_config(base_cfg)
            cfg.viewing_directions = view_data["directions"]

            print(f"\n=== view = {view_name}, angle = {view_data['angle_deg']} deg ===")

            threshold = find_visibility_threshold(
                base_cfg=cfg,
                candidate_hole_radii=candidate_hole_radii,
                save_xml=False,
            )

            writer.writerow(
                [
                    view_name,
                    view_data["angle_deg"],
                    cfg.viewing_directions,
                    cfg.ring_outer_radius,
                    cfg.ring_inner_radius,
                    cfg.ring_thickness,
                    cfg.ring_z,
                    cfg.depth_from_opening_plane,
                    "" if threshold is None else threshold,
                    "" if threshold is None else 2.0 * threshold,
                ]
            )

            print(f"Threshold hole radius: {threshold}")

    print(f"\nSaved threshold-vs-view-angle results to {csv_path}")


def run_critical_angle_vs_ring_inner_experiment(
    base_cfg: VisibilityCaseConfig,
    inner_radius_values: list[float],
    radial_width: float,
    ring_thickness: float,
    target_depth: float,
    view_configs: dict,
    candidate_hole_radii: list[float],
    csv_path: Path,
) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "view_name",
                "view_angle_deg",
                "viewing_directions",
                "ring_inner_radius",
                "radial_width",
                "ring_outer_radius",
                "ring_thickness",
                "ring_z",
                "depth_from_opening_plane",
                "threshold_hole_radius",
                "threshold_hole_diameter",
            ]
        )

        for inner_radius in inner_radius_values:
            for view_name, view_data in view_configs.items():
                cfg = clone_config(base_cfg)

                cfg.ring_inner_radius = inner_radius
                cfg.ring_outer_radius = inner_radius + radial_width
                cfg.ring_thickness = ring_thickness

                # Keep depth fixed while changing thickness/geometry.
                cfg.ring_z = cfg.cylinder_height - target_depth - cfg.ring_thickness

                cfg.viewing_directions = view_data["directions"]

                print(
                    f"\n=== inner = {cfg.ring_inner_radius:.2f}, "
                    f"outer = {cfg.ring_outer_radius:.2f}, "
                    f"angle = {view_data['angle_deg']} deg, "
                    f"depth = {cfg.depth_from_opening_plane:.2f} ==="
                )

                threshold = find_visibility_threshold(
                    base_cfg=cfg,
                    candidate_hole_radii=candidate_hole_radii,
                    save_xml=False,
                )

                writer.writerow(
                    [
                        view_name,
                        view_data["angle_deg"],
                        cfg.viewing_directions,
                        cfg.ring_inner_radius,
                        radial_width,
                        cfg.ring_outer_radius,
                        cfg.ring_thickness,
                        cfg.ring_z,
                        cfg.depth_from_opening_plane,
                        "" if threshold is None else threshold,
                        "" if threshold is None else 2.0 * threshold,
                    ]
                )

                print(f"Threshold hole radius: {threshold}")

    print(f"\nSaved critical-angle-vs-ring-inner results to {csv_path}")


def run_fixed_outer_radius_inner_sweep_experiment(
    base_cfg: VisibilityCaseConfig,
    inner_radius_values: list[float],
    fixed_outer_radius: float,
    ring_thickness: float,
    target_depth: float,
    view_configs: dict,
    candidate_hole_radii: list[float],
    csv_path: Path,
) -> None:
    """
    Experiment A:
    Keep outer radius fixed and vary inner radius.
    This separates the effect of central void size from total target footprint.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "view_name",
                "view_angle_deg",
                "viewing_directions",
                "ring_inner_radius",
                "radial_width",
                "ring_outer_radius",
                "ring_thickness",
                "ring_z",
                "depth_from_opening_plane",
                "threshold_hole_radius",
                "threshold_hole_diameter",
            ]
        )

        for inner_radius in inner_radius_values:
            if inner_radius >= fixed_outer_radius:
                print(
                    f"Skipping invalid geometry: inner={inner_radius}, "
                    f"outer={fixed_outer_radius}"
                )
                continue

            for view_name, view_data in view_configs.items():
                cfg = clone_config(base_cfg)

                cfg.ring_inner_radius = inner_radius
                cfg.ring_outer_radius = fixed_outer_radius
                cfg.ring_thickness = ring_thickness
                cfg.ring_z = cfg.cylinder_height - target_depth - cfg.ring_thickness
                cfg.viewing_directions = view_data["directions"]

                radial_width = cfg.ring_outer_radius - cfg.ring_inner_radius

                print(
                    f"\n[Fixed outer] inner={cfg.ring_inner_radius:.2f}, "
                    f"outer={cfg.ring_outer_radius:.2f}, "
                    f"width={radial_width:.2f}, "
                    f"angle={view_data['angle_deg']} deg"
                )

                threshold = find_visibility_threshold(
                    base_cfg=cfg,
                    candidate_hole_radii=candidate_hole_radii,
                    save_xml=False,
                )

                writer.writerow(
                    [
                        view_name,
                        view_data["angle_deg"],
                        cfg.viewing_directions,
                        cfg.ring_inner_radius,
                        radial_width,
                        cfg.ring_outer_radius,
                        cfg.ring_thickness,
                        cfg.ring_z,
                        cfg.depth_from_opening_plane,
                        "" if threshold is None else threshold,
                        "" if threshold is None else 2.0 * threshold,
                    ]
                )

                print(f"Threshold hole radius: {threshold}")

    print(f"\nSaved fixed-outer inner sweep results to {csv_path}")


def run_fixed_inner_radial_width_sweep_experiment(
    base_cfg: VisibilityCaseConfig,
    fixed_inner_radius: float,
    radial_width_values: list[float],
    ring_thickness: float,
    target_depth: float,
    view_configs: dict,
    candidate_hole_radii: list[float],
    csv_path: Path,
) -> None:
    """
    Experiment B:
    Keep inner radius fixed and vary radial width / outer radius.
    This tests whether larger physical target extent expands the visibility cone.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "view_name",
                "view_angle_deg",
                "viewing_directions",
                "ring_inner_radius",
                "radial_width",
                "ring_outer_radius",
                "ring_thickness",
                "ring_z",
                "depth_from_opening_plane",
                "threshold_hole_radius",
                "threshold_hole_diameter",
            ]
        )

        for radial_width in radial_width_values:
            for view_name, view_data in view_configs.items():
                cfg = clone_config(base_cfg)

                cfg.ring_inner_radius = fixed_inner_radius
                cfg.ring_outer_radius = fixed_inner_radius + radial_width
                cfg.ring_thickness = ring_thickness
                cfg.ring_z = cfg.cylinder_height - target_depth - cfg.ring_thickness
                cfg.viewing_directions = view_data["directions"]

                print(
                    f"\n[Fixed inner] inner={cfg.ring_inner_radius:.2f}, "
                    f"outer={cfg.ring_outer_radius:.2f}, "
                    f"width={radial_width:.2f}, "
                    f"angle={view_data['angle_deg']} deg"
                )

                threshold = find_visibility_threshold(
                    base_cfg=cfg,
                    candidate_hole_radii=candidate_hole_radii,
                    save_xml=False,
                )

                writer.writerow(
                    [
                        view_name,
                        view_data["angle_deg"],
                        cfg.viewing_directions,
                        cfg.ring_inner_radius,
                        radial_width,
                        cfg.ring_outer_radius,
                        cfg.ring_thickness,
                        cfg.ring_z,
                        cfg.depth_from_opening_plane,
                        "" if threshold is None else threshold,
                        "" if threshold is None else 2.0 * threshold,
                    ]
                )

                print(f"Threshold hole radius: {threshold}")

    print(f"\nSaved fixed-inner radial-width sweep results to {csv_path}")


def run_depth_under_oblique_views_experiment(
    base_cfg: VisibilityCaseConfig,
    ring_z_values: list[float],
    view_configs: dict,
    candidate_hole_radii: list[float],
    csv_path: Path,
) -> None:
    """
    Experiment C:
    Vary target depth under selected oblique views.
    This tests whether depth becomes relevant once the observation direction is no longer top-down.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "view_name",
                "view_angle_deg",
                "viewing_directions",
                "ring_z",
                "ring_top_z",
                "depth_from_opening_plane",
                "ring_inner_radius",
                "ring_outer_radius",
                "ring_thickness",
                "threshold_hole_radius",
                "threshold_hole_diameter",
            ]
        )

        for ring_z in ring_z_values:
            for view_name, view_data in view_configs.items():
                cfg = clone_config(base_cfg)

                cfg.ring_z = ring_z
                cfg.viewing_directions = view_data["directions"]

                print(
                    f"\n[Depth under oblique] ring_z={cfg.ring_z:.2f}, "
                    f"depth={cfg.depth_from_opening_plane:.2f}, "
                    f"angle={view_data['angle_deg']} deg"
                )

                threshold = find_visibility_threshold(
                    base_cfg=cfg,
                    candidate_hole_radii=candidate_hole_radii,
                    save_xml=False,
                )

                writer.writerow(
                    [
                        view_name,
                        view_data["angle_deg"],
                        cfg.viewing_directions,
                        cfg.ring_z,
                        cfg.ring_top_z,
                        cfg.depth_from_opening_plane,
                        cfg.ring_inner_radius,
                        cfg.ring_outer_radius,
                        cfg.ring_thickness,
                        "" if threshold is None else threshold,
                        "" if threshold is None else 2.0 * threshold,
                    ]
                )

                print(f"Threshold hole radius: {threshold}")

    print(f"\nSaved depth-under-oblique results to {csv_path}")


# def main() -> None:
#    check_required_scad_files()
#
#    base_cfg = VisibilityCaseConfig(
#        hole_radius=4.0,
#        cylinder_radius=20.0,
#        cylinder_height=60.0,
#        wall_thickness=2.0,
#        cap_thickness=2.0,
#        rod_radius=2.0,
#        rod_height=50.0,
#        ring_outer_radius=8.0,
#        ring_inner_radius=4.0,
#        ring_thickness=3.0,
#        ring_z=30.0,
#        contact_tollerance=0.5,
#        dot_product_threshold=-0.95,
#        viewing_directions=[
#            [0, 0, 1]
#        ],
#    )
#
#    # hole_radii = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
#    #candidate_hole_radii = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
##
#    ## 1) Raw visibility sweep
#    #visibility_csv = RESULTS / "visibility_results_hole_radii_3_4.csv"
#    #run_hole_radius_sweep(
#    #    base_cfg=base_cfg,
#    #    hole_radii=candidate_hole_radii,
#    #    csv_path=visibility_csv,
#    #    save_xml=True,
#    #)
##
#    ## 2) Threshold for the current base configuration
#    #threshold_csv = RESULTS / "threshold_result_single_config_hole_radii_3_4.csv"
#    #run_threshold_experiment(
#    #    base_cfg=base_cfg,
#    #    candidate_hole_radii=candidate_hole_radii,
#    #    csv_path=threshold_csv,
#    #)
#
#    #ring_z_values = [15, 20, 25, 30, 35, 40]
#    #candidate_hole_radii = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
##
#    #threshold_vs_ring_z_csv = RESULTS / "threshold_vs_ring_z_top_view_only.csv"
#    #run_threshold_vs_ring_z_experiment(
#    #    base_cfg=base_cfg,
#    #    ring_z_values=ring_z_values,
#    #    candidate_hole_radii=candidate_hole_radii,
#    #    csv_path=threshold_vs_ring_z_csv,
#    #)
#    target_depth = 27.0
#
#    inner_radius_values = [2.0, 4.0, 5.0]
#    radial_width_values = [1.0, 3.0]
#    thickness_values = [1.0, 3.0]
#
#    candidate_hole_radii = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
#
#    threshold_vs_ring_geometry_csv = RESULTS / "threshold_vs_ring_geometry_top_view_only.csv"
#    run_threshold_vs_ring_geometry_experiment(
#        base_cfg=base_cfg,
#        inner_radius_values=inner_radius_values,
#        radial_width_values=radial_width_values,
#        thickness_values=thickness_values,
#        candidate_hole_radii=candidate_hole_radii,
#        target_depth=target_depth,
#        csv_path=threshold_vs_ring_geometry_csv,
#    )
#
# def main() -> None:
#    check_required_scad_files()
#
#    base_cfg = VisibilityCaseConfig(
#        hole_radius=4.0,
#        cylinder_radius=20.0,
#        cylinder_height=60.0,
#        wall_thickness=2.0,
#        cap_thickness=2.0,
#        rod_radius=2.0,
#        rod_height=50.0,
#        ring_outer_radius=8.0,
#        ring_inner_radius=4.0,
#        ring_thickness=3.0,
#        ring_z=30.0,
#        contact_tollerance=0.5,
#        dot_product_threshold=-0.95,
#        viewing_directions=[[0.0, 0.0, 1.0]],
#    )
#
#    #view_configs = {
#    #    "top_0": {
#    #        "angle_deg": 0,
#    #        "directions": [[0.0, 0.0, 1.0]],
#    #    },
#    #    "oblique_15": {
#    #        "angle_deg": 15,
#    #        "directions": [[0.2588, 0.0, 0.9659]],
#    #    },
#    #    "oblique_30": {
#    #        "angle_deg": 30,
#    #        "directions": [[0.5, 0.0, 0.8660]],
#    #    },
#    #    "oblique_45": {
#    #        "angle_deg": 45,
#    #        "directions": [[0.7071, 0.0, 0.7071]],
#    #    },
#    #    "oblique_60": {
#    #    "angle_deg": 60,
#    #    "directions": [[0.8660, 0.0, 0.5]],
#    #    },
#    #    "oblique_75": {
#    #        "angle_deg": 75,
#    #        "directions": [[0.9659, 0.0, 0.2588]],
#    #    },
#    #}
#    view_configs = {
#        "oblique_20": {
#            "angle_deg": 20,
#            "directions": [[0.3420, 0.0, 0.9397]],
#        },
#
#        "oblique_25": {
#            "angle_deg": 25,
#            "directions": [[0.4226, 0.0, 0.9063]],
#        },
#
#        "oblique_30": {
#            "angle_deg": 30,
#            "directions": [[0.5000, 0.0, 0.8660]],
#        },
#
#        "oblique_35": {
#            "angle_deg": 35,
#            "directions": [[0.5736, 0.0, 0.8192]],
#        },
#
#        "oblique_40": {
#            "angle_deg": 40,
#            "directions": [[0.6428, 0.0, 0.7660]],
#        },
#    }
#
#    #candidate_hole_radii = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]
#    candidate_hole_radii = [
#        3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0
#    ]
#
#    threshold_vs_view_csv = RESULTS / "threshold_vs_view_angle_critical_region.csv"
#    run_threshold_vs_view_angle_experiment(
#        base_cfg=base_cfg,
#        view_configs=view_configs,
#        candidate_hole_radii=candidate_hole_radii,
#        csv_path=threshold_vs_view_csv,
#    )
# def main() -> None:
#    check_required_scad_files()
#
#    base_cfg = VisibilityCaseConfig(
#        hole_radius=4.0,
#        cylinder_radius=20.0,
#        cylinder_height=60.0,
#        wall_thickness=2.0,
#        cap_thickness=2.0,
#        rod_radius=2.0,
#        rod_height=50.0,
#        ring_outer_radius=8.0,
#        ring_inner_radius=4.0,
#        ring_thickness=3.0,
#        ring_z=30.0,
#        contact_tollerance=0.5,
#        dot_product_threshold=-0.95,
#        viewing_directions=[[0.0, 0.0, 1.0]],
#    )
#
#    view_configs = {
#        "oblique_20": {
#            "angle_deg": 20,
#            "directions": [[0.3420, 0.0, 0.9397]],
#        },
#        "oblique_25": {
#            "angle_deg": 25,
#            "directions": [[0.4226, 0.0, 0.9063]],
#        },
#        "oblique_30": {
#            "angle_deg": 30,
#            "directions": [[0.5000, 0.0, 0.8660]],
#        },
#        "oblique_35": {
#            "angle_deg": 35,
#            "directions": [[0.5736, 0.0, 0.8192]],
#        },
#    }
#
#    inner_radius_values = [2.0, 4.0, 6.0]
#    radial_width = 3.0
#    ring_thickness = 3.0
#    target_depth = 27.0
#
#    candidate_hole_radii = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
#
#    csv_path = RESULTS / "critical_angle_vs_ring_inner_radius.csv"
#    run_critical_angle_vs_ring_inner_experiment(
#        base_cfg=base_cfg,
#        inner_radius_values=inner_radius_values,
#        radial_width=radial_width,
#        ring_thickness=ring_thickness,
#        target_depth=target_depth,
#        view_configs=view_configs,
#        candidate_hole_radii=candidate_hole_radii,
#        csv_path=csv_path,
#    )


def main() -> None:
    check_required_scad_files()

    base_cfg = VisibilityCaseConfig(
        hole_radius=4.0,
        cylinder_radius=20.0,
        cylinder_height=60.0,
        wall_thickness=2.0,
        cap_thickness=2.0,
        rod_radius=2.0,
        rod_height=50.0,
        ring_outer_radius=8.0,
        ring_inner_radius=4.0,
        ring_thickness=3.0,
        ring_z=30.0,
        contact_tollerance=0.5,
        dot_product_threshold=-0.95,
        viewing_directions=[[0.0, 0.0, 1.0]],
    )

    # Critical angular region identified from previous experiments.
    critical_view_configs = {
        "oblique_20": {
            "angle_deg": 20,
            "directions": [[0.3420, 0.0, 0.9397]],
        },
        "oblique_25": {
            "angle_deg": 25,
            "directions": [[0.4226, 0.0, 0.9063]],
        },
        "oblique_30": {
            "angle_deg": 30,
            "directions": [[0.5000, 0.0, 0.8660]],
        },
        "oblique_35": {
            "angle_deg": 35,
            "directions": [[0.5736, 0.0, 0.8192]],
        },
    }

    # Keep the candidate range wide enough for oblique cases.
    candidate_hole_radii = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]

    # Shared target placement for geometry experiments.
    target_depth = 27.0
    ring_thickness = 3.0

    # ------------------------------------------------------------
    # Experiment A:
    # Fixed outer radius, varying inner radius.
    # Validates whether inner radius matters independently of
    # the overall target footprint.
    # Cases: 3 inner radii × 4 angles × 9 holes = 108 cases
    # ------------------------------------------------------------
    run_fixed_outer_radius_inner_sweep_experiment(
        base_cfg=base_cfg,
        inner_radius_values=[2.0, 4.0, 6.0],
        fixed_outer_radius=9.0,
        ring_thickness=ring_thickness,
        target_depth=target_depth,
        view_configs=critical_view_configs,
        candidate_hole_radii=candidate_hole_radii,
        csv_path=RESULTS / "critical_angle_fixed_outer_vary_inner.csv",
    )

    # ------------------------------------------------------------
    # Experiment B:
    # Fixed inner radius, varying radial width / outer radius.
    # Validates whether larger physical target extent expands
    # the effective visibility cone under oblique views.
    # Cases: 3 widths × 4 angles × 9 holes = 108 cases
    # ------------------------------------------------------------
    run_fixed_inner_radial_width_sweep_experiment(
        base_cfg=base_cfg,
        fixed_inner_radius=4.0,
        radial_width_values=[1.0, 3.0, 5.0],
        ring_thickness=ring_thickness,
        target_depth=target_depth,
        view_configs=critical_view_configs,
        candidate_hole_radii=candidate_hole_radii,
        csv_path=RESULTS / "critical_angle_fixed_inner_vary_radial_width.csv",
    )

    # ------------------------------------------------------------
    # Experiment C:
    # Depth sweep under selected oblique views.
    # Validates whether depth becomes influential once the
    # observation direction is no longer top-down.
    # Keep this smaller to avoid runtime explosion.
    # Cases: 3 depths × 2 angles × 9 holes = 54 cases
    # ------------------------------------------------------------
    depth_view_configs = {
        "oblique_20": critical_view_configs["oblique_20"],
        "oblique_30": critical_view_configs["oblique_30"],
    }

    run_depth_under_oblique_views_experiment(
        base_cfg=base_cfg,
        ring_z_values=[20.0, 30.0, 40.0],
        view_configs=depth_view_configs,
        candidate_hole_radii=candidate_hole_radii,
        csv_path=RESULTS / "depth_effect_under_oblique_views.csv",
    )


if __name__ == "__main__":
    main()
