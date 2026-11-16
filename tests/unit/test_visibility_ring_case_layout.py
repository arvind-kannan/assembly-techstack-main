"""The visibility ring case keeps each file once, and its build and plot scripts find them."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tests" / "internal_cases" / "visibility_ring_case"
ASSEMBLY = CASE / "benchmark_ring_case_assembly"
RESULTS = CASE / "benchmark_ring_case_results"


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def test_case_files_exist_only_in_benchmark_folders():
    assert not any((CASE / d).exists() for d in ("scad", "stl", "xml"))
    assert not list(CASE.glob("*.csv"))
    assert not (CASE / "plot_visibility_margin_results.py").exists()
    assert not (ROOT / "tools" / "plot_visibility_margin_results.py").exists()


def test_build_case_reads_and_writes_benchmark_folders():
    build_case = load(CASE / "build_case.py")
    assert (build_case.SCAD, build_case.STL, build_case.XML) == (
        ASSEMBLY / "scad",
        ASSEMBLY / "stl",
        ASSEMBLY / "xml",
    )
    assert build_case.RESULTS == RESULTS
    build_case.check_required_scad_files()


def test_plot_script_reads_existing_result_csvs():
    plot = load(ROOT / "tools" / "plotting_tool" / "plot_visibility_margin_results.py")
    csvs = [v for k, v in vars(plot).items() if k.startswith("CSV_")]
    assert csvs and all(p.parent == RESULTS and p.exists() for p in csvs), csvs


def test_render_script_points_at_benchmark_xml():
    text = (
        ROOT / "tools" / "rendering_tool" / "render_visibility_ring_cases.py"
    ).read_text(encoding="utf-8")
    assert '"benchmark_ring_case_assembly"' in text
