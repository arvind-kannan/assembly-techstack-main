"""Tracked case XMLs hold only paths relative to their own folder, and every one resolves."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUBTREES = ("crc_diss_assembly_dataclasses/", "crc_disassembly_sequence_generator/")
PATH_ATTR = re.compile(
    r'(?:mesh_path|points_filename|normals_filename|distances_filename)="([^"]*)"'
)


def tracked_case_xmls():
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.xml"], cwd=ROOT, capture_output=True, check=True
    )
    names = [n for n in out.stdout.decode("utf-8").split("\0") if n]
    return [n for n in names if not n.startswith(SUBTREES) and (ROOT / n).exists()]


@pytest.mark.parametrize("xml", tracked_case_xmls())
def test_paths_in_case_xml_are_relative_and_resolve(xml):
    xml_path = ROOT / xml
    for path in PATH_ATTR.findall(xml_path.read_text(encoding="utf-8")):
        assert not path.startswith("/") and not re.match(r"^[A-Za-z]:", path), (
            f"absolute path {path}"
        )
        assert "\\" not in path, f"backslash in {path}"
        assert (xml_path.parent / path).exists(), f"{path} not found next to {xml}"
