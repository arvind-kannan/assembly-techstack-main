"""The area table of the slim-clone scripts (AT 164): every path it lists exists in the repository."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "tools" / "clone" / "areas.tsv"


def rows():
    lines = [
        line
        for line in TABLE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    header, *body = [line.split("\t") for line in lines]
    return [dict(zip(header, row + [""] * (len(header) - len(row)))) for row in body]


def tracked_paths():
    """Index entries, so a sparse checkout (which leaves other areas' folders out of the working tree) passes too."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    files = set(out.splitlines())
    return files | {parent.as_posix() for f in files for parent in Path(f).parents}


def listed(row):
    return [
        (column, path)
        for column in ("sparse dirs", "tests")
        for path in row[column].split()
    ]


def test_table_has_the_every_area_row_and_unique_areas():
    areas = [row["area"] for row in rows()]
    assert areas[0] == "*"
    assert len(areas) == len(set(areas))
    assert len(areas) > 1


def test_every_area_has_a_description_and_tests():
    for row in rows()[1:]:
        assert row["description"], row["area"]
        assert row["tests"], row["area"]


@pytest.mark.parametrize("row", rows(), ids=lambda row: row["area"])
def test_listed_paths_exist(row):
    tracked = tracked_paths()
    missing = [(c, p) for c, p in listed(row) if p not in tracked]
    assert not missing, f"area '{row['area']}' lists paths that do not exist: {missing}"
