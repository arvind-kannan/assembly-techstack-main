"""The CI check that rejects absolute paths in tracked data files (AT 168)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_absolute_paths", ROOT / "tools" / "ci" / "check_absolute_paths.py"
)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def hits(text, name="case/product.xml"):
    return [(h.file, h.line, h.path) for h in check.scan({name: text})]


@pytest.mark.parametrize(
    "text, path",
    [
        (
            '<mesh mesh_path="/home/anna/thesis/part.stl"/>',
            "/home/anna/thesis/part.stl",
        ),
        ('{"folder": "/Users/bob/cases/emo"}', "/Users/bob/cases/emo"),
        ("folder: /mnt/d/cases/emo", "/mnt/d/cases/emo"),
        ("out = /tmp/run1/out.json", "/tmp/run1/out.json"),
        ('<a b="/opt/ros/humble/share"/>', "/opt/ros/humble/share"),
        ('<a b="D:/3d_files/emo"/>', "D:/3d_files/emo"),
        ('<a b="C:\\Users\\anna\\emo"/>', "C:\\Users\\anna\\emo"),
        ('{"p": "C:\\\\Users\\\\anna\\\\emo"}', "C:\\\\Users\\\\anna\\\\emo"),
        ('<a b="file:///home/anna/x.stl"/>', "file:///home/anna/x.stl"),
        ('<a b="file:///C:/x/y.stl"/>', "file:///C:/x/y.stl"),
    ],
)
def test_absolute_path_is_flagged(text, path):
    assert hits(text) == [("case/product.xml", 1, path)]


@pytest.mark.parametrize(
    "text",
    [
        '<a href="https://example.org/home/anna/page"/>',
        '<a href="http://example.org/tmp/x"/>',
        '<mesh filename="package://robot/meshes/base.stl"/>',
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">',
        '<a b="meshes/home/base.stl"/>',
        '<a b="../home/base.stl"/>',
        '<a b="./tmp/base.stl"/>',
        "<!-- was /home/anna/x.stl -->",
        "<!--\n  was\n  /home/anna/x.stl\n-->",
        '<a b="home/anna"/>',
    ],
)
def test_allowed_text_is_not_flagged(text):
    assert hits(text) == []


def test_hit_is_reported_with_its_line_number():
    text = '<a/>\n<!-- /home/x/c -->\n<b p="/home/anna/x"/>\n'
    assert hits(text) == [("case/product.xml", 3, "/home/anna/x")]


def test_comment_text_after_a_comment_closes_is_scanned():
    assert hits('<!-- ok --><a p="/home/anna/x"/>') == [
        ("case/product.xml", 1, "/home/anna/x")
    ]


@pytest.mark.parametrize(
    "name",
    [
        ".idea/workspace.xml",
        ".vscode/settings.json",
        "sub/.idea/misc.xml",
        ".claude/log/AT-1/evidence/run_record.json",
        "uv.lock",
        "crc_disassembly_sim/docker-compose.yml",
    ],
)
def test_excluded_file_is_not_scanned(name):
    assert hits('{"p": "/home/anna/x"}', name=name) == []


def test_every_exclude_has_a_reason():
    assert check.EXCLUDES
    assert all(reason.strip() for _, reason in check.EXCLUDES)


def test_only_data_file_types_are_scanned():
    assert hits("/home/anna/x", name="notes.md") == []
    assert hits("/home/anna/x", name="tool.py") == []
    assert hits("/home/anna/x", name="cell.URDF") != []
    assert hits("/home/anna/x", name="robot.xacro") != []


def test_main_fails_with_a_message_naming_file_line_path_and_fix(tmp_path, capsys):
    (tmp_path / "ok.json").write_text('{"p": "meshes/a.stl"}')
    (tmp_path / "bad.xml").write_text('<a/>\n<b p="/home/anna/x.stl"/>\n')
    status = check.main(["ok.json", "bad.xml"], root=tmp_path)
    err = capsys.readouterr().err
    assert status == 1
    assert "bad.xml:2" in err and "/home/anna/x.stl" in err
    assert "relative" in err and "ok.json" not in err
    assert "own folder" in err


def test_main_passes_on_clean_files(tmp_path, capsys):
    (tmp_path / "ok.json").write_text('{"p": "meshes/a.stl"}')
    assert check.main(["ok.json"], root=tmp_path) == 0
    assert capsys.readouterr().err == ""


def test_main_skips_listed_files_that_do_not_exist(tmp_path):
    assert check.main(["gone.xml"], root=tmp_path) == 0
