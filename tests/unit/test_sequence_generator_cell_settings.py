"""The sequence generator finds its cell_settings.json from any working directory."""

import json
from pathlib import Path

import pytest

from crc_disassembly_sequence_generator.planning import load_cell_settings

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "crc_disassembly_sequence_generator" / "cell_settings.json"


def test_load_cell_settings_from_another_working_directory(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)

    settings = load_cell_settings()

    assert settings == json.loads(SETTINGS.read_text(encoding="utf-8"))
    assert "Could not find cell_settings.json" not in capsys.readouterr().out


@pytest.mark.parametrize("start", [ROOT, ROOT / "tests"])
def test_load_cell_settings_from_repo_root_and_tests(start, monkeypatch, capsys):
    monkeypatch.chdir(start)

    assert load_cell_settings() == json.loads(SETTINGS.read_text(encoding="utf-8"))
    assert "Could not find cell_settings.json" not in capsys.readouterr().out
