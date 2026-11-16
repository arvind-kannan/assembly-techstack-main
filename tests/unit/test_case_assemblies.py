"""The demonstrator product models in cases/ load from their own folder and fit their assembly case."""

from pathlib import Path

import numpy as np
import pytest
from diss_assembly_dataclasses.assembly import Assembly

CASES = Path(__file__).resolve().parents[2] / "cases"


def load(monkeypatch, xml):
    monkeypatch.chdir(
        xml.parent
    )  # mesh and contact paths in the XML are relative to its folder
    asm = Assembly.load_from_xml(xml.name)
    for part in asm.parts.values():
        assert Path(part.mesh_path).is_file(), part.mesh_path
    return asm


def free_directions_without_screws(asm, part_name, screws):
    in_contact_with_screw = [
        c
        for c in asm.get_connections_by_part_name(part_name)
        if {c.part_name1, c.part_name2} & screws
    ]
    return np.array(
        asm.get_part_dof(
            asm.get_part_by_name(part_name), ignored_connections=in_contact_with_screw
        )
    )


@pytest.mark.parametrize(
    "xml, inserted, n_parts, n_screws",
    [
        (CASES / "angle_grinder/model/angle_grinder.xml", "BG_Tellerrad_V1p", 11, 4),
        (CASES / "battery_replacement/model/battery_replacement.xml", "lid", 51, 41),
    ],
)
def test_case_model_inserts_straight_down(
    monkeypatch, xml, inserted, n_parts, n_screws
):
    asm = load(monkeypatch, xml)
    screws = {c.part_name1 for c in asm.connections if c.connection_type == "screw"}
    assert len(asm.parts) == n_parts and len(screws) == n_screws
    touching = {
        n
        for c in asm.connections
        if c.connection_type == "geometric"
        for n in (c.part_name1, c.part_name2)
    }
    assert touching == set(asm.parts)

    dof = free_directions_without_screws(asm, inserted, screws)
    assert len(dof) > 0 and np.all(dof[:, 2] > 0.7), (
        dof
    )  # only away from the base, i.e. inserted along -z
