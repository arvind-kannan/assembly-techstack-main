from pathlib import Path

import pytest
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot

from disassembly_pomdp.env_builder import (
    Cell,
    build_env,
    cell_from_resources,
    load_product,
)
from disassembly_pomdp.gripping_and_removing_capability import (
    GrippingAndRemovingCapability,
)
from disassembly_pomdp.screwing_capability import ScrewingCapability
from mrtamp.cell_file import load_cell_file

EMO = Path(__file__).resolve().parents[2] / "emo_sr_v2"


def test_load_product_reads_setup():
    asm, base, target = load_product(EMO)
    assert base == ["08_lower_lid", "02_stator"] and target == ["03_rotor_short"]
    assert asm.get_part_by_name("03_rotor_short") is not None


def test_default_cell_gives_both_capabilities():
    env = build_env(EMO)
    assert [type(c) for c in env.capabilities] == [
        ScrewingCapability,
        GrippingAndRemovingCapability,
    ]


def test_capabilities_follow_the_cell():
    env = build_env(EMO, Cell(screwdrivers=()))
    assert [type(c) for c in env.capabilities] == [GrippingAndRemovingCapability]
    with pytest.raises(ValueError):
        build_env(EMO, Cell(screwdrivers=(), grippers=()))


def test_missing_setup_file(tmp_path):
    (tmp_path / "p.xml").write_text("<assembly name='p'/>")
    with pytest.raises(FileNotFoundError):
        load_product(tmp_path)


def test_setup_can_be_given_instead_of_the_stored_one():
    env = build_env(EMO, base_parts=["02_stator"], target_parts=[])
    assert env.target_part_names == []


# The robots of the cell (#230): capabilities find the robots for a tool through the cell's core robots.

CASES = Path(__file__).resolve().parents[2] / "cases"


def _capabilities_by_process(env):
    return {cap.process: cap for cap in env.capabilities}


def _cell_of_file(name):
    cell_file = load_cell_file(CASES / name)
    return cell_from_resources(cell_file.cell, cell_file.travel_times)


def test_the_default_cell_keeps_todays_robot_assignment():
    caps = _capabilities_by_process(build_env(EMO))

    assert caps[Process.UNSCREWING].robots_for("H5") == ("robot_1",)
    assert caps[Process.UNSCREWING].robots_for("H6") == ("robot_2",)
    assert caps[Process.GRIPPING_AND_REMOVING].robots_for("default_gripper") == (
        "robot_1",
    )


def test_on_the_real_transformer_cell_robot_2_unscrews_with_h5():
    caps = _capabilities_by_process(
        build_env(EMO, _cell_of_file("transformer_cell.json"))
    )

    assert caps[Process.UNSCREWING].robots_for("H5") == ("robot_2",)
    assert caps[Process.GRIPPING_AND_REMOVING].robots_for("default_gripper") == (
        "robot_1",
    )


@pytest.mark.parametrize(
    "cell_file", ["transformer_cell_sim.json", "transformer_cell_simulation.json"]
)
def test_in_a_simulation_variant_both_robots_unscrew_with_h5(cell_file):
    caps = _capabilities_by_process(build_env(EMO, _cell_of_file(cell_file)))

    assert caps[Process.UNSCREWING].robots_for("H5") == ("robot_1", "robot_2")


def _one_robot_cell():
    return Cell(
        screwdrivers=("H5", "H6"),
        robots=(
            Robot(
                "robot_1",
                1,
                tools=("H5", "default_gripper"),
                capabilities=(Process.UNSCREWING, Process.GRIPPING_AND_REMOVING),
            ),
        ),
    )


def test_a_tool_on_no_robot_gives_no_robot_the_capability():
    caps = _capabilities_by_process(build_env(EMO, _one_robot_cell()))

    assert caps[Process.UNSCREWING].robots_for("H6") == ()
    assert caps[Process.UNSCREWING].robot_for("H6") is None


def test_the_actions_of_a_tool_on_no_robot_are_masked():
    env = build_env(EMO, _one_robot_cell())
    env.reset(seed=0)
    cap_idx = next(
        i for i, c in enumerate(env.capabilities) if c.process is Process.UNSCREWING
    )
    tool_idx = env.capabilities[cap_idx].list_tools().index("H6")

    *_, info = env.step([cap_idx, tool_idx, 0])

    assert info["mask_reason"] == "no_robot"
    assert env.current_mounted_tool == "H5"


@pytest.mark.parametrize("cell_file", [None, "transformer_cell.json"])
def test_after_a_tool_change_every_core_robot_holds_the_mounted_tool(cell_file):
    env = build_env(EMO, None if cell_file is None else _cell_of_file(cell_file))
    env.reset(seed=0)

    env.current_mounted_tool = "default_gripper"

    assert env.cell_state.mounted_tools == {
        "robot_1": "default_gripper",
        "robot_2": "default_gripper",
    }
