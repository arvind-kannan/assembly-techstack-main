"""Stuck-part rules of a product (AT 227): the product declares which parts can be stuck and what a stuck part is fixed to."""

import json
from pathlib import Path

import pytest

from disassembly_pomdp.disassembly_pomdp_gym import (
    DisassemblyPOMDPEnv,
    inject_eol_stuck_fixed_connections,
)
from disassembly_pomdp.env_builder import build_env, load_product, load_stuck_part_rules
from disassembly_pomdp.eol_operator import EOLUnifiedSampler, PartEOLOperatorV2
from disassembly_pomdp.stuck_part_rules import StuckPartRule, StuckPartRules

EMO = Path(__file__).resolve().parents[2] / "emo_sr_v2"
SCREWS = ["04_screw_1", "05_screw_2", "06_screw_3", "07_screw_4"]
UPPER_LID, STATOR, LOWER_LID = "01_upper_lid", "02_stator", "08_lower_lid"


def fixed_pairs(assembly) -> set[frozenset[str]]:
    return {
        frozenset((c.part_name1, c.part_name2))
        for c in assembly.connections
        if c.connection_type == "fixed"
    }


def injected(stuck: set[str], rules) -> set[frozenset[str]]:
    """Fixed connections `inject_eol_stuck_fixed_connections` adds on EMO for the stuck parts."""
    assembly, _base, _target = load_product(EMO)
    before = fixed_pairs(assembly)
    inject_eol_stuck_fixed_connections(assembly, stuck, rules)
    return fixed_pairs(assembly) - before


# What the env hard-coded before the rules moved into the product data.
EMO_EXPECTED = [
    (set(), set()),
    (
        {SCREWS[0]},
        {frozenset((SCREWS[0], UPPER_LID)), frozenset((SCREWS[0], LOWER_LID))},
    ),
    (
        set(SCREWS),
        {frozenset((s, p)) for s in SCREWS for p in (UPPER_LID, LOWER_LID)},
    ),
    ({UPPER_LID}, {frozenset((UPPER_LID, STATOR))}),
    (
        {SCREWS[1], UPPER_LID},
        {
            frozenset((SCREWS[1], UPPER_LID)),
            frozenset((SCREWS[1], LOWER_LID)),
            frozenset((UPPER_LID, STATOR)),
        },
    ),
    ({STATOR}, set()),
    ({"03_rotor_short"}, set()),
]


@pytest.mark.parametrize("stuck, expected", EMO_EXPECTED)
def test_emo_rules_from_product_data_give_todays_fixed_connections(stuck, expected):
    assert injected(stuck, load_stuck_part_rules(EMO)) == expected


def test_a_product_without_rules_gets_no_fixed_connections():
    assert injected(set(SCREWS) | {UPPER_LID}, StuckPartRules()) == set()


def test_emo_rules_give_todays_sampler_priors():
    sampler = EOLUnifiedSampler(
        p_stuck_screw=0.05,
        p_stuck_upper=0.25,
        p_missing_screw=0.02,
        rules=load_stuck_part_rules(EMO),
    )
    assembly, _base, _target = load_product(EMO)
    stuck = {p: sampler.sample_stuck(p, assembly, None) for p in assembly.parts}
    missing = {p: sampler.sample_missing(p, assembly, None) for p in assembly.parts}

    assert {p for p, v in stuck.items() if v == 0.05} == set(SCREWS)
    assert {p for p, v in stuck.items() if v == 0.25} == {UPPER_LID}
    assert {p for p, v in stuck.items() if v not in (0.05, 0.25)} == set(
        assembly.parts
    ) - set(SCREWS) - {UPPER_LID}
    assert {p for p, v in missing.items() if v == 0.02} == set(SCREWS)
    assert {p for p, v in missing.items() if v != 0.02} == set(assembly.parts) - set(
        SCREWS
    )
    assert all(v == 0.0 for p, v in stuck.items() if p not in SCREWS + [UPPER_LID])


def test_env_without_rules_has_no_stuck_effects_with_certain_stuck_priors():
    """The probabilities are nonzero, so only the missing rules keep the assembly free."""
    stock = build_env(EMO)
    assembly, _base, targets = load_product(EMO)
    before = fixed_pairs(assembly)
    env = DisassemblyPOMDPEnv(
        assembly=assembly,
        capabilities=stock.capabilities,
        cell_settings=stock.cell_settings,
        target_part_names=targets,
        world_sampler=EOLUnifiedSampler(
            p_missing_screw=1.0, p_stuck_screw=1.0, p_stuck_upper=1.0
        ),
        eol_op=PartEOLOperatorV2(freeze_updates=True),
        belief_mode="fo",
        initial_mounted_tool=stock.initial_mounted_tool,
    )

    _obs, info = env.reset(seed=0)

    assert info["eol_missing_sampled_parts"] == []
    assert info["eol_stuck_sampled_parts"] == []
    assert fixed_pairs(env.assembly) == before


def test_build_env_on_a_product_without_rules_steps_without_injected_connections():
    tiny = Path(__file__).resolve().parents[1] / "test_products" / "tiny_test_product"
    env = build_env(tiny)
    before = fixed_pairs(env.assembly)

    env.reset(seed=0)
    env.step(env.action_space.sample())

    assert load_stuck_part_rules(tiny) == StuckPartRules()
    assert fixed_pairs(env.assembly) == before


def test_rules_are_read_from_product_setup_json(tmp_path):
    (tmp_path / "product_setup.json").write_text(
        json.dumps(
            {
                "base_parts": [],
                "target_parts": [],
                "stuck_part_rules": [
                    {
                        "part_prefixes": ["bolt_"],
                        "parts": ["cap"],
                        "prior": "screw",
                        "fixed_to": ["plate"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rules = load_stuck_part_rules(tmp_path)

    assert rules == StuckPartRules(
        (StuckPartRule(("cap",), ("bolt_",), "screw", ("plate",)),)
    )
    assert rules.rule_for("bolt_3") is rules.rules[0]
    assert rules.rule_for("cap") is rules.rules[0]
    assert rules.rule_for("plate") is None


def test_missing_rules_field_means_no_rules(tmp_path):
    (tmp_path / "product_setup.json").write_text("{}", encoding="utf-8")

    assert load_stuck_part_rules(tmp_path) == StuckPartRules()


def test_unknown_prior_is_refused(tmp_path):
    (tmp_path / "product_setup.json").write_text(
        json.dumps({"stuck_part_rules": [{"parts": ["a"], "prior": "bogus"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prior"):
        load_stuck_part_rules(tmp_path)
