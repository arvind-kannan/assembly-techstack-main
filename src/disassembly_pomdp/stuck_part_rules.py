"""Stuck-part rules: what a product declares about parts that can be stuck or missing (AT 227).

They sit in the product data (`product_setup.json`, field `stuck_part_rules`). A product without
the field has no stuck-part effects.
"""

from __future__ import annotations

from dataclasses import dataclass

PRIORS = ("screw", "upper")
"""The sampler priors a rule can use: "screw" -> `p_stuck_screw` and `p_missing_screw`
(the part can also be missing), "upper" -> `p_stuck_upper` (the part is never missing)."""


@dataclass(frozen=True)
class StuckPartRule:
    """One group of parts that can be stuck.

    `parts` (exact names) and `part_prefixes` select the parts, `prior` names which sampler
    priors apply to them and `fixed_to` lists the parts a stuck one gets a fixed connection to.
    """

    parts: tuple[str, ...] = ()
    part_prefixes: tuple[str, ...] = ()
    prior: str = "screw"
    fixed_to: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.prior not in PRIORS:
            raise ValueError(
                f"stuck part rule: prior must be one of {PRIORS}, got {self.prior!r}"
            )

    def matches(self, part_name: str) -> bool:
        return part_name in self.parts or (
            bool(self.part_prefixes) and part_name.startswith(self.part_prefixes)
        )

    @classmethod
    def from_dict(cls, data: dict) -> StuckPartRule:
        return cls(
            parts=tuple(data.get("parts", ())),
            part_prefixes=tuple(data.get("part_prefixes", ())),
            prior=data.get("prior", "screw"),
            fixed_to=tuple(data.get("fixed_to", ())),
        )


@dataclass(frozen=True)
class StuckPartRules:
    """The stuck-part rules of a product, in the order the product lists them; the first match wins."""

    rules: tuple[StuckPartRule, ...] = ()

    def rule_for(self, part_name: str) -> StuckPartRule | None:
        return next((r for r in self.rules if r.matches(part_name)), None)

    @classmethod
    def from_setup(cls, setup: dict) -> StuckPartRules:
        """The rules of a parsed `product_setup.json`; no field means no rules."""
        return cls(
            tuple(StuckPartRule.from_dict(r) for r in setup.get("stuck_part_rules", ()))
        )
