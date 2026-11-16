"""A step of the sequence graph, typed: what an edge of the graph says about one action.

The graph keeps its edge attributes as plain dicts (exports, the product explorer and the precedence
helper read them). `Step.from_edge` reads one such edge, `Step.to_attrs` writes one back; planners take
a `Step`, never a dict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from diss_assembly_dataclasses.process import Process


@dataclass(frozen=True)
class Step:
    """One step: `process` on `target_part` with `tool`, taking `action_time` seconds nominally.

    `inverse` marks a step added by reversal (`complete_by_reversal`); `supply_time` is the part of an inverse
    inserting step spent travelling to the supply; `direction` is the removal (or, reversed, insertion) direction.
    """

    process: Process
    target_part: str
    tool: str | None
    action_time: float
    inverse: bool = False
    direction: tuple[float, ...] | None = None
    screw_head: str | None = None
    supply_time: float | None = None
    label: str | None = None

    @classmethod
    def from_edge(cls, attrs: Mapping) -> Step:
        direction = attrs.get("direction")
        supply = attrs.get("supply_time")
        return cls(
            process=Process(attrs["process"]),
            target_part=attrs["target_part"],
            tool=attrs["tool"],
            action_time=float(attrs["action_time"]),
            inverse=bool(attrs.get("inverse", False)),
            direction=None if direction is None else tuple(float(x) for x in direction),
            screw_head=attrs.get("screw_head"),
            supply_time=None if supply is None else float(supply),
            label=attrs.get("label"),
        )

    def to_attrs(self) -> dict:
        """The edge attributes of this step; optional ones only when set, as the graph has always stored them."""
        attrs = {
            "process": self.process.value,
            "tool": self.tool,
            "target_part": self.target_part,
        }
        if self.inverse:
            attrs["inverse"] = True
        if self.label is not None:
            attrs["label"] = self.label
        if self.screw_head is not None:
            attrs["screw_head"] = self.screw_head
        if self.direction is not None:
            attrs["direction"] = list(self.direction)
        attrs["action_time"] = self.action_time
        if self.supply_time is not None:
            attrs["supply_time"] = self.supply_time
        return attrs
