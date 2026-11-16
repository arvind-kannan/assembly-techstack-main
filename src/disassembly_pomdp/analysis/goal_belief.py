from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
from diss_assembly_dataclasses.assembly import Assembly

from disassembly_pomdp.eol_operator import EOLContext, PartEOLOperatorV2

# -----------------------------
# Goal-level belief graph (AND/OR)
# -----------------------------


@dataclass(frozen=True)
class GoalNode:
    """
    A minimal AND/OR graph node for goal success probability aggregation.

    kind:
      - "LEAF": a leaf event whose probability is supplied by a callback
      - "AND": independent conjunction
      - "OR" : independent disjunction

    IMPORTANT fork point:
      - Independence is an approximation. If you later need correlations,
        replace the aggregator, not the part-level EOL operator.
    """

    kind: str
    name: str = ""
    children: tuple["GoalNode", ...] = ()


class GoalBeliefOperator:
    """
    Computes P(goal_success) from:
      - a part-level operator (for leaf probabilities),
      - a goal graph (AND/OR/LEAF),
      - an optional p_other context (passed through to the part-level operator).

    Intended usage:
      - Diagnostic / try-now heuristic
      - Explainability: inspect intermediate node probabilities

    Design note:
      - This module does not update beliefs. It only queries current beliefs.
      - Updates should remain local (PartEOLOperatorV2.on_attempt).

    When calling:
        goal = GoalBeliefOperator(
            leaf_prob=lambda asm, name, ctx: leaf_prob_from_parts(asm, part_eol, leaves, name, ctx)
        )
    """

    def __init__(
        self,
        leaf_prob: Callable[[Assembly, str, Optional[EOLContext]], float],
    ) -> None:
        """
        leaf_prob(assembly, leaf_name, ctx) -> probability in [0,1]

        The current leaf prob means "the probability of success if try now",
        not "the probability of eventually achieving the goal within the remaining horizon".

        Leaf strategies:
          - Deterministic predicate from state: 0/1 (e.g., already removed -> 1)
          - Stochastic success estimate from EOL: expected_success_prob(...)
          - Mixed (recommended): state facts dominate; otherwise use EOL estimate.
        """
        self.leaf_prob = leaf_prob

    def prob(
        self, node: GoalNode, assembly: Assembly, ctx: Optional[EOLContext] = None
    ) -> float:
        k = node.kind.upper()
        if k == "LEAF":
            return float(np.clip(self.leaf_prob(assembly, node.name, ctx), 0.0, 1.0))
        if k == "AND":
            out = 1.0
            for ch in node.children:
                out *= self.prob(ch, assembly, ctx)
            return float(np.clip(out, 0.0, 1.0))
        if k == "OR":
            prod_fail = 1.0
            for ch in node.children:
                prod_fail *= 1.0 - self.prob(ch, assembly, ctx)
            return float(np.clip(1.0 - prod_fail, 0.0, 1.0))
        raise ValueError(f"Unknown GoalNode kind: {node.kind}")


def leaf_prob_from_parts(
    assembly: Assembly,
    part_eol: PartEOLOperatorV2,
    leaves: Mapping[str, Iterable[str]],
    leaf_name: str,
    ctx: Optional[EOLContext] = None,
) -> float:
    """
    Returns the probability of the leaf ``leaf_name``, a goal leaf that is reached
    when all the parts ``leaves[leaf_name]`` are removed.

    Mixed strategy (facts dominate):
      - A part that is already removed counts as 1.0.
      - Otherwise the part counts as the expected success probability of attempting
        its removal now, from the part-level EOL belief. This is not the eventual
        completion probability; that needs planning / multi-step models and would
        change here.
      - The leaf is the independent conjunction (product) of its parts.

    Fork points:
      - Part aggregation (product / min / mean / ...).
      - Whether some leaves should be pure predicates (0/1) only.
      - Whether to introduce operation-specific EOL above part_eol.
    """
    if leaf_name not in leaves:
        raise KeyError(f"Unknown leaf name: {leaf_name}")

    def part_missing(part: str) -> bool:
        try:
            return assembly.get_part_by_name(part) is None
        except Exception:
            return part not in getattr(assembly, "parts", {})

    out = 1.0
    for part in leaves[leaf_name]:
        if not part_missing(part):
            out *= float(
                np.clip(part_eol.expected_success_prob(part, ctx=ctx), 0.0, 1.0)
            )
    return float(np.clip(out, 0.0, 1.0))
