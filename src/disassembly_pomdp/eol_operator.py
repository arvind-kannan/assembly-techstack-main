"""End-of-life belief and sampling utilities for disassembly POMDP episodes.

This module provides a small part-level EOL operator used by the environment to
maintain beliefs over missing and stuck parts.  It deliberately stays lightweight:
the operator does not own the assembly state itself, but updates probabilities
from reset priors, missing observations, and action outcome feedback.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from diss_assembly_dataclasses.assembly import Assembly

from disassembly_pomdp.stuck_part_rules import StuckPartRules

InspectionPriorSampler: TypeAlias = Callable[
    [str, Assembly, np.random.Generator],
    float,
]


@dataclass(frozen=True)
class EOLPartEvent:
    """Diagnostic record produced after updating one part-level EOL belief."""

    part: str
    success: bool
    p_exec: float
    p_success_prior: float
    p_stuck_prior: float
    p_success_post: float
    p_stuck_post: float


class PartEOLOperatorV2:
    """Part-level belief operator with binary latent state ``free``/``stuck``.

    The maintained belief is ``p_stuck[part] = P(Z=stuck | history)``.  Missing
    probability is tracked separately as a lightweight extension used by the
    expected-success model.

    ``p_exec`` is passed in by the environment and represents the non-EOL
    execution gate or execution success factor.  When it is too small, updates
    can be skipped because the action did not provide meaningful EOL evidence.
    """

    def __init__(
        self,
        p_free_success: float = 0.99,
        p_stuck_success: float = 0.0,
        default_p_stuck: float = 0.2,
        clip_eps: float = 1e-6,
        min_update_p_exec: float = 1e-3,
        freeze_updates: bool = False,
    ) -> None:
        self.p_free_success = float(p_free_success)
        self.p_stuck_success = float(p_stuck_success)
        self.default_p_stuck = float(default_p_stuck)
        self.clip_eps = float(clip_eps)
        self.min_update_p_exec = float(min_update_p_exec)
        self.freeze_updates = bool(freeze_updates)

        self._p_stuck: dict[str, float] = {}
        self._p_missing: dict[str, float] = {}

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate constructor probabilities and numerical thresholds."""
        if not 0.0 <= self.p_stuck_success <= 1.0:
            raise ValueError("p_stuck_success must be in [0,1]")
        if not 0.0 <= self.p_free_success <= 1.0:
            raise ValueError("p_free_success must be in [0,1]")
        if not 0.0 <= self.default_p_stuck <= 1.0:
            raise ValueError("default_p_stuck must be in [0,1]")
        if not 0.0 < self.clip_eps < 0.5:
            raise ValueError("clip_eps must be in (0,0.5)")
        if self.p_free_success < self.p_stuck_success:
            raise ValueError("p_free_success must be >= p_stuck_success")
        if self.min_update_p_exec < 0.0:
            raise ValueError("min_update_p_exec must be >= 0")

    @staticmethod
    def _clip_probability(value: float) -> float:
        """Return ``value`` clipped to the probability interval [0, 1]."""
        return float(np.clip(value, 0.0, 1.0))

    # ---------- lifecycle / reproducibility ----------

    def reset_for_parts(
        self,
        part_names: Iterable[str],
        rng: np.random.Generator,
        assembly: Assembly,
        inspection_prior_sampler: InspectionPriorSampler | None = None,
    ) -> None:
        """Reset beliefs for the provided part names.

        ``dict.fromkeys`` preserves order while removing duplicates.
        """
        self._p_stuck.clear()
        self._p_missing.clear()

        for part_name in dict.fromkeys(part_names):
            stuck_prior = self.default_p_stuck
            if inspection_prior_sampler is not None:
                try:
                    stuck_prior = float(
                        inspection_prior_sampler(part_name, assembly, rng)
                    )
                except Exception:  # pragma: no cover - defensive prior fallback
                    stuck_prior = self.default_p_stuck

            if not np.isfinite(stuck_prior):
                stuck_prior = self.default_p_stuck

            self._p_stuck[part_name] = self._clip_probability(stuck_prior)
            # MVP: default missing prior = 0 unless a sampler is added later.
            self._p_missing[part_name] = 0.0

    @staticmethod
    def prior_from_assembly_fixed(
        part_name: str,
        assembly: Assembly,
        rng: np.random.Generator,
        *,
        p_fixed: float = 0.95,
        p_free: float = 0.15,
    ) -> float:
        """Return a stuck prior based on whether the part has a fixed connection."""
        del rng  # Signature compatibility with InspectionPriorSampler.

        for connection in getattr(assembly, "connections", []):
            if getattr(connection, "connection_type", None) != "fixed":
                continue

            first_part = getattr(connection, "part_name1", None)
            second_part = getattr(connection, "part_name2", None)
            if part_name in {first_part, second_part}:
                return float(p_fixed)

        return float(p_free)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Return a serializable snapshot of the current EOL belief state."""
        return {"p_stuck": dict(self._p_stuck), "p_missing": dict(self._p_missing)}

    def restore(self, snap: Mapping[str, Mapping[str, float]]) -> None:
        """Restore a belief snapshot created by :meth:`snapshot`."""
        stuck_snapshot = snap.get("p_stuck", {})
        missing_snapshot = snap.get("p_missing", {})

        self._p_stuck = {
            part_name: self._clip_probability(value)
            for part_name, value in stuck_snapshot.items()
        }
        # Default to 0.0 for parts that exist in p_stuck but are absent in p_missing.
        self._p_missing = {
            part_name: self._clip_probability(missing_snapshot.get(part_name, 0.0))
            for part_name in self._p_stuck
        }

    # ---------- core API ----------

    def get_p_stuck(self, part_name: str) -> float:
        """Return the current stuck probability for ``part_name``."""
        return float(self._p_stuck.get(part_name, self.default_p_stuck))

    def get_p_missing(self, part_name: str) -> float:
        """Return the current missing probability for ``part_name``."""
        return float(self._p_missing.get(part_name, 0.0))

    def _p_success_given_free(self, p_exec: float) -> float:
        return self._clip_probability(p_exec) * self.p_free_success

    def _p_success_given_stuck(self, p_exec: float) -> float:
        return self._clip_probability(p_exec) * self.p_stuck_success

    def expected_success_prob(self, part_name: str, p_exec: float = 1.0) -> float:
        """Return the marginal probability of successful action execution."""
        p_exec = self._clip_probability(p_exec)
        p_missing = self.get_p_missing(part_name)
        p_stuck = self.get_p_stuck(part_name)

        p_existing = self._clip_probability(1.0 - p_missing)
        p_success_free = self._p_success_given_free(p_exec)
        p_success_stuck = self._p_success_given_stuck(p_exec)

        p_success = p_existing * (
            (1.0 - p_stuck) * p_success_free + p_stuck * p_success_stuck
        )
        return self._clip_probability(p_success)

    def on_missing_observed(
        self,
        part_name: str,
        *,
        kind: str = "target_missing",
    ) -> None:
        """Update belief after strong evidence that a part is missing."""
        del kind  # Kept for diagnostic call-site compatibility.

        missing_prob = self._clip_probability(1.0 - self.clip_eps)
        self._p_missing[part_name] = missing_prob

        # Keep p_stuck valid under (1 - p_missing).
        p_stuck = self.get_p_stuck(part_name)
        stuck_cap = max(self.clip_eps, 1.0 - missing_prob - self.clip_eps)
        if p_stuck > stuck_cap:
            self._p_stuck[part_name] = stuck_cap

    def update_on_outcome(
        self,
        part_name: str,
        observed_success: bool,
        p_exec: float = 1.0,
    ) -> float:
        """Bayesian stuck-belief update after an attempted action outcome."""
        if self.get_p_missing(part_name) > 0.5:
            return self.get_p_stuck(part_name)

        p_exec = self._clip_probability(p_exec)
        if p_exec < self.min_update_p_exec:
            return self.get_p_stuck(part_name)

        p_stuck = self.get_p_stuck(part_name)
        p_free_likelihood = float(
            np.clip(
                self._p_success_given_free(p_exec),
                self.clip_eps,
                1.0 - self.clip_eps,
            )
        )
        p_stuck_likelihood = float(
            np.clip(
                self._p_success_given_stuck(p_exec),
                self.clip_eps,
                1.0 - self.clip_eps,
            )
        )

        if bool(observed_success):
            numerator = p_stuck * p_stuck_likelihood
            denominator = numerator + (1.0 - p_stuck) * p_free_likelihood
        else:
            numerator = p_stuck * (1.0 - p_stuck_likelihood)
            denominator = numerator + (1.0 - p_stuck) * (1.0 - p_free_likelihood)

        if denominator <= 0.0 or not np.isfinite(denominator):
            return p_stuck

        updated_p_stuck = float(
            np.clip(numerator / denominator, self.clip_eps, 1.0 - self.clip_eps)
        )
        self._p_stuck[part_name] = updated_p_stuck
        return updated_p_stuck

    def on_attempt(
        self,
        part_name: str,
        success: bool,
        p_exec: float = 1.0,
    ) -> EOLPartEvent:
        """Update EOL belief after a real action attempt and return diagnostics."""
        p_exec = self._clip_probability(p_exec)

        stuck_prior = self.get_p_stuck(part_name)
        success_prior = self.expected_success_prob(part_name, p_exec=p_exec)

        if self.freeze_updates:
            stuck_post = stuck_prior
        else:
            stuck_post = self.update_on_outcome(part_name, success, p_exec=p_exec)

        success_post = self.expected_success_prob(part_name, p_exec=p_exec)

        if bool(success):
            # If we truly removed it, it certainly existed at attempt time.
            self._p_missing[part_name] = float(self.clip_eps)

        return EOLPartEvent(
            part=part_name,
            success=bool(success),
            p_exec=float(p_exec),
            p_success_prior=float(success_prior),
            p_stuck_prior=float(stuck_prior),
            p_success_post=float(success_post),
            p_stuck_post=float(stuck_post),
        )


class EOLUnifiedSampler:
    """Simple source of reset-time missing and stuck priors.

    Which parts can be stuck or missing comes from the product's ``rules``; without
    rules no part is stuck or missing.  Each rule's ``prior`` picks the probability.

    One object provides:
      - ``sample_stuck(part, asm, rng) -> p_stuck``
      - ``sample_missing(part, asm, rng) -> p_missing``

    The environment samples missing and stuck events independently from separate
    random draws.
    """

    def __init__(
        self,
        *,
        p_stuck_screw: float = 0.05,
        p_stuck_upper: float = 0.05,
        p_missing_screw: float = 0.02,
        rules: StuckPartRules | None = None,
    ) -> None:
        self.p_stuck_screw = float(p_stuck_screw)
        self.p_stuck_upper = float(p_stuck_upper)
        self.p_missing_screw = float(p_missing_screw)
        self.rules = rules or StuckPartRules()

    def sample_stuck(
        self,
        part_name: str,
        asm: Assembly,
        rng: np.random.Generator,
    ) -> float:
        """Return the reset-time stuck prior for ``part_name``."""
        del asm, rng  # Current sampler is deterministic by part name.

        rule = self.rules.rule_for(part_name)
        if rule is None:
            return 0.0
        return self.p_stuck_screw if rule.prior == "screw" else self.p_stuck_upper

    def sample_missing(
        self,
        part_name: str,
        asm: Assembly,
        rng: np.random.Generator,
    ) -> float:
        """Return the reset-time missing prior for ``part_name``."""
        del asm, rng  # Current sampler is deterministic by part name.

        rule = self.rules.rule_for(part_name)
        if rule is not None and rule.prior == "screw":
            return self.p_missing_screw
        return 0.0
