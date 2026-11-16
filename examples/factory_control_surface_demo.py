"""Factory control surface of the goal belief on the EMO motor (was the `__main__` block of the plotter).

Run: python examples/factory_control_surface_demo.py
"""

import copy
from pathlib import Path

import numpy as np

from disassembly_pomdp.analysis.factory_control_surface_plotter import (
    build_goal_surface,
    export_html,
    make_surface_figure,
)
from disassembly_pomdp.analysis.goal_belief import (
    GoalBeliefOperator,
    GoalNode,
    leaf_prob_from_parts,
)
from disassembly_pomdp.env_builder import load_product
from disassembly_pomdp.eol_operator import PartEOLOperatorV2, SigmoidPOther

EMO_FOLDER = Path(__file__).resolve().parents[1] / "emo_sr_v2"

# The goal leaves of the EMO motor and the parts each one removes.
LEAVES = {
    "all_screws_removed": ("04_screw_1", "05_screw_2", "06_screw_3", "07_screw_4"),
    "upper_lid_removed": ("01_upper_lid",),
    "rotor_short_removed": ("03_rotor_short",),
}

if __name__ == "__main__":
    goal_root = GoalNode(
        kind="AND",
        children=tuple(GoalNode(kind="LEAF", name=name) for name in LEAVES),
    )

    part_eol = PartEOLOperatorV2(
        p_other_provider=SigmoidPOther(alpha=0.5, beta=0.6, t0=14, e0=4)
    )
    goal_belief = GoalBeliefOperator(
        leaf_prob=lambda asm, name, ctx: leaf_prob_from_parts(
            asm, part_eol, LEAVES, name, ctx
        )
    )

    t_values = np.linspace(0, 25, 30)  # X axis bins
    checkpoints = np.arange(0, 11, 1)  # Y axis: 0..10

    # MVP: use one identical initial assembly per checkpoint
    base_asm, _base, _target = load_product(EMO_FOLDER)
    assemblies_by_ckpt = [[copy.deepcopy(base_asm)] for _ in checkpoints]
    e_by_ckpt = [int(c) for c in checkpoints]

    data = build_goal_surface(
        goal_belief=goal_belief,
        goal_root=goal_root,
        t_values=t_values,
        checkpoints=checkpoints,
        assemblies_by_ckpt=assemblies_by_ckpt,
        e_by_ckpt=e_by_ckpt,
        mode="smooth",
        reduce="mean",
    )

    fig = make_surface_figure(
        data, title="Factory Control Surface (Goal belief)", goal_label="03_rotor_short"
    )
    export_html(fig, "factory_control_surface_goal.html")
