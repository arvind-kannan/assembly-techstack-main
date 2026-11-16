from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from diss_assembly_dataclasses.connection import Connection

from disassembly_pomdp.disassembly_pomdp_gym import DisassemblyPOMDPEnv
from disassembly_pomdp.milling_capability import MillingCapability


def test_a_milling_action_takes_a_finite_time_with_the_tool_change_counted_once(
    simple_env_assembly,
):
    geometric = Connection("part_1", "part_2")
    geometric.connection_type = "geometric"
    simple_env_assembly.connections = [geometric]

    settings = SimpleNamespace(
        # tool change: assembly -> storage (1) + change (2) + storage -> assembly (1) = 4 s
        toolchange_time_dropoff_to_storage=1.0,
        toolchange_time_assembly_to_storage=1.0,
        toolchange_time_at_toolstorage=2.0,
        toolchange_time_storage_to_assembly=1.0,
        traveltime_dropoff_to_assembly=1.0,
        # milling: approach 10 mm at 10 mm/s (1 s) + pull out 20 mm at 20 mm/s (1 s) = 2 s
        gripping_start_offset=10.0,
        gripping_minimal_pullout_value=20.0,
        gripping_pullout_factor=2.0,
        feed_rate_gripping_and_removing_approaching=10.0,
        feed_rate_gripping_and_removing_leaving=20.0,
    )
    milling = MillingCapability.from_assembly(
        simple_env_assembly,
        available_milling_tools=["mill_5mm", "mill_3mm"],
        cell_settings=settings,
    )
    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[milling],
        cell_settings=settings,
        target_part_names=["part_1"],
        belief_mode="fo",
        observer=None,
        initial_mounted_tool="mill_3mm",
    )
    env.reset(seed=0)

    obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 0, 0]))

    assert obs["accepted"] == 1
    assert info["time_cost"] == pytest.approx(4.0 + 2.0)
