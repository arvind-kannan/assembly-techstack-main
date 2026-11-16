import sys
from pathlib import Path

# Add project src/ folder to sys.path so that `import disassembly_pomdp` works
ROOT = Path(__file__).resolve().parents[1]  # .../pomdp_disassembly_gym
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import open3d as o3d
import trimesh
from diss_assembly_dataclasses.assembly import Assembly

from crc_disassembly_sequence_generator.operations import (
    GrippingAndRemovingOperation,
    ScrewingOperation,
)
from crc_disassembly_sequence_generator.planning import (
    HierarchicalDisassemblySequencePlanner,
)
from disassembly_pomdp.config import CellSettings
from disassembly_pomdp.disassembly_pomdp_gym import DisassemblyPOMDPEnv
from disassembly_pomdp.env_builder import load_stuck_part_rules
from disassembly_pomdp.eol_operator import EOLUnifiedSampler
from disassembly_pomdp.gripping_and_removing_capability import (
    GrippingAndRemovingCapability,
)
from disassembly_pomdp.screwing_capability import ScrewingCapability


def load_emo_sr_v2_assembly() -> Assembly:
    """Load EMO_SR_V2.xml from tests/test_env/data."""
    data_dir = ROOT / "tests" / "test_env" / "data"
    xml_path = data_dir / "EMO_SR_V2.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"EMO_SR_V2.xml not found at {xml_path}")

    asm = Assembly.load_from_xml(str(xml_path))
    return asm


def make_cell_settings() -> CellSettings:
    """Reuse the same numeric values as in the tests."""
    return CellSettings(
        feed_rate_milling=100.0,
        feed_rate_gripping_and_removing_approaching=100.0,
        feed_rate_gripping_and_removing_leaving=100.0,
        feed_rate_screwing_approaching=100.0,
        feed_rate_screwing_leaving=100.0,
        toolchange_time_dropoff_to_storage=1.0,
        toolchange_time_assembly_to_storage=1.0,
        toolchange_time_at_toolstorage=1.0,
        toolchange_time_start_to_toolstorage=1.0,
        toolchange_time_storage_to_assembly=1.0,
        traveltime_assembly_to_dropoff=1.0,
        traveltime_dropoff_to_assembly=1.0,
        screwing_start_offset=10.0,
        screwing_pullout_factor=1.0,
        screwing_minimal_pullout_value=5.0,
        gripping_start_offset=10.0,
        gripping_pullout_factor=1.0,
        gripping_minimal_pullout_value=5.0,
        milling_penalty=0.0,
    )


# ---------------------------------------------------------------------
# Adapter: connect HierarchicalDisassemblySequencePlanner → GrippingAndRemovingCapability
# ---------------------------------------------------------------------


def make_planner(asm: Assembly) -> HierarchicalDisassemblySequencePlanner:
    """
    Build the same planner you would normally use in the big system.
    Adjust numeric constants if you like.
    """
    planner = HierarchicalDisassemblySequencePlanner(
        assembly=asm,
        cost_milling=10.0,
        cost_screwing=5.0,
        cost_gripping_and_removing=5.0,
        cost_safety_ring=5.0,
        mill_diameter=6.0,
        path_to_working_dir=None,
        available_screwdrivers=["H5", "T6"],
    )
    return planner


def make_plan_screwing_adapter(
    planner: HierarchicalDisassemblySequencePlanner,
):
    """
    Wrap planner.plan_screwing into a (assembly, op) -> (op, new_assembly)
    function that ScrewingCapability expects.

    The ScrewingOperation 'op' already contains the screw connection and cost.
    We:
      - read the connection object from op.data["connection"],
      - call planner.plan_screwing(assembly, connection, cost_screwing, ...),
      - merge the planner's result back into op.data,
      - store the new Assembly in op.data["result_assembly"].
    """

    def plan_screwing_adapter(assembly: Assembly, op: ScrewingOperation):
        data = getattr(op, "data", {})
        if not isinstance(data, dict):
            return op, assembly

        connection = data.get("connection")
        if connection is None:
            return op, assembly

        cost_screwing = float(data.get("cost", 5.0))

        # HierarchicalDisassemblySequencePlanner.plan_screwing signature.
        planned_op, new_asm = planner.plan_screwing(
            assembly=assembly,
            connection=connection,
            cost_screwing=cost_screwing,
        )

        # Merge planner's data back into our op
        if hasattr(planned_op, "data") and isinstance(planned_op.data, dict):
            data.update(planned_op.data)

        # Expose new Assembly so ScrewingCapability.deterministic_transition can use it
        if isinstance(new_asm, Assembly):
            data["result_assembly"] = new_asm

        op.data = data
        return op, new_asm

    return plan_screwing_adapter


def make_plan_gripping_and_removing_adapter(
    planner: HierarchicalDisassemblySequencePlanner,
):
    """
    Wrap planner.plan_gripping_and_removing into a (assembly, op) -> (op, new_assembly) function
    that GrippingAndRemovingCapability expects.

    The GrippingAndRemovingOperation 'op' already contains part_name and cost.
    We:
      - look up the Part object,
      - call planner.plan_gripping_and_removing(assembly, part, cost_gripping_and_removing, max_grip_width),
      - merge the planner's result back into op.data,
      - store the new Assembly in op.data["result_assembly"].
    """

    def plan_gripping_and_removing_adapter(
        assembly: Assembly, op: GrippingAndRemovingOperation
    ):
        data = getattr(op, "data", {})
        if not isinstance(data, dict):
            return op, assembly

        part_name = data.get("part_name")
        if part_name is None:
            return op, assembly

        part = assembly.get_part_by_name(part_name)
        if part is None:
            return op, assembly

        cost_gripping_and_removing = float(data.get("cost", 5.0))
        max_grip_width = 50.0 * 999.0  # large enough upper bound

        planned_op, new_asm = planner.plan_gripping_and_removing(
            assembly=assembly,
            part=part,
            cost_gripping_and_removing=cost_gripping_and_removing,
            max_grip_width=max_grip_width,
        )

        # Merge planner's data back into our op
        if hasattr(planned_op, "data") and isinstance(planned_op.data, dict):
            data.update(planned_op.data)

        # Expose new Assembly so GrippingAndRemovingCapability.deterministic_transition can use it
        if isinstance(new_asm, Assembly):
            data["result_assembly"] = new_asm

        op.data = data
        return op, new_asm

    return plan_gripping_and_removing_adapter


# ---------------------------------------------------------------------
# Main visualization sequence
# ---------------------------------------------------------------------


def main():
    asm = load_emo_sr_v2_assembly()
    settings = make_cell_settings()

    # 1) Build planner and adapter
    planner = make_planner(asm)
    plan_screwing_adapter = make_plan_screwing_adapter(planner)
    plan_gripping_and_removing_adapter = make_plan_gripping_and_removing_adapter(
        planner
    )

    # 2) Build capabilities
    screwing = ScrewingCapability(
        available_screwdrivers=["H5"],
        cell_settings=settings,
        operation_cls=ScrewingOperation,
        plan_func=plan_screwing_adapter,
    )
    gripping_and_removing = GrippingAndRemovingCapability.from_assembly(
        asm,
        available_grippers=["g1"],
        cell_settings=settings,
        operation_cls=GrippingAndRemovingOperation,
        plan_func=plan_gripping_and_removing_adapter,
    )

    # 3) Build env (Stage2-aware env)
    env = DisassemblyPOMDPEnv(
        assembly=asm,
        capabilities=[screwing, gripping_and_removing],
        cell_settings=settings,
        milling_penalty=0.0,
        initial_mounted_tool="H5",
        world_sampler=EOLUnifiedSampler(
            rules=load_stuck_part_rules(ROOT / "emo_sr_v2")
        ),
    )

    obs, info = env.reset(seed=0)

    # Make the sequence deterministic: "feasible ⇒ success, infeasible ⇒ no-op"
    screwing.execution_success_factor = lambda assembly, a: 1.0

    orig_get_success = gripping_and_removing.execution_success_factor

    def deterministic_success(assembly, a):
        p = float(orig_get_success(assembly, a))
        return 1.0 if p > 0.0 else 0.0

    gripping_and_removing.execution_success_factor = deterministic_success

    # Helper: step + render
    def step_and_render(action, label: str):
        print(f"\n=== {label} ===")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        print(f"remaining parts: {[p.name for p in env.assembly.parts.values()]}")

        scene = env.render()
        if isinstance(scene, trimesh.Scene):
            scene.show()
        elif isinstance(scene, (list, tuple)) and len(scene) > 0:
            # Capabilities return list[o3d.geometry.Geometry] via Operation.scene_o3d()
            o3d.visualization.draw_geometries(list(scene))
        else:
            print("No scene returned; printed text fallback instead.")
        return obs, reward, terminated, truncated, info

    # 4) Sequence:
    #    (a) unscrew all four screws
    #    (b) remove upper lid
    #    (c) remove rotor

    # (a) Unscrew 4 screws
    for i in range(4):
        action_unscrew = np.array(
            [0, 0, i], dtype=int
        )  # cap_idx=0, tool_idx=0, conn_idx=i
        step_and_render(action_unscrew, f"Step {i + 1}: unscrew screw {i + 1}")

    # (b) Remove upper lid
    upper_idx = gripping_and_removing.name_to_part_index["01_upper_lid"]
    action_grab_upper = np.array([1, 0, upper_idx], dtype=int)  # cap_idx=1, gripper=0
    step_and_render(action_grab_upper, "Step 5: grab upper lid")

    # (c) Remove rotor
    rotor_idx = gripping_and_removing.name_to_part_index["03_rotor_short"]
    action_grab_rotor = np.array([1, 0, rotor_idx], dtype=int)
    step_and_render(action_grab_rotor, "Step 6: grab rotor")


if __name__ == "__main__":
    main()
