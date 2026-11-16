"""Build a DisassemblyPOMDPEnv for any product folder and cell (see #83)."""

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell as CellResources
from diss_assembly_dataclasses.resource import Robot

from crc_disassembly_sequence_generator.operations import (
    GrippingAndRemovingOperation,
    ScrewingOperation,
)
from disassembly_pomdp.config import CellSettings
from disassembly_pomdp.disassembly_pomdp_gym import DisassemblyPOMDPEnv
from disassembly_pomdp.eol_operator import EOLUnifiedSampler, PartEOLOperatorV2
from disassembly_pomdp.gripping_and_removing_capability import (
    GrippingAndRemovingCapability,
)
from disassembly_pomdp.product_model import ProductModel
from disassembly_pomdp.screwing_capability import ScrewingCapability
from disassembly_pomdp.stuck_part_rules import StuckPartRules
from disassembly_pomdp.travel_time_table import TravelTimeTable
from motion_planning.removal_planning import (
    gripping_and_removing_plan_func,
    unscrewing_plan_func,
)

PRODUCT_SETUP_FILE = "product_setup.json"


def default_cell_settings() -> CellSettings:
    """The cell timings and offsets used so far by the examples and tests."""
    return CellSettings(
        feed_rate_milling=1000.0,
        feed_rate_gripping_and_removing_approaching=2000.0,
        feed_rate_gripping_and_removing_leaving=1000.0,
        feed_rate_screwing_approaching=2000.0,
        feed_rate_screwing_leaving=1000.0,
        toolchange_time_dropoff_to_storage=10.0,
        toolchange_time_assembly_to_storage=10.0,
        toolchange_time_at_toolstorage=60.0,
        toolchange_time_start_to_toolstorage=10.0,
        toolchange_time_storage_to_assembly=10.0,
        traveltime_assembly_to_dropoff=15.0,
        traveltime_dropoff_to_assembly=15.0,
        screwing_start_offset=100.0,
        screwing_pullout_factor=1.1,
        screwing_minimal_pullout_value=5.0,
        gripping_start_offset=100.0,
        gripping_pullout_factor=2.1,
        gripping_minimal_pullout_value=5.0,
        milling_penalty=100.0,
    )


_SCREWING = (Process.UNSCREWING, Process.SCREWING_IN)
_GRIPPING = (
    Process.GRIPPING,
    Process.REMOVING,
    Process.GRIPPING_AND_REMOVING,
    Process.INSERTING,
    Process.RELEASING,
    Process.HOLDING,
)


def default_robots() -> tuple[Robot, ...]:
    """The core robots of the default cell: robot 1 screws and grips with H5 and the default gripper,
    robot 2 screws with H6."""
    return (
        Robot(
            "robot_1",
            1,
            tools=("H5", "default_gripper"),
            capabilities=_SCREWING + _GRIPPING,
        ),
        Robot("robot_2", 2, tools=("H6",), capabilities=_SCREWING),
    )


@dataclass
class Cell:
    """Tools, timings and core robots of a cell. Screwing needs screwdrivers, gripping-and-removing
    needs grippers; the robots that carry a tool and can carry out a capability's process act with it.
    A tool no robot carries gives no robot the capability: its actions are masked (`no_robot`)."""

    settings: CellSettings = field(default_factory=default_cell_settings)
    screwdrivers: tuple[str, ...] = ("H5", "H6")
    grippers: tuple[str, ...] = ("default_gripper",)
    robots: tuple[Robot, ...] = field(default_factory=default_robots)


def _read_setup(folder: Path) -> dict:
    setup_path = folder / PRODUCT_SETUP_FILE
    if not setup_path.exists():
        raise FileNotFoundError(f"{setup_path} not found")
    return json.loads(setup_path.read_text(encoding="utf-8"))


def cell_from_resources(
    resources: CellResources, travel_times: TravelTimeTable
) -> Cell:
    """The cell of a cell file (#245): its tools by process (screwing and unscrewing tools are the
    screwdrivers, gripping-family tools the grippers; inspecting and milling tools carry no capability),
    the times of its travel-time table over the default settings and its core robots (#230)."""
    names = {s.name for s in fields(CellSettings)}
    settings = replace(
        default_cell_settings(),
        **{k: v for k, v in travel_times.times.items() if k in names},
    )
    return Cell(
        settings,
        tuple(t.name for t in resources.tools if t.process in _SCREWING),
        tuple(t.name for t in resources.tools if t.process in _GRIPPING),
        resources.robots,
    )


def load_product(product_folder) -> tuple[Assembly, list[str], list[str]]:
    """Load the product model of a product folder (by the loader rule, `ProductModel.find`) plus the base and target parts of its product setup."""
    folder = Path(product_folder)
    model = ProductModel.find(folder)
    setup = _read_setup(folder)
    return (
        model.assembly(),
        list(setup.get("base_parts", [])),
        list(setup.get("target_parts", [])),
    )


def load_stuck_part_rules(product_folder) -> StuckPartRules:
    """The stuck-part rules of a product folder's product setup; none if the setup has no such field."""
    return StuckPartRules.from_setup(_read_setup(Path(product_folder)))


def build_env(
    product_folder,
    cell: Cell | None = None,
    base_parts: list[str] | None = None,
    target_parts: list[str] | None = None,
    assembly=None,
) -> DisassemblyPOMDPEnv:
    """Env for the product in `product_folder` on `cell` (default cell if None).

    `base_parts` / `target_parts` replace the stored product setup (e.g. one not yet stored).
    `assembly`, if given, is the product model already loaded (by the Product explorer's adapter, #238);
    the folder is then not read and base and target parts are taken as given (none if not passed);
    the stuck-part rules are then none too (the EOL probabilities are zero here anyway).
    Capabilities come from the cell's tools; they act with the cell's core robots. Env extras are fixed: uncertainty off, 50 steps,
    initial tool = first tool of the cell.
    """
    cell = cell or Cell()
    if assembly is None:
        assembly, stored_base, stored_target = load_product(product_folder)
        rules = load_stuck_part_rules(product_folder)
    else:
        stored_base, stored_target = [], []
        rules = StuckPartRules()
    base_parts = stored_base if base_parts is None else list(base_parts)
    target_parts = stored_target if target_parts is None else list(target_parts)

    capabilities = []
    if cell.screwdrivers:
        capabilities.append(
            ScrewingCapability(
                available_screwdrivers=list(cell.screwdrivers),
                cell_settings=cell.settings,
                operation_cls=ScrewingOperation,
                plan_func=unscrewing_plan_func,
                robots=cell.robots,
            )
        )
    if cell.grippers:
        capabilities.append(
            GrippingAndRemovingCapability.from_assembly(
                assembly,
                available_grippers=list(cell.grippers),
                cell_settings=cell.settings,
                operation_cls=GrippingAndRemovingOperation,
                plan_func=gripping_and_removing_plan_func,
                base_parts=tuple(base_parts),
                robots=cell.robots,
            )
        )
    if not capabilities:
        raise ValueError("The cell has neither screwdrivers nor grippers")

    return DisassemblyPOMDPEnv(
        assembly=assembly,
        capabilities=capabilities,
        cell_settings=cell.settings,
        target_part_names=target_parts,
        max_steps=50,
        truncation_penalty=200.0,
        world_sampler=EOLUnifiedSampler(
            p_stuck_screw=0.0, p_stuck_upper=0.0, p_missing_screw=0.0, rules=rules
        ),
        eol_op=PartEOLOperatorV2(freeze_updates=True),
        expose_eol_belief=False,
        belief_mode="fo",
        initial_mounted_tool=(cell.screwdrivers or cell.grippers)[0],
    )
