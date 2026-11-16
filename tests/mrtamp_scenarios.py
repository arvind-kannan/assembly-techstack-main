"""The small lid-on-base scenario shared by the MRTAMP run, planner and cell simulation tests."""

import atexit
import functools
import shutil
import tempfile
from pathlib import Path

import networkx as nx
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell, Robot, SupplySlot, Tool

from disassembly_pomdp.product_model import ProductModel
from disassembly_pomdp.travel_time_table import TravelTimeTable
from mrtamp.assembly_cell import with_assembly_capabilities

START_XML = """<assembly name="lid_screwed_on_base">
	<part name="base" mesh_path="base.stl" density="8000.0" material="Generic" />
	<part name="lid" mesh_path="lid.stl" density="8000.0" material="Generic" />
	<part name="screw" mesh_path="screw.stl" density="8000.0" material="Generic" />
	<connection type="screw" part_name1="screw" part_name2="lid" screw_head="T6" />
	<connection type="generic" part_name1="base" part_name2="lid" />
</assembly>"""

ALL = ("base", "lid", "screw")


@functools.cache
def product_model(xml: str = START_XML) -> ProductModel:
    """`xml` as the product model of a product folder of its own (a temporary folder, one per XML and process)."""
    folder = Path(tempfile.mkdtemp(prefix="product_model_"))
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    (folder / "product.xml").write_text(xml, encoding="utf-8")
    return ProductModel.find(folder)


def step(process, tool, target, action_time):
    return {
        "process": process.value,
        "tool": tool,
        "target_part": target,
        "action_time": action_time,
    }


def lid_screwed_on_base_graph() -> nx.MultiDiGraph:
    """Unscrew, then grip and remove the lid; or mill the lid off directly (slower)."""
    g = nx.MultiDiGraph()
    g.add_edge(ALL, ("base", "lid"), **step(Process.UNSCREWING, "T6", "screw", 5.0))
    g.add_edge(
        ("base", "lid"),
        ("base",),
        **step(Process.GRIPPING_AND_REMOVING, "H5", "lid", 8.0),
    )
    g.add_edge(ALL, ("base", "screw"), **step(Process.MILLING, "mill", "lid", 150.0))
    return g


def transformer_cell() -> Cell:
    return Cell(
        robots=(
            Robot(
                "robot_1",
                1,
                tools=("H5", "T6", "mill"),
                capabilities=(
                    Process.GRIPPING_AND_REMOVING,
                    Process.UNSCREWING,
                    Process.MILLING,
                ),
            ),
            Robot("robot_2", 2, tools=(), capabilities=(Process.INSPECTING,)),
        ),
        tools=(
            Tool("H5", Process.GRIPPING_AND_REMOVING),
            Tool("T6", Process.UNSCREWING),
            Tool("mill", Process.MILLING),
        ),
    )


def travel_times() -> TravelTimeTable:
    """The travel-time table of the scenario cells: a tool change takes 80 s, an inspection 10 s."""
    return TravelTimeTable(
        {
            "toolchange_time_assembly_to_storage": 10.0,
            "toolchange_time_at_toolstorage": 60.0,
            "toolchange_time_storage_to_assembly": 10.0,
            "inspection_time": 10.0,
        }
    )


def start_snapshot() -> Cell_Assembly_Snapshot:
    return Cell_Assembly_Snapshot(
        START_XML, CellState(mounted_tools={"robot_1": "H5", "robot_2": None})
    )


BASE_ONLY_XML = """<assembly name="base_only">
	<part name="base" mesh_path="base.stl" density="8000.0" material="Generic" />
</assembly>"""

SUPPLY_SLOTS = (SupplySlot("lid_slot", (0.0, 0.5, 0.1, 0.0, 0.0, 0.0), part="lid"),)


def assembly_cell() -> Cell:
    return with_assembly_capabilities(transformer_cell())


def base_only_snapshot() -> Cell_Assembly_Snapshot:
    """The start of an assembly from scratch: only the base part sits in the fixture."""
    return Cell_Assembly_Snapshot(
        BASE_ONLY_XML, CellState(mounted_tools={"robot_1": "H5", "robot_2": None})
    )


TWO_SCREWS_XML = """<assembly name="lid_with_two_screws">
	<part name="base" mesh_path="base.stl" density="8000.0" material="Generic" />
	<part name="lid" mesh_path="lid.stl" density="8000.0" material="Generic" />
	<part name="screw_1" mesh_path="screw.stl" density="8000.0" material="Generic" />
	<part name="screw_2" mesh_path="screw.stl" density="8000.0" material="Generic" />
	<connection type="screw" part_name1="screw_1" part_name2="lid" screw_head="T6" />
	<connection type="screw" part_name1="screw_2" part_name2="lid" screw_head="T6" />
	<connection type="generic" part_name1="base" part_name2="lid" />
</assembly>"""


def two_screws_graph(screw_2_time: float = 5.0) -> nx.MultiDiGraph:
    """Two independent unscrewings (either order), then the lid comes off (depends on both)."""
    g = nx.MultiDiGraph()
    full = ("base", "lid", "screw_1", "screw_2")
    only_2, only_1, bare = (
        ("base", "lid", "screw_2"),
        ("base", "lid", "screw_1"),
        ("base", "lid"),
    )
    g.add_edge(full, only_2, **step(Process.UNSCREWING, "T6", "screw_1", 5.0))
    g.add_edge(full, only_1, **step(Process.UNSCREWING, "T6", "screw_2", screw_2_time))
    g.add_edge(only_2, bare, **step(Process.UNSCREWING, "T6", "screw_2", screw_2_time))
    g.add_edge(only_1, bare, **step(Process.UNSCREWING, "T6", "screw_1", 5.0))
    g.add_edge(bare, ("base",), **step(Process.GRIPPING_AND_REMOVING, "H5", "lid", 8.0))
    return g


def two_screws_snapshot() -> Cell_Assembly_Snapshot:
    return Cell_Assembly_Snapshot(
        TWO_SCREWS_XML, CellState(mounted_tools={"robot_1": "T6", "robot_2": None})
    )


def two_robot_cell(robot_2_removes: bool = True) -> Cell:
    """Both robots can unscrew, grip and remove and mill; Robot 2 also inspects. Without `robot_2_removes` it only unscrews."""
    cell = transformer_cell()
    first, second = cell.robots
    capabilities = first.capabilities if robot_2_removes else (Process.UNSCREWING,)
    second = Robot(
        "robot_2", 2, tools=first.tools, capabilities=capabilities + second.capabilities
    )
    return Cell(robots=(first, second), tools=cell.tools)


def two_screws_snapshot_both_on_t6() -> Cell_Assembly_Snapshot:
    return Cell_Assembly_Snapshot(
        TWO_SCREWS_XML, CellState(mounted_tools={"robot_1": "T6", "robot_2": "T6"})
    )
