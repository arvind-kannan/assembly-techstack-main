"""The real cell simulation's request handling, served over an in-process socket pair, for contract and pipeline tests.

The simulation is the `crc_disassembly_sim/` subtree (ADR-0002); only its pybullet-free part (`cell_simulation.py`,
`sim_server_protocol.py`) is needed. Set `CRC_DISASSEMBLY_SIM_DIR` to test another checkout, e.g. a branch of the
upstream repo; the tests that use it are skipped if that folder has no `cell_simulation.py`.
The motion planner here is a test double (a straight polyline at a fixed feed, no collision model): the real
planners are registered in the simulation itself and chosen by `configure_planner`.
"""

import importlib
import os
import socket
import sys
import threading
from pathlib import Path

import pytest

SIM_DIR = os.environ.get("CRC_DISASSEMBLY_SIM_DIR") or str(
    Path(__file__).resolve().parents[1] / "crc_disassembly_sim"
)
SIM_AVAILABLE = (Path(SIM_DIR) / "cell_simulation.py").is_file()
requires_sim = pytest.mark.skipif(
    not SIM_AVAILABLE, reason=f"no cell_simulation.py in {SIM_DIR}"
)


def _sim_modules():
    if SIM_DIR not in sys.path:
        sys.path.append(SIM_DIR)
    return importlib.import_module("cell_simulation"), importlib.import_module(
        "sim_server_protocol"
    )


def planner_worker_simulation(address, options=None):
    """A `RealCellSimulation` wired to a fresh simulation service in this process whose motion planner is the
    simulation's own `planner_worker` client, configured for the planner worker at `address` (`"host:port"`)."""
    from mrtamp.sim_client import RealCellSimulation

    cell_simulation, _ = _sim_modules()
    client = importlib.import_module("planner_worker_client")
    service = cell_simulation.CellSimulationService()
    client.register(service)
    simulation = RealCellSimulation(_connector(service))
    simulation.configure_planner(
        "planner_worker", {"address": address, **(options or {})}
    )
    return simulation


def loopback_simulation(planner_options=None, configure=True):
    """A `RealCellSimulation` wired to a fresh simulation service in this process, with the test planner
    registered and (unless `configure` is False) configured; returns it together with the service.
    Planner option `conflicting`: pairs of segment ids that cannot run at the same time."""
    from mrtamp.sim_client import RealCellSimulation

    cell_simulation, _ = _sim_modules()

    class StraightLine(cell_simulation.MotionPlanner):
        """Length of the polyline over `feed` (path units per second); a transit costs `transit` seconds."""

        name = "straight_line"

        def __init__(self, options):
            self.feed, self.transit = (
                options.get("feed", 100.0),
                options.get("transit", 1.0),
            )
            self.conflicting = {tuple(pair) for pair in options.get("conflicting", ())}

        def plan_share(self, robot, operation, share):
            path = share["path"]
            length = sum(
                sum((b - a) ** 2 for a, b in zip(p, q)) ** 0.5
                for p, q in zip(path, path[1:])
            )
            return cell_simulation.ShareMotion(
                length / self.feed,
                [{"comment": f"{operation['id']} on {robot['id']}"}],
                tuple(path[-1]),
            )

        def plan_transit(self, robot, start_pose, goal_pose):
            return cell_simulation.ShareMotion(self.transit, [], tuple(goal_pose))

        def conflicts(self, first, second):
            return (first.id, second.id) in self.conflicting or (
                second.id,
                first.id,
            ) in self.conflicting

    service = cell_simulation.CellSimulationService()
    service.register_planner("straight_line", StraightLine)
    if configure:
        service.configure_planner("straight_line", planner_options)

    return RealCellSimulation(_connector(service)), service


def _connector(service):
    """`connect()` for a `RealCellSimulation`: a socket pair whose other end `service` answers in a thread."""
    protocol = _sim_modules()[1]

    def connect():
        client, server = socket.socketpair()
        threading.Thread(
            target=protocol.serve_connection, args=(server, service.handle), daemon=True
        ).start()
        return client

    return connect
