"""Client of the real cell simulation (`crc_disassembly_sim`): an adapter of `mrtamp.cell_simulation`.

Requests and answers are plain dicts (`to_dict` forms) over the framed protocol of the simulation server: a
16-byte big-endian length, then the pickle, in both directions, with no ACK. A request the client can tell is
malformed is rejected here as `invalid_request`, as the stand-in does; one the server rejects or fails on
(`{"ok": False, "error": ...}`) comes back as a rejection too. The planner is chosen once, not per request:
see `configure_planner`.
"""

from __future__ import annotations

import pickle
import socket
from typing import Callable

from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot
from diss_assembly_dataclasses.feedback import ReasonCode
from diss_assembly_dataclasses.resource import Cell

from disassembly_pomdp.product_model import ProductModel
from disassembly_pomdp.travel_time_table import TravelTimeTable
from mrtamp.cell_simulation import (
    FeasibilityAnswer,
    SimulationReport,
    SimulationRequest,
    as_group,
    group_problem,
    rejected_answer,
    rejected_report,
    request_problem,
)

LENGTH_BYTES = 16


class RealCellSimulation:
    """`connect()` gives a connected socket to the simulation server (one per call, or kept: see `keep_open`).
    `simulate` and `check_feasibility` send the XML of the product model they are given (from the planning
    request), from which an assembly run takes the parts it inserts."""

    produces_nc_programs = True

    def __init__(
        self,
        connect: Callable[[], socket.socket],
        keep_open: bool = False,
    ):
        self._connect, self._socket, self.keep_open = connect, None, keep_open

    @classmethod
    def at(
        cls, host: str = "127.0.0.1", port: int = 12345, **kwargs
    ) -> RealCellSimulation:
        return cls(lambda: socket.create_connection((host, port)), **kwargs)

    def configure_planner(self, name: str, options: dict | None = None) -> None:
        """Choose the simulation's motion planner (once per simulation session); raises if it refuses."""
        answer = self._call(
            {
                "request": "configure_planner",
                "planner": name,
                "options": dict(options or {}),
            }
        )
        if answer.get("ok") is False:
            raise ValueError(
                f"simulation refused planner {name!r}: {answer.get('message')}"
            )

    def simulate(self, request: SimulationRequest) -> SimulationReport:
        problem = request_problem(request)
        if problem is not None:
            return rejected_report(problem)
        wire = {
            "request": "simulate",
            "operation_plan": request.operation_plan.to_dict(),
            "start_snapshot": request.start_snapshot.to_dict(),
            "cell": request.cell.to_dict(),
            "target": request.target,
            "execution_mode": request.execution_mode,
            "product_model_xml": request.product_model.xml,
            "travel_times": request.travel_times.to_dict(),
        }
        answer = self._call(wire)
        if answer.get("ok") is False:
            return rejected_report(
                answer.get("message", ""), ReasonCode(answer["error"])
            )
        return SimulationReport.from_dict(answer)

    def check_feasibility(
        self,
        concurrent_group,
        snapshot: Cell_Assembly_Snapshot,
        cell: Cell,
        *,
        product_model: ProductModel,
        travel_times: TravelTimeTable,
    ) -> FeasibilityAnswer:
        concurrent_group = as_group(concurrent_group)
        problem = group_problem(
            concurrent_group, snapshot, cell, product_model, travel_times
        )
        if problem is not None:
            return rejected_answer(problem)
        answer = self._call(
            {
                "request": "check_feasibility",
                "concurrent_group": [op.to_dict() for op in concurrent_group],
                "snapshot": snapshot.to_dict(),
                "cell": cell.to_dict(),
                "product_model_xml": product_model.xml,
                "travel_times": travel_times.to_dict(),
            }
        )
        if answer.get("ok") is False:
            return rejected_answer(
                answer.get("message", ""), ReasonCode(answer["error"])
            )
        return FeasibilityAnswer.from_dict(answer)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _call(self, wire: dict) -> dict:
        conn = self._socket or self._connect()
        if self.keep_open:
            self._socket = conn
        try:
            payload = pickle.dumps(wire)
            conn.sendall(len(payload).to_bytes(LENGTH_BYTES, "big") + payload)
            size = int.from_bytes(_receive(conn, LENGTH_BYTES), "big")
            return pickle.loads(_receive(conn, size))
        except BaseException:
            self.close()  # a failed exchange leaves the connection in an unknown state: the next call reconnects
            conn.close()
            raise
        finally:
            if not self.keep_open:
                conn.close()


def _receive(conn: socket.socket, n: int) -> bytes:
    chunks = []
    while n:
        chunk = conn.recv(min(n, 1 << 20))
        if not chunk:
            raise ConnectionError(
                "the simulation server closed the connection mid-message"
            )
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)
