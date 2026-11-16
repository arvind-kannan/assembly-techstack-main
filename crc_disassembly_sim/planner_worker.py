"""Planner workers: motion planners that run as their own process (techstack #212, ADR-0011 of the techstack).

Planners that need an NVIDIA GPU or Linux (cuMotion, cuRobo, OMPL) do not run inside the cell simulation. Each runs
as a planner worker in its own environment; the cell simulation's `planner_worker` client (`planner_worker_client.py`)
sends it one operation per call over the sim server's framing (a 16-byte big-endian length, then the pickle, both
ways, no ACK) and gets back the NC commands of its shares (ADR-0010 of the techstack).

This module is the worker side and the wire format. It imports nothing but `sim_server_protocol`, so a worker's own
venv (another Python, no pybullet) needs only these two files. A worker subclasses `PlannerWorker` and calls `serve`.

Requests (plain dicts, `protocol` is `PROTOCOL`):
  * `{"request": "plan_operation", "protocol": 1, "scene": <scene description>}` answers either
    `{"ok": True, "motions": {robot id: {"transit": <motion>, "share": <motion>}}}`, one entry per share with a path,
    or `{"ok": True, "infeasible": {"reason_code": ..., "robot": ..., "during": "transit" | "share", "text": ...}}`
    with a reason code from the closed list of infeasible operations (never direct motion as a fallback).
  * `{"request": "conflicts", "protocol": 1, "first": <segment>, "second": <segment>}` answers
    `{"ok": True, "conflict": bool}`: whether two robots' segments that overlap in time come too close.
A motion is `{"duration": s, "commands": [NC command dicts], "end_pose": pose or None,
"end_configuration": joint values or None, "conflict_data": anything plain}`; the client sends `conflict_data` back
unchanged in a segment, `{"robot", "id", "kind", "start", "end", "motion": conflict_data}`. A request the worker
cannot read, or a failure inside it, answers the sim server's error form `{"ok": False, "error", "message"}`.

The scene description (built by the client from the cell-assembly snapshot before the operation):
  * `operation`: the operation dict, all of its shares;
  * `robots`: every robot of the cell, `{"start_pose", "start_configuration"}` (both None at home),
    `"mounted_tool"`, `"holds"` (the part it holds or None), `"in_operation"`, and for a robot of the operation its
    cell dict (`nc_channel`, `tools`, `capabilities`);
  * `product`: `assembly_xml` (the parts present), `product_model_xml` (every part), `reference_part` and
    `reference_pose` (`x, y, z, rx, ry, rz` in the cell frame, or None);
  * `held_parts`: robot id -> the part it holds (not in `assembly_xml`; it moves with the robot);
  * `obstacles`: the cell's static obstacles as configured for the client, passed on unchanged.
Poses are in metres, as everywhere in the cell simulation.
"""
from __future__ import annotations

import pickle
import socket
from typing import Callable

from sim_server_protocol import LENGTH_BYTES, InvalidRequest, _recv_exact, serve_connection

PROTOCOL = 1


class PlannerWorker:
    """What a planner worker provides; `serve` answers the requests with it."""

    def plan_operation(self, scene: dict) -> dict:
        """`{"motions": ...}` or `{"infeasible": ...}` for the scene's operation (see the module docstring)."""
        raise NotImplementedError

    def conflicts(self, first: dict, second: dict) -> bool:
        """Whether two segments of different robots cannot run at the same time (paths too close)."""
        raise NotImplementedError


def dispatcher(worker: PlannerWorker) -> Callable[[dict], dict]:
    """The `dispatch(request) -> answer` that `serve_connection` runs for `worker`."""
    def dispatch(request: dict) -> dict:
        if request.get("protocol") != PROTOCOL:
            raise InvalidRequest(f"planner worker protocol {request.get('protocol')!r}; this worker speaks {PROTOCOL}")
        kind = request["request"]
        if kind == "plan_operation":
            scene = request["scene"]
            if not isinstance(scene, dict):
                raise InvalidRequest("scene must be a dict")
            return {"ok": True, **worker.plan_operation(scene)}
        if kind == "conflicts":
            return {"ok": True, "conflict": bool(worker.conflicts(request["first"], request["second"]))}
        raise InvalidRequest(f"unknown request {kind!r}")
    return dispatch


def serve(worker: PlannerWorker, host: str = "127.0.0.1", port: int = 0,
          ready: Callable[[tuple], None] | None = None) -> None:
    """Answer the cell simulation on `host:port`, one connection after the other, until the process ends.
    `ready` gets the bound address (port 0 picks a free one)."""
    with socket.create_server((host, port)) as server:
        if ready is not None:
            ready(server.getsockname()[:2])
        dispatch = dispatcher(worker)
        while True:
            conn, _ = server.accept()
            with conn:
                serve_connection(conn, dispatch)


def call(address: tuple[str, int], request: dict, timeout: float) -> dict:
    """Send one framed request to the worker at `address` and return its answer. Raises `OSError` when the worker
    cannot be reached or does not answer within `timeout` seconds (connecting and answering each)."""
    with socket.create_connection(address, timeout=timeout) as conn:
        payload = pickle.dumps(request)
        conn.sendall(len(payload).to_bytes(LENGTH_BYTES, "big") + payload)
        size = int.from_bytes(_recv_exact(conn, LENGTH_BYTES), "big")
        return pickle.loads(_recv_exact(conn, size))
