"""The Product explorer's local server (#75): FastAPI + uvicorn, a thin browser client in `static/`.

Every command answers with the session view; the browser only renders it. Generate graph, the mesh
folder import and the mesh update run in a background thread and report progress over the `/ws/job` WebSocket.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from disassembly_pomdp.graph_extraction import DISASSEMBLED_STATES
from product_explorer.session import Conflict, ExplorerError, ExplorerSession

STATIC = Path(__file__).resolve().parent / "static"


class Folder(BaseModel):
    folder: str


class SaveTarget(BaseModel):
    folder: str | None = (
        None  # the target folder of the first save of an imported product
    )


class Import(BaseModel):
    folder: str
    depth: int = 0
    name: str | None = None
    tolerance: float = 1.0
    angle: float = -0.99


class UpdateCheck(BaseModel):
    folder: str
    depth: int = 0


class Update(UpdateCheck):
    add: list[str] = []  # new mesh files that join the product


class Setup(BaseModel):
    base_parts: list[str]
    target_parts: list[str]


class Edge(BaseModel):
    edge: str


class Node(BaseModel):
    node: str


class Rename(BaseModel):
    part: str
    name: str


class RulesFile(BaseModel):
    rules_file: str


class Properties(BaseModel):
    part: str
    material: str
    density: float


class Part(BaseModel):
    part: str


class Merge(BaseModel):
    parts: list[str]
    name: str


class PartPair(BaseModel):
    part1: str  # of a screw connection: the screw
    part2: str  # of a screw connection: the held part


class NewConnection(PartPair):
    kind: str
    screw_head: str | None = None


class ConnectionIndex(BaseModel):
    index: int


class EditConnection(PartPair, ConnectionIndex):
    screw_head: str | None = None


class CellFilePath(BaseModel):
    path: str


class Recording(BaseModel):
    on: bool


class OutPath(BaseModel):
    path: str


class ProductRootSetting(BaseModel):
    path: str | None = None  # None clears the app setting


class OutDir(BaseModel):
    out_dir: str


class Generate(BaseModel):
    extraction_depth: str = DISASSEMBLED_STATES


def create_app(
    session: ExplorerSession | None = None, default_out_dir: Path | None = None
) -> FastAPI:
    session = session or ExplorerSession()
    app = FastAPI(title="Product explorer")
    app.state.session = session

    def run(command, *args):
        """Run a session command and answer with the new view."""
        _guarded(command, *args)
        return session.view()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state():
        return session.view() | {
            "default_out_dir": None if default_out_dir is None else str(default_out_dir)
        }

    @app.post("/api/product-root")
    def product_root(body: ProductRootSetting):
        return run(session.set_product_root, body.path.strip() if body.path else None)

    @app.get("/api/meshes")
    def meshes():
        return _guarded(session.meshes)

    @app.post("/api/load")
    def load(body: Folder):
        return run(session.load, body.folder)

    @app.post("/api/setup")
    def setup(body: Setup):
        return run(session.set_setup, body.base_parts, body.target_parts)

    @app.post("/api/cell")
    def cell(body: CellFilePath):
        return run(session.set_cell, body.path)

    @app.post("/api/store")
    def store():
        return run(session.store)

    @app.post("/api/import", status_code=202)
    def import_folder(body: Import):
        return run(
            session.import_folder,
            body.folder,
            body.depth,
            body.name,
            body.tolerance,
            body.angle,
        )

    @app.post("/api/update/check")
    def update_check(body: UpdateCheck):
        return _guarded(session.check_update, body.folder, body.depth)

    @app.post("/api/update", status_code=202)
    def update(body: Update):
        return run(session.update_meshes, body.folder, body.depth, body.add)

    @app.post("/api/cancel")
    def cancel():
        return run(session.cancel)

    @app.post("/api/save")
    def save(body: SaveTarget | None = None):
        return run(session.save, body.folder if body else None)

    @app.post("/api/rename")
    def rename(body: Rename):
        return run(session.rename_part, body.part, body.name)

    @app.post("/api/rename-by-rules")
    def rename_by_rules(body: RulesFile):
        return run(session.rename_parts_by_rules, body.rules_file)

    @app.post("/api/part/properties")
    def part_properties(body: Properties):
        return run(session.set_part_properties, body.part, body.material, body.density)

    @app.post("/api/part/delete")
    def part_delete(body: Part):
        return run(session.delete_part, body.part)

    @app.post("/api/merge")
    def merge(body: Merge):
        return run(session.merge_parts, body.parts, body.name)

    @app.post("/api/connection")
    def add_connection(body: NewConnection):
        return run(
            session.add_connection, body.kind, body.part1, body.part2, body.screw_head
        )

    @app.post("/api/connection/edit")
    def edit_connection(body: EditConnection):
        return run(
            session.edit_connection,
            body.index,
            body.part1,
            body.part2,
            body.screw_head,
        )

    @app.post("/api/connection/remove")
    def remove_connection(body: ConnectionIndex):
        return run(session.remove_connection, body.index)

    @app.post("/api/contacts/redetect")
    def redetect_contacts(body: PartPair):
        return run(session.redetect_contacts, body.part1, body.part2)

    @app.post("/api/undo")
    def undo():
        return run(session.undo)

    @app.post("/api/redo")
    def redo():
        return run(session.redo)

    @app.post("/api/generate", status_code=202)
    def generate(body: Generate | None = None):
        return run(session.generate, (body or Generate()).extraction_depth)

    @app.post("/api/step")
    def step(body: Edge):
        return run(session.step, body.edge)

    @app.post("/api/back")
    def back():
        return run(session.back)

    @app.post("/api/restart")
    def restart():
        return run(session.restart)

    @app.post("/api/jump")
    def jump(body: Node):
        return run(session.jump, body.node)

    @app.get("/api/rejection")
    def rejection(part: str):
        return _guarded(session.rejection, part)

    @app.post("/api/rec")
    def rec(body: Recording):
        return run(session.set_recording, body.on)

    @app.post("/api/rec/save")
    def rec_save(body: OutPath):
        return {"path": str(_guarded(session.save_recording, body.path))}

    @app.post("/api/export")
    def export(body: OutDir):
        return {"out_dir": str(_guarded(session.export, body.out_dir))}

    @app.websocket("/ws/job")
    async def job(ws: WebSocket):
        """Progress of Generate graph until it is done (one message if nothing runs)."""
        await ws.accept()
        last = None
        while True:
            current = session.job.as_dict()
            if current != last:
                await ws.send_json(current)
                last = current
            if not current["running"]:
                break
            await asyncio.sleep(0.25)
        await ws.close()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _guarded(command, *args):
    try:
        return command(*args)
    except Conflict as e:
        raise HTTPException(409, str(e)) from e
    except ExplorerError as e:
        raise HTTPException(400, str(e)) from e
