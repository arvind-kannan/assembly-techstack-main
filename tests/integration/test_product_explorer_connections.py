"""Connection authoring in the Product explorer (AT 242), on a copy of the tiny test product through the explorer's API.

Seam: the HTTP API and the session behind it. The available actions come from the re-check that follows each edit;
the rejection reason of a picked part comes from `/api/rejection`. Contact detection runs for real, on the tiny meshes.
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from product_explorer.server import create_app

TINY = Path(__file__).resolve().parents[1] / "test_products" / "tiny_test_product"
SCREW_HOLDING = ("screw", "04_screw", "01_base", "H5")


@pytest.fixture
def client(tmp_path) -> TestClient:
    folder = tmp_path / "tiny_test_product"
    shutil.copytree(TINY, folder)
    client = TestClient(create_app())
    assert client.post("/api/load", json={"folder": str(folder)}).status_code == 200
    return client


def _post(client, url, **body) -> dict:
    response = client.post(url, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _state(client) -> dict:
    return client.get("/api/state").json()


def _connections(view, kind=None, pair=None) -> list[tuple]:
    return [
        (c["type"], c["part1"], c["part2"], c["screw_head"])
        for c in view["connections"]
        if (kind is None or c["type"] == kind)
        and (pair is None or {c["part1"], c["part2"]} == set(pair))
    ]


def _index(view, kind, pair) -> int:
    return next(
        c["index"]
        for c in view["connections"]
        if c["type"] == kind and {c["part1"], c["part2"]} == set(pair)
    )


def _available(view) -> set[str]:
    return {a["target_part"] for a in view["recheck"]["available"]}


def _rejection(client, part) -> dict:
    return client.get("/api/rejection", params={"part": part}).json()


def test_screw_connection_is_saved_with_the_screw_first_whatever_the_pick_order(client):
    for first, second in (("04_screw", "03_cover"), ("03_cover", "04_screw")):
        view = _post(
            client,
            "/api/connection",
            kind="screw",
            part1=first,
            part2=second,
            screw_head="H6",
        )
        assert ("screw", "04_screw", "03_cover", "H6") in _connections(view)
        _post(client, "/api/undo")


def test_a_rejected_held_part_names_the_connections_that_hold_it(client):
    before = _state(client)
    assert before["screw_heads"] == ["H5", "H6"]
    view = _post(
        client,
        "/api/connection",
        kind="screw",
        part1="02_target",
        part2="01_base",
        screw_head="H6",
    )
    assert ("screw", "02_target", "01_base", "H6") in _connections(view)
    # the gym rejects the base part; its rejection reason names every connection that holds it
    rejected = _rejection(client, "01_base")
    assert not rejected["available"]
    for name in (
        "screw connection: 02_target <-> 01_base",
        "screw connection: 04_screw <-> 01_base",
    ):
        assert name in rejected["holding"] and name in rejected["text"]
    assert _post(client, "/api/undo")["connections"] == before["connections"]


def test_screw_head_must_be_supported_by_the_cells_screwdrivers(client):
    before = _state(client)
    response = client.post(
        "/api/connection",
        json={
            "kind": "screw",
            "part1": "04_screw",
            "part2": "03_cover",
            "screw_head": "T9",
        },
    )
    assert response.status_code == 400 and "H5, H6" in response.json()["detail"]
    assert _state(client)["connections"] == before["connections"]


def test_an_unknown_part_type_or_duplicate_connection_is_refused(client):
    refused = [
        {"kind": "glue", "part1": "04_screw", "part2": "03_cover"},
        {"kind": "fixed", "part1": "04_screw", "part2": "no-such-part"},
        {"kind": "fixed", "part1": "04_screw", "part2": "04_screw"},
        {"kind": "screw", "part1": "04_screw", "part2": "01_base", "screw_head": "H6"},
    ]
    for body in refused:
        assert client.post("/api/connection", json=body).status_code == 400, body
    assert client.post("/api/connection/remove", json={"index": 99}).status_code == 400
    geometric = next(
        c for c in _state(client)["connections"] if c["type"] == "geometric"
    )
    edit = {"index": geometric["index"], "part1": "04_screw", "part2": "03_cover"}
    assert client.post("/api/connection/edit", json=edit).status_code == 400


def test_safety_ring_and_fixed_connections_are_added_edited_and_removed(client):
    view = _post(
        client, "/api/connection", kind="fixed", part1="04_screw", part2="03_cover"
    )
    assert _connections(view, "fixed") == [("fixed", "04_screw", "03_cover", None)]
    assert not _rejection(client, "04_screw")["available"]
    index = _index(view, "fixed", ("04_screw", "03_cover"))
    view = _post(
        client, "/api/connection/edit", index=index, part1="02_target", part2="03_cover"
    )
    assert _connections(view, "fixed") == [("fixed", "02_target", "03_cover", None)]
    view = _post(client, "/api/connection/remove", index=index)
    assert _connections(view, "fixed") == []
    ring = _post(
        client,
        "/api/connection",
        kind="safety_ring",
        part1="02_target",
        part2="03_cover",
    )
    assert _connections(ring, "safety_ring") == [
        ("safety_ring", "02_target", "03_cover", None)
    ]
    assert _post(client, "/api/undo")["connections"] == view["connections"]


def test_removing_a_fixed_connection_makes_the_held_parts_available_again(client):
    before = _post(
        client, "/api/connection", kind="fixed", part1="02_target", part2="03_cover"
    )
    assert not {"02_target", "03_cover"} & _available(before)
    after = _post(
        client,
        "/api/connection/remove",
        index=_index(before, "fixed", ("02_target", "03_cover")),
    )
    assert {"03_cover"} <= _available(after)


def test_deleting_a_geometric_connection_removes_that_connection_only(client):
    before = _state(client)
    pair = ("03_cover", "04_screw")
    target = _index(before, "geometric", pair)
    view = _post(client, "/api/connection/remove", index=target)
    kept = [c for c in before["connections"] if c["index"] != target]
    assert _connections(view) == [
        (c["type"], c["part1"], c["part2"], c["screw_head"]) for c in kept
    ]
    assert _post(client, "/api/undo")["connections"] == before["connections"]


def test_redetecting_one_pair_replaces_only_its_geometric_connections(client):
    pair = ("01_base", "04_screw")
    view = _post(
        client, "/api/connection", kind="fixed", part1="02_target", part2="03_cover"
    )
    view = _post(
        client, "/api/connection/remove", index=_index(view, "geometric", pair)
    )
    assert _connections(view, "geometric", pair) == []
    redetected = _post(client, "/api/contacts/redetect", part1=pair[0], part2=pair[1])
    assert len(_connections(redetected, "geometric", pair)) == 1

    def rest(v):
        return sorted(
            c
            for c in _connections(v)
            if not (c[0] == "geometric" and set(c[1:3]) == set(pair))
        )

    assert rest(redetected) == rest(
        view
    )  # hand-made connections and other pairs are unchanged
    assert _connections(_post(client, "/api/undo"), "geometric", pair) == []


def test_loaded_screw_connections_with_no_or_both_screw_names_warn(client):
    assert _state(client)["screw_warnings"] == []
    view = _post(
        client,
        "/api/connection",
        kind="screw",
        part1="02_target",
        part2="03_cover",
        screw_head="H5",
    )
    assert len(view["screw_warnings"]) == 1 and "neither" in view["screw_warnings"][0]
    _post(client, "/api/rename", part="03_cover", name="03_screw_cover")
    view = _post(
        client,
        "/api/connection",
        kind="screw",
        part1="04_screw",
        part2="03_screw_cover",
        screw_head="H5",
    )
    assert (
        len(view["screw_warnings"]) == 1 and "both" in view["screw_warnings"][0]
    )  # the rename fixed the first


def test_every_edit_is_reverted_by_one_undo(client):
    pair = {"part1": "01_base", "part2": "04_screw"}
    edits = [
        (
            "/api/connection",
            {
                "kind": "screw",
                "part1": "04_screw",
                "part2": "03_cover",
                "screw_head": "H6",
            },
        ),
        (
            "/api/connection",
            {"kind": "fixed", "part1": "04_screw", "part2": "03_cover"},
        ),
        (
            "/api/connection",
            {"kind": "safety_ring", "part1": "02_target", "part2": "03_cover"},
        ),
        (
            "/api/connection/edit",
            {"index": 4, "part1": "02_target", "part2": "03_cover", "screw_head": "H6"},
        ),
        ("/api/connection/remove", {"index": 0}),
        ("/api/contacts/redetect", pair),
    ]
    for url, body in edits:
        before = _state(client)
        _post(client, url, **body)
        after = _post(client, "/api/undo")
        assert after["connections"] == before["connections"], url
        assert after["can_undo"] == before["can_undo"], url


def test_whole_product_contact_detection_is_not_offered(client):
    assert client.post("/api/contacts/detect").status_code in (404, 405)
