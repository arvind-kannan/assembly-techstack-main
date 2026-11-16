"""The sequence graph export (#104): node-link JSON with a schema version and a provenance header."""

import json
import os
import subprocess

import networkx as nx
import pytest

from disassembly_pomdp.graph_export import (
    SCHEMA_VERSION,
    UnsupportedSchemaVersion,
    export_graph,
    files_repository_root,
    fingerprint,
    fingerprint_of_cell,
    make_provenance,
    read_export,
)
from disassembly_pomdp.graph_extraction import DISASSEMBLED_STATES, FULL_TEARDOWN
from tests.mrtamp_scenarios import START_XML, lid_screwed_on_base_graph


@pytest.fixture
def product_folder(tmp_path):
    folder = tmp_path / "lid_screwed_on_base"
    (folder / "xml").mkdir(parents=True)
    (folder / "xml" / "model.xml").write_text(START_XML, encoding="utf-8")
    return folder


def test_export_round_trips_the_graph_with_tuple_nodes_and_parallel_steps(
    tmp_path, product_folder
):
    graph = lid_screwed_on_base_graph()
    graph.add_edge(
        ("base", "lid", "screw"),
        ("base", "lid"),
        process="unscrewing",
        tool="T6",
        target_part="screw",
        action_time=7.0,
    )  # a second step between the same two states
    provenance = make_provenance(
        "lid_screwed_on_base", product_folder, "xml/model.xml", export_folder=tmp_path
    )

    export_graph(graph, tmp_path / "graph.json", provenance)
    loaded, header = read_export(tmp_path / "graph.json")

    assert nx.utils.graphs_equal(loaded, graph)
    assert header == provenance


def test_export_has_a_schema_version_and_a_provenance_header(tmp_path, product_folder):
    provenance = make_provenance(
        "lid_screwed_on_base", product_folder, "xml/model.xml", export_folder=tmp_path
    )
    export_graph(lid_screwed_on_base_graph(), tmp_path / "graph.json", provenance)

    data = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))

    assert data["schema_version"] == SCHEMA_VERSION
    header = data["provenance"]
    assert header["product_name"] == "lid_screwed_on_base"
    assert (
        header["product_folder"] == "lid_screwed_on_base"
    )  # relative to the export folder
    assert header["path_bases"] == {
        "product_folder": "export_folder",
        "cell_file": None,
    }
    assert header["product_model_xml"] == "xml/model.xml"
    assert set(header["fingerprints"]) == {
        "product_model_xml",
        "product_setup",
        "cell_file",
    }
    assert header["fingerprints"]["product_model_xml"] == fingerprint(
        product_folder / "xml" / "model.xml"
    )
    assert (
        header["fingerprints"]["product_setup"] is None
    )  # no product setup file given
    assert {"repo_commit", "code_version"} <= set(header)


@pytest.mark.parametrize("depth", [DISASSEMBLED_STATES, FULL_TEARDOWN])
def test_the_provenance_records_the_extraction_depth(tmp_path, product_folder, depth):
    provenance = make_provenance(
        "lid_screwed_on_base",
        product_folder,
        "xml/model.xml",
        export_folder=tmp_path,
        extraction_depth=depth,
    )
    export_graph(lid_screwed_on_base_graph(), tmp_path / "graph.json", provenance)

    assert read_export(tmp_path / "graph.json")[1]["extraction_depth"] == depth


def test_an_unknown_extraction_depth_is_refused_in_the_provenance(
    tmp_path, product_folder
):
    with pytest.raises(ValueError, match="depth"):
        make_provenance(
            "p",
            product_folder,
            "xml/model.xml",
            export_folder=tmp_path,
            extraction_depth="halfway",
        )


def test_unknown_schema_version_is_refused(tmp_path, product_folder):
    export_graph(
        lid_screwed_on_base_graph(),
        tmp_path / "graph.json",
        make_provenance("p", product_folder, "xml/model.xml", export_folder=tmp_path),
    )
    data = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    data["schema_version"] = SCHEMA_VERSION + 1
    (tmp_path / "graph.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(UnsupportedSchemaVersion, match=str(SCHEMA_VERSION + 1)):
        read_export(tmp_path / "graph.json")


def test_a_file_without_a_schema_version_is_refused(tmp_path):
    (tmp_path / "old.json").write_text(
        json.dumps({"nodes": [], "edges": []}), encoding="utf-8"
    )

    with pytest.raises(UnsupportedSchemaVersion):
        read_export(tmp_path / "old.json")


def test_fingerprint_follows_the_content_and_ignores_line_endings(tmp_path):
    a, b, c = tmp_path / "a.xml", tmp_path / "b.xml", tmp_path / "c.xml"
    a.write_bytes(b"<a>\n</a>\n")
    b.write_bytes(b"<a>\r\n</a>\r\n")
    c.write_bytes(b"<a>\n<b/></a>\n")

    assert fingerprint(a) == fingerprint(b) != fingerprint(c)
    assert fingerprint(tmp_path / "missing.xml") is None


def _git_repo(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
        check=True,
    )
    return root


def _files_repository(tmp_path):
    """A stand-in 3D files repository with a product folder and a cell file in it."""
    repo = _git_repo(tmp_path / "crc_1574_3D_Files")
    folder = repo / "Lid" / "product"
    (folder / "xml").mkdir(parents=True)
    (folder / "xml" / "model.xml").write_text(START_XML, encoding="utf-8")
    cell = repo / "cells" / "cell.json"
    cell.parent.mkdir()
    cell.write_text("{}", encoding="utf-8")
    return repo, folder, cell


def test_product_folder_and_cell_file_in_the_files_repository_are_recorded_from_its_root(
    tmp_path,
):
    repo, folder, cell = _files_repository(tmp_path)

    header = make_provenance(
        "lid", folder, "xml/model.xml", cell_file=cell, export_folder=tmp_path / "out"
    )

    assert (
        header["product_folder"] == "Lid/product"
        and header["cell_file"] == "cells/cell.json"
    )
    assert header["path_bases"] == {
        "product_folder": "files_repository",
        "cell_file": "files_repository",
    }
    assert header["repo_commit"]  # information only
    assert header["fingerprints"]["cell_file"] == fingerprint_of_cell(cell)


def test_paths_outside_the_files_repository_are_relative_to_the_export_folder(
    tmp_path, product_folder
):
    cell = tmp_path / "cells" / "cell.json"
    cell.parent.mkdir()
    cell.write_text("{}", encoding="utf-8")

    header = make_provenance(
        "lid",
        product_folder,
        "xml/model.xml",
        cell_file=cell,
        export_folder=tmp_path / "out",
    )

    assert (
        header["product_folder"] == "../lid_screwed_on_base"
        and header["cell_file"] == "../cells/cell.json"
    )
    assert set(header["path_bases"].values()) == {"export_folder"}


def test_no_absolute_path_of_this_machine_is_in_the_provenance(tmp_path):
    repo, folder, cell = _files_repository(tmp_path)
    out = tmp_path / "out"
    for product, where in ((folder, out), (tmp_path / "elsewhere", out)):
        (product / "xml").mkdir(parents=True, exist_ok=True)
        (product / "xml" / "model.xml").write_text(START_XML, encoding="utf-8")
        export_graph(
            lid_screwed_on_base_graph(),
            where / "graph.json",
            make_provenance(
                "lid", product, "xml/model.xml", cell_file=cell, export_folder=where
            ),
        )
        text = (where / "graph.json").read_text(encoding="utf-8")
        assert (
            str(tmp_path) not in text and str(tmp_path).replace("\\", "/") not in text
        )
        assert str(tmp_path).replace("\\", "\\\\") not in text


def test_a_path_on_another_drive_is_recorded_as_the_bare_name(
    tmp_path, product_folder, monkeypatch
):
    def other_drive(*_):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(os.path, "relpath", other_drive)

    header = make_provenance(
        "lid", product_folder, "xml/model.xml", export_folder=tmp_path / "out"
    )

    assert header["product_folder"] == "lid_screwed_on_base"
    assert header["path_bases"]["product_folder"] == "unresolved"


def test_a_folder_in_another_git_repository_is_not_in_the_files_repository(tmp_path):
    other = _git_repo(tmp_path / "some_other_project")
    (other / "product" / "xml").mkdir(parents=True)
    (other / "product" / "xml" / "model.xml").write_text(START_XML, encoding="utf-8")

    header = make_provenance(
        "p", other / "product", "xml/model.xml", export_folder=tmp_path / "out"
    )

    assert header["product_folder"] == "../some_other_project/product"
    assert header["path_bases"]["product_folder"] == "export_folder"


def test_files_repository_root_looks_at_the_nearest_existing_ancestor(tmp_path):
    repo, folder, _ = _files_repository(tmp_path)

    assert files_repository_root(repo / "not" / "yet" / "there") == repo.resolve()
    assert files_repository_root(tmp_path / "out") is None
