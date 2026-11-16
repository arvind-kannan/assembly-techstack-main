"""The product root (#257): where it comes from and how export paths are recorded relative to it."""

import json

import pytest

from disassembly_pomdp.graph_export import (
    functional_roots,
    make_provenance,
    record_path,
    resolve_recorded_path,
)
from product_explorer.product_root import (
    APP_SETTING,
    CODE_DEFAULT,
    ENV_VAR,
    ENVIRONMENT,
    SETTING,
    SETTINGS_FILE,
    InvalidProductRoot,
    read_setting,
    resolve_product_root,
    write_setting,
)
from tests.unit.test_graph_export import _files_repository, _git_repo


@pytest.fixture
def config(tmp_path):
    return tmp_path / "config"


def test_the_code_default_is_the_files_repository_of_the_product(tmp_path, config):
    repo, folder, _ = _files_repository(tmp_path)

    root = resolve_product_root(folder, config, environ={})

    assert (root.path, root.source) == (repo.resolve(), CODE_DEFAULT)


def test_no_source_gives_no_root(tmp_path, config):
    assert resolve_product_root(tmp_path, config, environ={}) is None
    assert resolve_product_root(None, config, environ={}) is None


def test_the_environment_variable_wins_over_the_code_default(tmp_path, config):
    _, folder, _ = _files_repository(tmp_path)
    mine = tmp_path / "mine"
    mine.mkdir()

    root = resolve_product_root(folder, config, environ={ENV_VAR: str(mine)})

    assert (root.path, root.source) == (mine.resolve(), ENVIRONMENT)


def test_the_app_setting_wins_over_the_environment_variable(tmp_path, config):
    _, folder, _ = _files_repository(tmp_path)
    env, app = tmp_path / "env", tmp_path / "app"
    env.mkdir()
    app.mkdir()
    write_setting(app, config)

    root = resolve_product_root(folder, config, environ={ENV_VAR: str(env)})

    assert (root.path, root.source) == (app.resolve(), APP_SETTING)


def test_the_setting_is_a_json_file_in_the_configuration_directory_and_can_be_cleared(
    tmp_path, config
):
    mine = tmp_path / "mine"
    mine.mkdir()

    write_setting(mine, config)

    assert json.loads((config / SETTINGS_FILE).read_text(encoding="utf-8")) == {
        SETTING: str(mine.resolve())
    }
    assert read_setting(config) == str(mine.resolve())  # a new session reads it back
    write_setting(None, config)
    assert read_setting(config) is None


def test_a_root_that_is_not_an_absolute_existing_folder_is_refused(tmp_path, config):
    for bad in ("relative/folder", str(tmp_path / "missing")):
        with pytest.raises(InvalidProductRoot):
            write_setting(bad, config)
    assert read_setting(config) is None
    with pytest.raises(InvalidProductRoot):
        resolve_product_root(None, config, environ={ENV_VAR: "relative"})


def _tree(tmp_path):
    """repo/Lid/product with the product root at repo/Lid, plus the repo's cells folder."""
    repo, folder, cell = _files_repository(tmp_path)
    return repo.resolve(), (repo / "Lid").resolve(), folder, cell


def test_functional_roots_are_the_product_root_and_its_ancestors_up_to_the_repository(
    tmp_path,
):
    repo, lid, _, _ = _tree(tmp_path)
    deep = lid / "a"
    deep.mkdir()

    assert functional_roots(deep) == [deep.resolve(), lid, repo]
    assert functional_roots(repo) == [repo]


def test_functional_roots_outside_a_repository_are_the_product_root_alone(tmp_path):
    other = _git_repo(tmp_path / "other_project")
    assert functional_roots(other) == [other.resolve()]


def test_a_path_under_the_product_root_is_relative_to_it(tmp_path):
    repo, lid, folder, _ = _tree(tmp_path)

    assert record_path(folder, lid) == ("product", "product_root")


def test_a_path_under_a_higher_functional_root_names_that_root_by_its_level(tmp_path):
    repo, lid, folder, cell = _tree(tmp_path)

    assert record_path(cell, lid) == ("cells/cell.json", "product_root+1")
    deep = lid / "a" / "b"
    deep.mkdir(parents=True)
    assert record_path(cell, deep) == ("cells/cell.json", "product_root+3")
    assert record_path(lid / "x", deep) == ("x", "product_root+2")  # nearest root


def test_a_path_under_no_functional_root_is_absolute(tmp_path):
    repo, lid, _, _ = _tree(tmp_path)
    elsewhere = tmp_path / "elsewhere" / "p"
    elsewhere.mkdir(parents=True)

    text, base = record_path(elsewhere, lid)

    assert base == "absolute" and text == elsewhere.resolve().as_posix()


def test_outside_a_repository_only_the_product_root_is_a_functional_root(tmp_path):
    root = tmp_path / "products"
    (root / "p").mkdir(parents=True)
    (tmp_path / "cells").mkdir()

    assert record_path(root / "p", root) == ("p", "product_root")
    assert record_path(tmp_path / "cells", root)[1] == "absolute"


def test_a_recorded_path_is_rebuilt_from_the_product_root_of_another_machine(tmp_path):
    repo, lid, folder, cell = _tree(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    recorded = {p: record_path(p, lid) for p in (folder, cell, elsewhere)}

    copy = tmp_path / "other_machine" / "crc_1574_3D_Files"  # same layout, other place
    (copy / "Lid").mkdir(parents=True)
    for path, (text, base) in recorded.items():
        full = resolve_recorded_path(text, base, copy / "Lid")
        if base == "absolute":
            assert full == path.resolve()
        else:
            assert full == (copy / text if base.endswith("+1") else copy / "Lid" / text)


def test_the_provenance_with_a_root_records_relative_to_it_wherever_it_is_exported(
    tmp_path,
):
    repo, lid, folder, cell = _tree(tmp_path)
    headers = [
        make_provenance(
            "lid",
            folder,
            "xml/model.xml",
            cell_file=cell,
            export_folder=out,
            product_root=lid,
        )
        for out in (tmp_path / "out", tmp_path / "far" / "away" / "out")
    ]

    for header in headers:
        assert header["product_folder"] == "product"
        assert header["cell_file"] == "cells/cell.json"
        assert header["path_bases"] == {
            "product_folder": "product_root",
            "cell_file": "product_root+1",
        }
        assert str(lid) not in json.dumps(header)  # the root itself is never written
    assert headers[0] == headers[1]


def test_without_a_root_the_provenance_is_the_one_of_the_portable_export_paths(
    tmp_path,
):
    repo, folder, cell = _files_repository(tmp_path)

    header = make_provenance(
        "lid", folder, "xml/model.xml", cell_file=cell, export_folder=tmp_path / "out"
    )

    assert header["path_bases"] == {
        "product_folder": "files_repository",
        "cell_file": "files_repository",
    }


def test_the_server_sets_shows_and_clears_the_app_setting(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from product_explorer.server import create_app
    from product_explorer.session import ExplorerSession

    mine = tmp_path / "mine"
    mine.mkdir()
    client = TestClient(create_app(ExplorerSession(config_dir=tmp_path / "config")))
    assert client.get("/api/state").json()["product_root"]["path"] is None

    view = client.post("/api/product-root", json={"path": str(mine)}).json()
    assert view["product_root"]["path"] == str(mine.resolve())
    assert view["product_root"]["source"] == "app setting"
    assert view["product_root"]["env_var"] == ENV_VAR
    assert (
        client.post("/api/product-root", json={"path": "relative"}).status_code == 400
    )
    again = TestClient(create_app(ExplorerSession(config_dir=tmp_path / "config")))
    assert again.get("/api/state").json()["product_root"]["setting"] == str(
        mine.resolve()
    )
    view = client.post("/api/product-root", json={"path": None}).json()
    assert view["product_root"]["path"] is None
