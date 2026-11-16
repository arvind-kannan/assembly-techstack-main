"""A product folder with its product model in an `xml` subfolder (the bolt-and-nut benchmark's layout) loads in
the Product explorer, the gym and the MRTAMP app alike: all find it through the loader rule (#273).
Seam: the explorer session; every test works on a temporary copy of the tiny test product."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from diss_assembly_dataclasses.assembly import Assembly

from disassembly_pomdp.env_builder import load_product
from mrtamp.exported_graph import load_export
from product_explorer.session import ExplorerSession

TINY = Path(__file__).resolve().parents[1] / "test_products" / "tiny_test_product"
STEM = "tiny_test_product"


@pytest.fixture(autouse=True)
def no_contact_detection(monkeypatch):
    """The test product ships its contact data, so nothing here may detect contacts."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("contact detection ran")

    monkeypatch.setattr(Assembly, "find_contacts", refuse)
    monkeypatch.setattr(Assembly, "check_pair_for_contact_new", refuse)


@pytest.fixture
def folder(tmp_path) -> Path:
    """The tiny test product with its model, meshes and contact data moved into `xml/`; the setup stays."""
    copy = tmp_path / STEM
    shutil.copytree(TINY, copy)
    (copy / "xml").mkdir()
    for name in (f"{STEM}.xml", STEM, f"{STEM}_data"):
        (copy / name).rename(copy / "xml" / name)
    return copy


def test_the_explorer_and_the_gym_load_the_same_product_model(folder):
    session = ExplorerSession()
    session.load(folder)

    assert session.xml == folder / "xml" / f"{STEM}.xml"
    assert sorted(session.assembly.parts) == sorted(load_product(folder)[0].parts)


def test_the_export_records_the_subfolder_path_and_loads_again(folder, tmp_path):
    session = ExplorerSession()
    session.load(folder)
    session.generate().join()
    assert session.job.error is None
    session.store()
    out = tmp_path / "out"
    session.export(out)

    export = load_export(out / "sequence_graph.json", product_folder=folder)

    assert export.provenance["product_model_xml"] == f"xml/{STEM}.xml"
    assert export.provenance["fingerprints"]["product_model_xml"] is not None
    assert export.warnings == ()
    assert export.product_model.xml_path == folder / "xml" / f"{STEM}.xml"
