# test_1

Five-part disassembly test product made from the supplied Assemble-Them-All OBJ files (`0.obj` through `4.obj`). Part `0` is the stationary base; parts `1`, `2`, `3` and `4` are removal targets. Connections represent geometric contacts; screw types and fastening relationships were not supplied and are not inferred from appearance.

From the repository root:

```sh
uv sync
uv run python cases/test_1/run_ui.py
```

The launcher materializes the bundled contact cache and opens the Product explorer at **http://localhost:8765** with this case already loaded. Click **Rebuild graph**. For parallel removal analysis, enable **Parallel**, select **2 robots**, keep **Minimum gap 50 mm**, and rebuild. Full teardown explores until only the base remains. Some pairs are deliberately too close for the 50 mm screen.

To use a different port:

```sh
uv run python cases/test_1/run_ui.py --port 8766
```

For command-line graph extraction:

```sh
uv run python cases/test_1/run_ui.py --prepare-only
uv run python examples/extract_disassembly_graph.py --product cases/test_1/model --parallel --robots 2 --output examples/runs/test_1
```

## Geometry and scale

`source/` retains the five uploads unchanged. `model/test_1/` contains copies uniformly scaled by **10**, without moving or independently centring any part. The prepared assembly is **80 × 80 × 100 mm**. This is a chosen synthetic test scale, not a claim about the original manufactured dimensions. OBJ does not encode physical units.

The XML and product setup are ready to load once the launcher has materialized their contact files. Numeric contact arrays are bundled as `contacts.json`; the launcher writes their local NumPy cache without rerunning detection or needing Git LFS downloads. This keeps this small test product portable. `import_manifest.json` records preparation settings and original file checksums.
