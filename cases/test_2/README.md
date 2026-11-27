# test_2

An 18-part geometric disassembly test product from the supplied Assemble-Them-All OBJ files (`0.obj`–`17.obj`). Part **8**, the largest enclosed body by mesh volume, is the stationary base. The other **17 parts** are removal targets. Connections are geometric contacts only: no screw, screw-head, fixed-connection or feature labels were supplied or inferred.

Run from the repository root:

```sh
uv sync
uv run python cases/test_2/run_ui.py
```

The launcher prepares the bundled contact cache and opens **http://localhost:8765** with this product loaded. Select **Parallel**, **2 robots**, and **Minimum gap 50 mm**, then **Rebuild graph**. Use `--port 8766` if another UI is running.

With 17 removal targets, exhaustive graph generation can be expensive (up to 131,072 subsets of removable parts before feasibility pruning). For a quick first check, use **Set targets** to make parts **10** and **15** the targets and all other parts bases. Rebuild the graph, then restore base **8** and the remaining parts as targets for the full case. Freezing parts for this smaller check changes the planning problem; it is not a full disassembly result.

For command-line graph extraction:

```sh
uv run python cases/test_2/run_ui.py --prepare-only
uv run python examples/extract_disassembly_graph.py --product cases/test_2/model --parallel --robots 2 --output examples/runs/test_2
```

## Preparation and units

`source/` retains all 18 uploads byte for byte. `model/test_2/` contains copies uniformly scaled by **10**, with their assembled positions preserved and no individual part centring. The resulting assembly spans approximately **86.456 × 100 × 37.827 mm**. This is a synthetic test scale, not a recovery of original manufactured dimensions; OBJ files do not encode units.

`contacts.json` stores the precomputed numeric contact arrays as text. The launcher materializes the local NumPy cache, so no mesh preprocessing or Git LFS download is needed for this case. Use the launcher before opening `model/` directly through another entry point.

`import_manifest.json` records the original checksums, scale, contact settings, detected part pairs and any import warnings. Contact detection used temporary subdivided meshes; the stored prepared meshes keep the original triangulation. The model does not reproduce Assemble-Them-All's physics simulator, rotational search or fastening semantics.

## Import warning

The geometric importer reports **no contact for part 16**. Refining parts 9 and 16 to 1 mm triangle edges and trying both pair orders did not find an opposing-normal contact under the standard settings. Their supplied meshes nevertheless register an intersection in FCL. Their positions are retained, no connection type is invented, and normal collision checks remain enabled. This case is therefore a geometric stress test; successful full teardown of every supplied part is not assumed.
