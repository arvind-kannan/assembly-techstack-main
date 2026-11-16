# Angle grinder: product model for Case 1

Product model of the FEIN CG15-125BL as far as Case 1 of the map *Assembly demo: angle grinder and battery on the Transformer cell* needs it: the headless grinder (motor housing, rotor with bevel pinion and M6 nut, gear housing), the crown-gear subassembly `BG_Tellerrad_V1p` (modelled as one part) and the 4× M4×16 that hold it.

- `model/angle_grinder.xml`: the assembly; `model/angle_grinder/` part meshes, `model/angle_grinder_data/` contact data. Paths are relative to `model/`, so load it from there (`os.chdir`, as in `examples/build_env.py`).
- Copied from `tests/internal_cases/Modell 2-Plannungssoftware/Stl/Assembly 1/` (CAD coordinates, mm) with the absolute paths made relative; contacts and connections unchanged. Handle and M4×40 are not part of the case.
- **Start product state** of Case 1: all parts except `BG_Tellerrad_V1p` and the four `Schraube_M4x16_540_Schr*`. **Assembly sequence**: insert `BG_Tellerrad_V1p` along −z, then screw in `Schr1` → `Schr2` → `Schr3` → `Schr4`.
- The screw head `H5` is a placeholder from the source model; the real drive is measured in *Design and print the screw nest plate*.
- `model/product_setup.json`: the product setup of Case 1. Base parts are the headless grinder; the target part is `BG_Tellerrad_V1p`.
- **For the MRTAMP app:** `python examples/export_demo_products.py angle_grinder` writes the sequence graph export (Product explorer export) to `examples/runs/demo_products/angle_grinder/`. Load it with `mrtamp.exported_graph.load_export` and the cell file `cases/transformer_cell.json` (real cell) or `cases/transformer_cell_simulation.json` (cell simulation, both robots screw in).
- **Reassembly baseline (AT 270):** the default-depth export (the graph stops at the disassembled states) is stored in `tests/test_products/angle_grinder_baseline/` (graph and manifest only). Its one disassembled state is the grinder without `BG_Tellerrad_V1p` and the four screws; `tests/integration/test_angle_grinder_reassembly.py` reassembles from it. Regenerate it with `python examples/export_demo_products.py --baseline` after any change to the model or the product setup.
