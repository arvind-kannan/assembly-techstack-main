# Battery replacement product: build package

Workshop package for the ticket *Build the battery replacement product* (map *Assembly demo: angle grinder and battery on the Transformer cell*).
Spec: resolution of *Decide the battery replacement product requirements*. All coordinates are Yonghang model coordinates in mm (origin = tray centre, z = 0 at the tray bottom, lid underside at z = 178).

Files:
- `lid_cnc.dxf`: lid outline (R20 corners), 41× Ø4.8 holes, pin hole Ø8, pin slot 8×12. Layers `OUTLINE`, `HOLES_D4.8`, `PIN`. Units mm.
- `nut_positions.csv`: every slot nut (41 screws + 2 pins), by profile and distance from the profile's start end.
- `make_build_files.py`: makes both files from `battery_yonghang/` (re-run if the model changes).
- `drawing_lid.pdf`, `drawing_base_plate.pdf` (+ `.png` previews): technical drawings, A4 at 1:10, datum = lower-left corner seen from the top, outline dimensions + hole coordinate tables; made by `make_drawings.py`.
- `model/battery_replacement.xml` (+ `model/battery_replacement/` meshes, `model/battery_replacement_data/` contacts): the **product model** for Case 2, with paths relative to `model/` (load it from there). Parts: `tray` (frame, posts, base plate, slot nuts, pins), `lid` (plate with holes + pressure blocks), `module_1..8`, `lid_screw_01..41` (Yonghang numbering, the angle grinder's M4×16 mesh). One screw connection per screw into the tray; screw head `H5` is a placeholder until the drive is measured. Start product state = tray + modules; assembly sequence = place the lid along −z, then screw in the 41 screws. Rebuild with `uv run make_model_meshes.py` (here), then `python cases/battery_replacement/make_model.py` (repo root, repo venv).
- **For the MRTAMP app:** `python examples/export_demo_products.py battery_replacement` writes the sequence graph export (Product explorer export, about 1.5 minutes) to `examples/runs/demo_products/battery_replacement/`. Load it with `mrtamp.exported_graph.load_export` and the cell file `cases/transformer_cell.json` (real cell) or `cases/transformer_cell_simulation.json` (cell simulation, both robots screw in).
- `render_frame.py`: simple rendering of the whole build → `render_frame.png`, `render_exploded.png`, `render_assembled.png`, `battery_replacement.glb` / `battery_replacement_exploded.glb` (rotatable in Windows 3D Viewer).

## 1. Parts list

| # | Part | Qty | Spec / note |
|---|---|---|---|
| 1 | 40×40 profile, slot 8 (or slot 10, one series throughout) | 2 × 1200, 1 × 794, 1 × 560, 2 × 202, 1 × 124, 1 × 110, 8 × 128 (posts) | sawn square, ±0.2 on length; ~5.4 m in total |
| 2 | Hidden end connectors (central bolt in the core bore, into a side or bottom slot) | 8 frame joints + 8 post tops | **no brackets in the top slots**: the top slots carry the nuts |
| 3 | Base plate 10 mm, stepped like the frame (1200 × 874 main + 202 × 560 step) | 1 | MDF or alu composite, one piece; 8 × M8 through it into the post core bores |
| 4 | Lid: hard PVC 5 mm, opaque, matte | 1 | CNC-routed from `lid_cnc.dxf`, **never laser-cut** (HCl); sheet ≥ 1400 × 870; ~7.8 kg |
| 5 | M4 slot nuts for the profile series | 41 + 5 spare | |
| 6 | M5 slot nuts | 2 | for the pins |
| 7 | Locating pins Ø8, chamfered or ball-nosed tip, M5 threaded stud, ~10 mm above the profile | 2 | steel |
| 8 | Screws M4×16 | 41 + 10 spare | the angle grinder's original screw if magnetic steel, else ISO 14580 M4×16 T20 steel; **wait for the drive / magnet test in *Design and print the screw nest plate***; no washers |
| 9 | Printed nut-carrier strips | per slot run, see §3 | PETG |
| 10 | Printed pressure-plate blocks, 5 mm high | 5 | sizes in §4, bonded to the lid top |
| 11 | Printed inner modules ~250 × 250 × ≤150, hollow, top grip feature | 8 | needs a ≥ 256 mm build volume |
| 12 | Printed locating corners for the modules | 16 (2 per module, diagonal) | screwed or bonded to the base plate |

## 2. Frame (top ring)

The top slot centrelines run exactly under the screw lines. The frame **follows the lid's stepped outline**: the stepped end is the outer edge, so the tray outline from above is asymmetric (helps the perception find the orientation). The frame stands out 10 mm beyond the lid all round.

The **long profiles run through** (otherwise the corner screws at x = ±663 miss the slot), and so does the **right profile** (otherwise screw 19 at y = −247 would sit on a joint).

| Profile | Slot centreline | Runs from → to | Length |
|---|---|---|---|
| long_minus | y = −417 | x −701 → 499 | 1200 |
| long_plus | y = +417 | x −701 → 499 | 1200 |
| left | x = −681 | y −397 → 397 | 794 |
| right | x = +681 | y −287 → 273 | 560 |
| step_h_minus | y = −267 | x 459 → 661 | 202 |
| step_h_plus | y = +253 | x 459 → 661 | 202 |
| step_v_minus | x = +479 | y −397 → −287 | 110 |
| step_v_plus | x = +479 | y 273 → 397 | 124 |

At the stepped end the two short horizontal pieces butt into the right profile (their x = 459 end is the outer corner), and the two short vertical pieces butt between the long profile and the horizontal piece. The long profiles end flush with the vertical pieces' outer face (x = 499).

"Start end" in `nut_positions.csv` = the end with the lower coordinate (left end of the long profiles, the −y end of the others).

Height: base plate 10 + posts 128 + top ring 40 = **178** (tray top = lid underside in the model). Eight posts: the six outer corners (−681, ±417), (479, ±417), (681, −267), (681, 253), plus one at mid-span under each long profile (−101, ±417), which keeps the long spans stiff under the `SCREW_IN` push force.

Leave the **outer side slots of the top ring free**: the tray fixture on the cell (still to be decided) will probably clamp into them.

## 3. Nut-carrier strips

One printed strip per slot run, sitting in the top slot and holding each slot nut at its position from `nut_positions.csv` (±0.5 mm).
- The strip references the **profile start end** (an end stop that hooks onto the profile end face), not the previous nut.
- Long profiles need several segments: chain them with tongue-and-groove joints, the first one hooked on the start end.
- Under each nut leave a clearance hole of Ø5 × ≥ 4 mm deep: the screw comes ~3 mm through the nut.
- Measure the slot (opening, undercut, depth) on the actual profile before modelling, because the shape differs between vendors.
- The two M5 pin nuts go into the left and right strips at the pin positions.

## 4. Lid

1. Route from `lid_cnc.dxf` (outline + holes in one setup, so hole-to-pin positions are exact). Deburr the holes.
2. Bond the five pressure-plate blocks (5 mm high) on the **top face**, centred on y ≈ 0:

   | Block | x range | y range |
   |---|---|---|
   | 1 | −633 … −613 | −39.1 … 42.9 |
   | 2 | −398 … −362 | −37.1 … 42.9 |
   | 3 | −140 … −104 | −37.1 … 42.9 |
   | 4 | 118 … 154 | −37.1 … 42.9 |
   | 5 | 366 … 386 | −39.1 … 42.9 |

   Matte, not glossy (Zivid). Their spacing is irregular on purpose: it rules out a 180° mis-registration.

## 5. Inner modules

4 × 2 grid in the main bay, footprint 250 × 250, centres x = −512, −238, 36, 310 and y = ±174 (24 mm gaps in x, 98 in y; clear of the profiles and the stepped end). Height ≤ 150 leaves 18 mm under the lid. Each has a top grip feature for a later robot pick (shape: free for now, robot placing is out of scope). Two diagonal locating corners per module on the base plate.

## 6. Assembly order

1. Saw the profiles, tap/bore the core holes for the end connectors.
2. Print the strips, load the slot nuts, slide them into the top slots **before** closing the frame (the butted profiles block the slot ends).
3. Assemble the top ring on a flat table, check the diagonals of the main bay (equal within 1 mm).
4. Fit the posts, screw the base plate on.
5. Screw in the two pins (round pin left at y = −292, slot pin right at y = 105).
6. Fit the lid on the pins, then hand-drive all 41 screws once: every one must start by hand without forcing. Mark any that bind (move that nut in its strip, don't drill out the lid).
7. Remove the screws again, stock them in the nest plate.
8. Place the modules in their corners.

## 7. Acceptance checks (to close the ticket)

- [ ] Lid fits on the pins one way only (turned 180° it does not go on).
- [ ] All 41 screws start by hand and seat flush, no washers needed.
- [ ] Lid top height 183 ± 1 mm above the base underside, at the four corners and the middle (no sag over the frame).
- [ ] Pressure blocks at their positions ±1 mm.
- [ ] Weight lid ≈ 8 kg (log the actual value; it feeds the lid gripper decision).
- [ ] Photo of the built product + actual measured deviations added to the ticket.

## Assumptions (deviations from the requirements resolution → check)

- **Lid outline = the stepped Yonghang outline** (1:1), not a 1382 × 854 rectangle; 1382 × 854 is its bounding box.
- **Pin positions** chosen here: round (−681, −292), slot (681, 105), both on the screw lines between two screws.
- **Frame follows the stepped outline** (decided 2026-10-01, for an unambiguous orientation in perception), instead of the ~1402 × 874 rectangle in the requirements.
- **Frame construction** (8 posts + stepped base plate, no bottom ring) and the joint order are my proposals.
