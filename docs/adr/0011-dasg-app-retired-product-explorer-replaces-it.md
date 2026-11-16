# The DASG app is retired; the Product explorer replaces it

The DASG app (the Tkinter operator GUI `dasg_hmi*.py` in the `crc_disassembly_sequence_generator/` subtree, with its planner menus, Q-learning, demos, perception and Sinumerik link) is retired. Its authoring and viewing functions are rebuilt, not ported, in the **Product explorer**: a local FastAPI server with a thin browser client in `src/product_explorer/`, outside the subtrees ([ADR-0002](0002-sub-repos-as-git-subtrees.md)), that edits the product model only through core calls ([ADR-0003](0003-core-model-change-process.md)). Its v1 is specified in #222. Running the cell is not the explorer's job; the cell link goes to the MRTAMP app (operator side).

Only the app is retired, not the subtree: `operations.py`, `planning.py` (the `plan_*` functions), `graspplanning.py`, `utility.py`, `colors.py` and `cell_settings.json` stay, because motion planning, the gym, the examples and the tests import them. The app code leaves in two removals:

1. **First removal**, once v1 of the explorer is built (#170): `dasg_hmi*.py`, Q-learning, demos, the legacy planner exports, the data copies (`EMO_SR_V2*`, `urdf_DM_EMO`, `planner_export`, `rivetdata`, `gmdata`, `.pkl` files) and the launch files; then #52 drops the duplicate EMO copy.
2. **Second removal**, once the operator side and the part-pose pipeline (#40) have taken what they need (#171): perception and cell-link code (`perception*.py`, `Zivid.py`, `Registration.py`, the OPC UA / sim client, `config.ini`).

Until then the !3 HMI fixes come to main as one subtree commit of their own (#169). Each removal is pushed with `git subtree push` to a branch on WBK-Robotics/crc_disassembly_sequence_generator; WBK decides on the merge and we pull afterwards.

Exports of the explorer (the sequence graph file and per-part URDFs) are **derived**: they go to a free output folder with a provenance header and are never committed to the 3D files repository (`crc_1574_3D_Files`), which holds the product folders as source data. A consumer that must keep an export stores its own copy with its results.

Decided in #80, #74, #75 and #77 on map #70; from the inventory in #71.

## Considered options

- **Fix and keep the DASG app**: it no longer runs here, mixes authoring with planner, Q-learning, perception and cell-link code in one ~5600-line Tkinter file, writes absolute Windows paths and has no undo; fixing it would cost about as much as a rebuild and keep the coupling.
- **Remove the whole subtree**: `operations.py` and the `plan_*` functions are still the motion-planning core of the gym and the plan functions ([ADR-0004](0004-plan-function-is-the-motion-planning-seam.md)); moving them out first is a larger change with an upstream on the other side.
- **Move the kept utilities out of the subtree now**: cleaner ownership, but it forks the code from WBK-Robotics before the upstream route is settled (map #155).
- **Remove everything in one go**: perception and the cell link are still the only reference for the Zivid pipeline and the Sinumerik link, which the operator side has not taken over yet.
- **A desktop toolkit (Qt) or a static HTML page for the new app**: Qt would rebuild the viewer, and a static page cannot import, edit or check available actions live (#75).
- **Commit exports next to the product folder**: exports can always be rebuilt and go stale with every edit; keeping them in the 3D files repository would mix source data with derived files.

## Consequences

- The `area:dasg-hmi` label stays for legacy issues (#10, #52) and is deleted after the second removal; new explorer work uses `area:product-explorer`.
- Plain-text OPC UA credentials in `config.ini` stay until the second removal (airgapped network).
- Each removal is a subtree-only commit, so it can be pushed upstream as is (ADR-0002).
- Gym and MRTAMP app read a product folder written by the explorer directly; the product model XML is not an export.
- Consumers check an export against its provenance fingerprints (product model, product setup, cell file) and decide themselves what a mismatch means; the explorer shows a stale sequence graph.
