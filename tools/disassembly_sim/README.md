# DisassemblySim server: setup and smoke test

`crc_disassembly_sim/` is the PyBullet server that the legacy HMI (`crc_disassembly_sequence_generator/dasg_hmi.py`) talks to over TCP. It is a git subtree of the KIT GitLab [crc-disassembly-sim](https://gitlab.kit.edu/kit/wbk/map/imk/sfb1574_kreislauffabrik/crc-disassembly-sim) `main` @ `d81ed0f` (see `docs/adr/0002`, re-added in #148). That `main` grew from the thesis snapshot (cuMotion, TPG, `WAITM`, dry/physics) and adds the robust server (#109) and the cell simulation for operation plans (`cell_simulation.py`: `simulate`, `check_feasibility`, `configure_planner`, #110). The setup and smoke test below were written for the earlier GitHub `main` @ `19ea74b`, which served exactly the 15 requests the HMI sends; they have not been re-run against `d81ed0f`.

## Setup (Windows)

```sh
uv venv --python 3.12 crc_disassembly_sim/.venv
uv pip install --python crc_disassembly_sim/.venv/Scripts/python.exe -r tools/disassembly_sim/requirements.txt
```

- **Python 3.12 or newer**: `DisassemblySim.py` nests double quotes inside f-strings.
- **pybullet has no Windows wheel**: it compiles from source (MSVC Build Tools needed, about 10 minutes).
- **Core model**: the sim needs `diss_assembly_dataclasses` at `24a0f35`, the version pinned by the original umbrella repo [WBK-Robotics/probabilistic_dissasembly_planner](https://github.com/WBK-Robotics/probabilistic_dissasembly_planner). It is installed as a package. The sim's `sys.path` entry `submodule_crc_diss_assembly_dataclasses/src` does not exist here, so the installed package is used.
- The default `pb_examples_path` (`C:\MaH\pybullet_industrial\examples`) is never read; no fix needed.

## Start

Run from the folder of the assembly you will load, because the case XMLs hold mesh paths relative to their `model/` folder:

```sh
cd cases/angle_grinder/model
../../../crc_disassembly_sim/.venv/Scripts/python.exe ../../../crc_disassembly_sim/DisassemblySim.py --server_active=True
```

The server opens a PyBullet window and listens on `127.0.0.1:12345` (hard-coded in `start_server`). It does not connect to the cell's OPC UA server unless a request needs it (`update_robots`, `get_zivid_image`).

## Smoke test

```sh
crc_disassembly_sim/.venv/Scripts/python.exe tools/disassembly_sim/roundtrip.py <absolute path to angle_grinder.xml> [part_name]
```

Sends `get_state`, `load_assembly` and `look_at_part` (default part `BG_Tellerrad_V1p`). Passing run on 2026-10-02: `running`, `assembly loaded`, then a 140 MB frame with `points` (2 332 800 × 3), `index_array` and `view_matrix`; `look_at_part` takes about 25 s.

## Known legacy problems

- Any exception while handling a request ends the whole server process; restart it.
- `load_assembly` with `use_loading_point=True` crashes (`orienti` unbound in `loadAssembly`). The HMI always sends `False`.
- `dasg_hmi.py` `receive_data` reads the reply with a single `recv(size)`, which can cut off a large `look_at_part` frame. `roundtrip.py` reads until the reply is complete.
