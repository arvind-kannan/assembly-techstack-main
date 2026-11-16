# Assembly techstack

Simulation and planning stack for the robotic disassembly of end-of-life products, developed in the CRC 1574 *Circular Factory* at KIT (wbk, IMK). It models a product as parts and connections, extracts every possible disassembly sequence, plans the robot motions for each step, and decides under uncertainty (stuck or missing parts) with a POMDP gym.

Several theses build on this repo at once. Read [CONTEXT-MAP.md](CONTEXT-MAP.md) first: it names the four contexts, their vocabulary and where their code lives.

## Repository layout

| Path | Context | What it is |
|---|---|---|
| `crc_diss_assembly_dataclasses/` | Core model | `Assembly`, `Part`, connection types, free directions. Protected, git subtree. |
| `crc_disassembly_sequence_generator/` | Motion planning (+ legacy) | Operations and `plan_*` functions; legacy DASG app and planner search graph. Git subtree. |
| `src/disassembly_pomdp/` | Decision making, Sequence graph | POMDP env, capabilities, EOL belief, observer, graph extraction. |
| `src/motion_planning/` | Motion planning | Assembly plan functions (inserting, screwing in, gripping at the supply) beside the sequence generator's removal planners. |
| `src/product_explorer/` | Product explorer | Local server and browser client: load a product folder, set targets, generate, explore and export the sequence graph. |
| `src/mrtamp/` | MRTAMP app | Run controller, MRTAMP planners (v1, B1), step expansion, inspections, model-level cell stand-in, exported-graph input. |
| `examples/` | | Building the env, training / evaluating a Q-learning policy, extracting and navigating the sequence graph, line balancing, UI. |
| `tests/` | | `unit/`, `integration/`, internal benchmark cases. |
| `tools/`, `figures/`, `benchmark_cases/` | | Analysis and rendering scripts, thesis figures, benchmark products. |
| `emo_sr_v2/` | | The main benchmark product, the electric motor (today `EMO_SR_V2`; becomes `electric_motor` when it moves to the 3D files repository in #220). |
| `docs/` | | Glossaries (`contexts/`), decisions (`adr/`), agent conventions (`agents/`). |

## Setup

**New thesis student?** Follow the [startup guide](docs/student-guide.md): it clones only the area you need and takes you to a green draft merge request.

Python ≥ 3.10. Install [uv](https://docs.astral.sh/uv/), then from the repo root install what your role needs; versions come from `uv.lock`:

| You are | Install |
|---|---|
| Product explorer or MRTAMP app user | `uv sync --no-dev` |
| Full-stack developer (tests, CI tools, analysis and rendering tools) | `uv sync` |
| Cell user (real cell, camera: `zivid`, `opcua`, `cryptography`; needs the Zivid SDK) | add `--group hardware` |

Run tools inside the environment with `uv run …` (e.g. `uv run pytest`), or activate `.venv`. Developers also run `uv run pre-commit install` once, so commits get the same ruff checks as CI (ADR-0009), and `git config blame.ignoreRevsFile .git-blame-ignore-revs`, so `git blame` skips the bulk formatting commits. To `import disassembly_pomdp` outside pytest and the example scripts, also run `uv pip install -e crc_diss_assembly_dataclasses/src -e src`. Add a dependency with `uv add <package>` (or `uv add --dev`), which updates `uv.lock`.

On Windows run `git config core.longpaths true` before checkout (some test data paths are longer than 260 characters). Binary assets are stored in Git LFS: install [git-lfs](https://git-lfs.com) and run `git lfs pull`.

## Run

```sh
python -m pytest tests/unit          # unit tests only
python -m pytest -m "not slow and not geom and not gpu" -n 4   # fast suite = the CI gate (~1 min on 4 cores)
python -m pytest                     # everything, incl. slow and geom tests
python examples/train_q_agent.py     # train a Q-learning policy on the EMO motor
python examples/extract_disassembly_graph.py
python examples/run_product_explorer.py --product emo_sr_v2   # Product explorer in the browser
```

No install is needed: `import disassembly_pomdp` adds the repo root (for `crc_disassembly_sequence_generator`) and `crc_diss_assembly_dataclasses/src` to `sys.path`, so tests, examples and tools run from any shell without `PYTHONPATH`. `pytest.ini` lists the same folders.

## Documentation

- [CONTEXT-MAP.md](CONTEXT-MAP.md): contexts, relationships, legacy code, upstream repos.
- [docs/contexts/](docs/contexts/): one glossary per context. Use these terms in code, issues and theses.
- [docs/adr/](docs/adr/): architecture decisions and why they were made.
- [AGENTS.md](AGENTS.md): rules for coding agents (Claude Code reads it through `CLAUDE.md`).

## Credits

The line-balancing code (`graph_precedence.py`, `line_balancing.py`, `line_balance_report.py` in `src/disassembly_pomdp/`) is ported from Jan's `line_balance_disassembly.py`, received as a zip in 2026-10 (no other version known). Open question: the licence. Ask Jan before the code is made public outside the KIT GitLab.

## Contributing

- Work starts from a GitLab issue with an `area:` label ([labels](docs/agents/triage-labels.md)); merge requests use the template in `.gitlab/merge_request_templates/`.
- **Core model**: changes only through an `area:core-model` issue approved by a core maintainer ([ADR-0003](docs/adr/0003-core-model-change-process.md)). Everything else extends it from outside.
- **Motion planners** plug in through a capability's plan function ([ADR-0004](docs/adr/0004-plan-function-is-the-motion-planning-seam.md)).
- **Subtrees**: keep commits that touch `crc_diss_assembly_dataclasses/`, `crc_disassembly_sequence_generator/` or `crc_disassembly_sim/` separate; they are pushed back to their upstreams (see `CONTEXT-MAP.md`) with `git subtree push` ([ADR-0002](docs/adr/0002-sub-repos-as-git-subtrees.md)).
