# Startup guide for thesis students

You know Python. You do not know this repository, GitLab CI, uv or our conventions yet. This page takes you from nothing to a draft merge request (MR) with a green pipeline, and then explains the rules you will meet during your thesis. You only ever need one **area** of the repository (for example motion planning or product sequencing), so you clone only that.

The commands were run on a Linux machine (the clone for every area, `uv sync`, the hooks, the tests, `uv add`, `diff-cover`, committing, tagging, merging `main`). Steps that only work on Windows are marked **(Windows walkthrough)** and are checked in a separate walkthrough, as is the PowerShell script. `git push` (of a branch or a tag) and the merge request need your own GitLab permissions, so they were not run against the real project. If one of those does not work for you, ask (see [Where to ask](#where-to-ask)).

## First-day checklist

Work through these five steps in order. The details below the checklist explain each one.

1. **Get access.** You need
   - a KIT GitLab login (sign in at `gitlab.kit.edu` with your KIT account),
   - the **Developer** role on the project `assembly-techstack`,
   - a **protected thesis branch** named after your topic, such as `thesis/gripping-capability`. A maintainer creates it and protects it with you as the one allowed to push and merge. The [drift job](#the-drift-issue) finds you through that rule,
   - your **bwSyncAndShare** folder for data and results.

   Ask your supervisor for all four; they are done by a maintainer, not by you.
2. **Install git, git-lfs and uv** (Windows), and turn on long paths in git. See [Install the tools](#install-the-tools).
3. **Clone your area.** Download the script for your shell, run it with the name of your area, and it does a partial clone, a sparse checkout of the core plus your area, and downloads the LFS files of those folders only:

   ```sh
   bash slim-clone.sh motion-planning thesis-repo --ref thesis/gripping-capability     # Git Bash
   ```

   ```powershell
   powershell -ExecutionPolicy Bypass -File slim-clone.ps1 motion-planning -Folder thesis-repo -Ref thesis/gripping-capability   # PowerShell
   ```

   Pick your area from the [areas table](#areas-and-folders). Leave out `--ref` / `-Ref` to start on `main`. See [Clone your area](#clone-your-area) for where to get the script.
4. **Set up the environment, install the hooks, run the tests of your area.**

   ```sh
   cd thesis-repo
   uv sync
   uv run pre-commit install
   git config blame.ignoreRevsFile .git-blame-ignore-revs
   uv run pytest -q -m "not slow and not geom and not gpu" <the tests of your area>
   ```

   The script prints the exact test command for your area at the end; the [areas table](#areas-and-folders) lists the same files.
5. **Commit, push, open a draft MR, read the pipeline.**

   ```sh
   git switch thesis/gripping-capability       # your protected branch (skip if the script checked it out)
   git add path/to/changed/files
   git commit -m "Describe what changed"
   git push origin thesis/gripping-capability
   ```

   Then open the project in the browser, choose **Merge requests → New merge request**, pick your branch, start the title with `Draft:` and create it. The pipeline starts by itself. See [Read the pipeline](#the-merge-gate-in-plain-words) for what each job means.

## Install the tools

On Windows install [Git for Windows](https://git-scm.com/download/win) (it brings Git Bash and git-lfs), then [uv](https://docs.astral.sh/uv/getting-started/installation/) **(Windows walkthrough)**:

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
git lfs install            # once per machine
git config --global core.longpaths true
```

On Linux install git, git-lfs and uv with your package manager (`apt install git git-lfs`; uv from the page above), then `git lfs install`.

Python is not installed by hand: `uv sync` downloads the Python version the project needs and creates the environment in `.venv`.

Long paths matter: some test data paths are longer than 260 characters, which Windows refuses without `core.longpaths`. The clone scripts set it for the clone, and you set it globally once so it also holds for tools you start yourself.

## Clone your area

The scripts are `tools/clone/slim-clone.sh` (Git Bash, Linux, macOS) and `tools/clone/slim-clone.ps1` (PowerShell). They use only git and git-lfs, so you can run them before anything else is installed. You do not need a clone to get them: open the file in the GitLab web page of the project (`main`, folder `tools/clone`), press **Download** (the "Download raw file" button), and run it from any folder.

```text
slim-clone.sh <area> [folder] [--url <repo url>] [--ref <branch>]
slim-clone.ps1 <area> [-Folder <folder>] [-Url <repo url>] [-Ref <branch>]
```

What they do:

1. `git clone --filter=blob:none --no-checkout`: a **partial clone**, which downloads the history but no file contents.
2. Read the area table (`tools/clone/areas.tsv`, one file for both scripts) and stop with the list of valid areas if yours is not in it. A name that is not an area, such as `list`, is the way to see them.
3. `git sparse-checkout set --cone`: a **sparse checkout** of the folders every area needs plus the folders of your area. Everything else stays out of your working folder.
4. `git checkout`: downloads the file contents of those folders, and the LFS files in them. LFS files outside them are never downloaded.

Download size and time, measured on 2026-10-27 on the lab box (a Linux machine inside the KIT network, so a fast connection: yours will take longer). "Download" is the size of the repository store (`.git`: history, the file contents of your folders, LFS objects); "On disk" is the whole folder with the checked-out files:

| Area | Download | On disk | Time |
|---|---|---|---|
| `core-model` | 90 MB | 266 MB | 12 s |
| `sequence-graph` | 54 MB | 196 MB | 5 s |
| `motion-planning` | 87 MB | 261 MB | 11 s |
| `decision-making` | 54 MB | 196 MB | 6 s |
| `mrtamp-app` | 90 MB | 266 MB | 13 s |
| `cell-simulation` | 528 MB | 1.2 GB | 11 s |
| `product-explorer` | 87 MB | 261 MB | 11 s |
| `cases` | 136 MB | 371 MB | 22 s |

The time is clone plus checkout on the box; it does not include `uv sync`, which downloads the Python packages.

A later `git pull` brings new LFS files of your folders by itself. To add a second area later, run `git sparse-checkout add <folder> ...` with its folders from the table.

## Areas and folders

The table is the one in `tools/clone/areas.tsv`; a test (`tests/unit/test_area_table.py`) checks that every folder and file in it exists, so a moved folder fails CI instead of failing you.

Every area gets: `crc_diss_assembly_dataclasses/` (core model, subtree), `crc_disassembly_sequence_generator/` (subtree), `src/`, `tests/unit`, `tests/integration`, `tests/contract`, `tests/test_env`, `tests/test_products`, `tools/`, `examples/`, `emo_sr_v2/`, `docs/` and the root files (`pyproject.toml`, `uv.lock`, `pytest.ini`, `README.md`, ...). On top of that:

| Area | What it is | Extra folders (besides the ones every area gets) |
|---|---|---|
| `core-model` | Core model: parts, connections, free directions (change process: ADR-0003) | `cases/`, `benchmark_cases/` |
| `sequence-graph` | Sequence graph extraction, precedence, line balancing | - |
| `motion-planning` | Operations, plan functions, feasibility checks | `cases/` |
| `decision-making` | POMDP gym: capabilities, belief, observer, reward | - |
| `mrtamp-app` | MRTAMP app: planners, run controller, cell stand-in | `benchmark_cases/`, `cases/` |
| `cell-simulation` | Cell simulation: the sim subtree and its client | `crc_disassembly_sim/` (subtree) |
| `product-explorer` | Product explorer: import, authoring, export | `cases/` |
| `cases` | Cases: products, cases, internal test cases | `cases/`, `benchmark_cases/`, `tests/internal_cases/` |

**Tests.** The unit tests are not split by area, so each area lists the **test files** that belong to it: the last column (`tests`) of `tools/clone/areas.tsv`, and the command the script prints at the end. The extra folders above are also the LFS paths you download. In a sparse clone you can run exactly those. **CI always runs the full fast suite** on every MR, whatever you cloned, so a change in your area that breaks another area's test shows up there. To run more locally, add folders with `git sparse-checkout add` and run the other test files.

The fast suite means: `-m "not slow and not geom and not gpu"`. The `slow` and `geom` tests (heavy geometry) run on `main` after a merge and are manual on MRs; the `gpu` tests need an NVIDIA GPU and never run in CI (you report their result in the MR).

The sim's own tests (`crc_disassembly_sim/tests`) need the sim's environment and are not part of the fast suite; see the sim's `README.md` if your thesis changes the cell simulation.

## Orientation

Read these before you change code:

- [README.md](../README.md): layout and setup.
- [CONTEXT-MAP.md](../CONTEXT-MAP.md): the contexts, their vocabulary and where their code lives.
- The glossary of your context in [docs/contexts/](contexts/), and use its terms in code, issues and your thesis.
- The ADRs for your area in [docs/adr/](adr/):

| Area | ADRs |
|---|---|
| core-model | [0002](adr/0002-sub-repos-as-git-subtrees.md) subtrees, [0003](adr/0003-core-model-change-process.md) core change process, [0006](adr/0006-core-grows-into-the-ppr-model.md) PPR model |
| sequence-graph | [0005](adr/0005-sequence-graph-is-extracted-from-the-env.md) graph is extracted from the env |
| motion-planning | [0004](adr/0004-plan-function-is-the-motion-planning-seam.md) plan function seam |
| decision-making | [0001](adr/0001-baseline-is-gym-pomdp-observer.md) baseline, [0005](adr/0005-sequence-graph-is-extracted-from-the-env.md) |
| mrtamp-app | [0007](adr/0007-mrtamp-seam-is-a-stateless-plan-call.md) stateless plan call, [0008](adr/0008-cell-simulation-sits-before-the-real-cell.md) cell simulation, [0010](adr/0010-cell-simulation-planner-plans-one-operation-and-returns-nc-commands.md), [0012](adr/0012-cell-description-holds-no-times.md) |
| cell-simulation | [0002](adr/0002-sub-repos-as-git-subtrees.md), [0008](adr/0008-cell-simulation-sits-before-the-real-cell.md), [0010](adr/0010-cell-simulation-planner-plans-one-operation-and-returns-nc-commands.md), [0011](adr/0011-gpu-and-linux-planners-run-as-planner-workers.md) planner workers |
| product-explorer | [0011](adr/0011-dasg-app-retired-product-explorer-replaces-it.md) DASG retired, [0011](adr/0011-product-folders-live-in-the-3d-files-repository.md) product folders |
| cases | [0011](adr/0011-product-folders-live-in-the-3d-files-repository.md) product folders |

Read [ADR-0009](adr/0009-ci-is-the-merge-gate-for-agents-and-theses.md) too: it is the reason for most of this page.

**The guardrails** (the same ones the coding agents follow, see [AGENTS.md](../AGENTS.md)):

- **The core model is protected.** `crc_diss_assembly_dataclasses/` changes only through an issue labelled `area:core-model` whose scope a core maintainer approved ([ADR-0003](adr/0003-core-model-change-process.md)). Otherwise extend it from your own context (a helper, an adapter, a subclass).
- **Subtrees go upstream.** `crc_diss_assembly_dataclasses/`, `crc_disassembly_sequence_generator/` and `crc_disassembly_sim/` are git subtrees. A commit that touches one of them contains nothing else ([ADR-0002](adr/0002-sub-repos-as-git-subtrees.md)).
- **Planners plug in through the plan function** (`plan_func` of a capability, [ADR-0004](adr/0004-plan-function-is-the-motion-planning-seam.md)) or, for MRTAMP planners, through the stateless `plan()` call ([ADR-0007](adr/0007-mrtamp-seam-is-a-stateless-plan-call.md)). You do not change capability or env code to add a planner.
- **No unchecked motion to the cell.** NC programs for the real cell come only from the cell simulation. Never add a silent fallback to unplanned motion ([ADR-0008](adr/0008-cell-simulation-sits-before-the-real-cell.md)).
- **Legacy stays as is.** The DASG app and the planner search graph are reference code: ask before changing them.

## The merge gate in plain words

Everything reaches `main` through a merge request that a green pipeline gates ([ADR-0009](adr/0009-ci-is-the-merge-gate-for-agents-and-theses.md)). The pipeline runs on the KIT shared runners (Linux, CPU only, one hour per job). Open the MR's **Pipelines** tab; each job is a box:

| Job | What it checks | If it is red |
|---|---|---|
| `lint` | `ruff` (style and common errors, through `pre-commit`) on the files your branch changes. A file you touch is reformatted as a whole. | Run `uv run pre-commit run --from-ref origin/main --to-ref HEAD` locally; it fixes most findings itself. Commit the result. |
| `absolute-paths` | No absolute paths (`/home/...`, `C:\...`) in tracked data files (XML, URDF, JSON, YAML, INI). Paths in data files are relative to the file. | The log names the file and line. Make the path relative. |
| `fast-tests` | The fast suite (`not slow and not geom and not gpu`), then at least **60 %** of the lines you changed under `src/` must be covered by tests (`diff-cover`). Examples, tools, figures, notebooks and the subtrees are not measured. | Open the job log, search for `FAILED` or for the `diff-cover` table. Reproduce it locally, below. |
| `slow-tests` | Slow and geom tests. **Manual** on MRs (press play when you want them); they run by themselves on `main` after a merge. | Only when you start it. |
| GPU tests | Never in CI. Run them on your machine and write the result into the MR. | - |

An MR that changes no code (only docs, ADRs, `*.md`, figures) runs the lint jobs only. A red `fast-tests` job: reproduce it with the same command CI uses.

```sh
uv sync --locked
uv run pytest -q -m "not slow and not geom and not gpu" -n 4            # the full fast suite (needs the whole repo; a sparse clone runs the tests of its area)
uv run pytest -q path/to/test_file.py::test_name                           # one failing test
uv run --only-dev diff-cover coverage.xml --compare-branch=origin/main --fail-under=60   # after: uv run pytest ... --cov --cov-report=xml
```

If coverage is the problem, write a test for the new behaviour. `# pragma: no cover` on a line or block is fine for code that cannot run in CI on purpose: a branch that needs the real cell, a GPU or a camera, or a defensive `raise` that no test can reach. It is not a way to skip tests for code that could be tested; reviewers read every one.

## Milestone MRs

- Open a **draft** MR early: it keeps the pipeline and the diff visible, and your supervisor can look. Mark it **ready** at a milestone, when your supervisor asks, and at the end.
- The reviewer expects: the MR template filled in (`.gitlab/merge_request_templates/Default.md`), the result of your GPU tests if you have any (they do not run in CI), tests for every new behaviour, and all discussion threads resolved.
- **A change to the core model needs a core maintainer's approval** ([ADR-0003](adr/0003-core-model-change-process.md)). Open an `area:core-model` issue that scopes the one behaviour first, and keep the MR to that scope.
- Merge requests go through a **merge train**: GitLab tests your MR once more on top of the newest `main`, together with the MRs ahead of it, and merges it only when that is green. **The maintainer merges thesis MRs**, not you.
- Smaller pieces can go to `main` as separate MRs, from their own branches off `main` (for example a bug fix you found on the way). Your thesis branch then catches up with the merged result.

## The drift issue

Every two weeks a scheduled job merges `main` into every thesis branch in a scratch copy and keeps one issue per thesis, titled **Thesis drift: thesis/<topic> vs main**, assigned to you. It lists conflicts, tests that fail after the merge, and `main` commits that touch files your thesis changed. Facts only; the issue closes itself while your branch is in sync.

**You choose when to catch up**, so you rework your code as rarely as possible. When you do:

```sh
git switch thesis/<topic>
git fetch origin
git merge origin/main       # resolve conflicts in your editor, then: git add <files>; git commit
uv sync
uv run pytest -q -m "not slow and not geom and not gpu" <the tests of your area>
git push origin thesis/<topic>
```

**No force-push and no rebase of the shared branch**: the branch is protected, and the merge commits stay in its history. If a merge goes wrong, `git merge --abort` returns to where you were.

## Results and data

- **Code, configs and seeds go in git.** Everything needed to rerun an experiment.
- **Data and results go to your bwSyncAndShare folder**, under a git tag that names the code state that produced them: `thesis-<name>-vN` (`thesis-gripping-capability-v1`, `-v2`, ...).
- **Nothing large goes into git or LFS.** No result tables, no videos, no trained models.

To tag a result:

```sh
git tag thesis-gripping-capability-v1
git push origin thesis-gripping-capability-v1
```

Put the result files in a folder `thesis-gripping-capability-v1` in bwSyncAndShare, so the tag and the folder carry the same name. At the end of the thesis the results are archived into the wbk project folders (your supervisor tells you where).

## Dependencies

Add a dependency with `uv add <package>` (`uv add --dev <package>` for test and tool packages), and **commit `pyproject.toml` and `uv.lock`** together. Never `pip install` into the project by hand: the next `uv sync` removes it, and CI does not have it.

## AI assistants

If you use a coding agent (Claude Code, Copilot, anything else), [AGENTS.md](../AGENTS.md) applies to it, and **the same gate applies to its output**: you read the diff, you run the tests, you answer for the MR. Point the agent to `AGENTS.md` at the start of a session.

## End of thesis

- The final MR is ready, green, and merged by the maintainer.
- The final result tag is set (`thesis-<name>-vN`) and the results are archived.
- A short `README` for your area (in your thesis folder or next to your code) says what you built, how to run it and what is open, so the next student can continue.

## Where to ask

- A question about the code or a bug: an issue in the project, with the `area:` label of your area (`area:core-model`, `area:sequence-graph`, `area:motion-planning`, `area:decision-making`, `area:product-explorer`, `area:mrtamp-app`, `area:env-setup` for packaging, dependencies and CI). Cell simulation and cases have no `area:` label: ask in the project issues with the label of the closest area, and put issues about the sim's own code into the `crc-disassembly-sim` project. The labels are listed in [triage-labels.md](agents/triage-labels.md).
- A question about your thesis: your supervisor.

## Pitfalls

- **Windows long paths.** An error like `Filename too long` or a missing test file means `core.longpaths` is off: `git config --global core.longpaths true`, then check out again.
- **LFS pointer files.** A file that starts with `version https://git-lfs.github.com/spec/v1` and a `sha256` line is an LFS *pointer*: the real content was not downloaded (a folder outside your area, or `git lfs install` was missing). Fix it with `git lfs install`, then delete the folder and check it out again: `rm -rf path/to/folder; git checkout -- path/to/folder` (PowerShell: `Remove-Item -Recurse -Force path/to/folder`). Do not use `git lfs pull` in a slim clone: in our measurement it took more than seven minutes for 20 MB. A folder you do not need stays a pointer on purpose.
- **`pre-commit` fails with `patch does not apply`** **(Windows walkthrough)**. `pre-commit` hides the changes you did not `git add` while it runs and puts them back afterwards; when a hook changes a file that also has hidden changes, putting them back fails. Stage everything you want to commit (`git add <files>`) before `git commit`, and keep `git status` clean of unrelated changes. The case-only duplicate robot descriptions that used to make every Windows checkout dirty are gone from `main`; if a fresh checkout still shows modified files in `git status`, tell your supervisor instead of committing them.
- **No GPU.** Tests with the `gpu` marker are skipped by the fast suite. Do not delete the marker to make a test run.
