# CI is the merge gate for agent and thesis MRs

Everything reaches `main` through a merge request that a green pipeline gates, whether an agent session or a thesis student wrote it. Agents keep one ticket → one MR. Students work on a branch named after their topic, such as `thesis/gripping-capability`, in this repo and open an MR at a milestone, on request or at the end; a draft MR from early on keeps the pipeline and the diff visible. Agent MRs merge themselves through the merge train once green; the maintainer merges thesis MRs.

The pipeline (`.gitlab-ci.yml`) runs on the KIT shared runners (Linux, CPU, 1 h per job). **The gate:** ruff (`pre-commit`) on the files the MR changes, the fast tests (`not slow and not geom and not gpu`), and at least 60 % of the changed lines under `src/` covered (`diff-cover`). An MR that touches no code path (only docs, ADRs, `*.md`, figures) runs ruff only. The merge train runs this fast stage only. Slow and geom tests are a manual job on MRs; on `main` they run by themselves after every merge that touches a code path, and a failing `main` pipeline opens one `ci::main-red` issue (or adds a note to the open one); GPU tests (marker `gpu`) never run in CI and are reported in the MR. Dependencies are locked in `uv.lock` (`pyproject.toml`; groups `dev` and `hardware`).

Merging is guarded by the project settings: pipelines must succeed, all threads resolved, merged-results pipelines and merge trains (so MRs that pass alone cannot break `main` together). The merge method is a merge commit, not fast-forward: thesis branches keep their history, including the drift job's merges, and are never rebased. Agent MRs are squashed (the orchestrator sets it), so `git log --first-parent main` shows one entry per MR; thesis and student MRs keep their commits unless someone ticks squash. On `main`, Maintainers and the group bot may merge; nobody force-pushes or rewrites `main`. No approval is required: CODEOWNERS only suggests reviewers. Thesis branches are protected: only that student and Maintainers push, no force-push.

**No absolute paths in data files (AT 168).** The check-stage job `absolute-paths` runs next to `lint` on MR pipelines (merge trains included) and thesis branches, and scans every tracked XML, URDF, SDF, xacro, JSON, YAML and INI file for absolute paths (POSIX home, user, mount, temp and opt roots, Windows drive paths, `file://` URIs; URLs, `package://` URIs and XML comments pass). Data paths are relative to the file's own folder, so cases run on every machine and in CI. The script is `tools/ci/check_absolute_paths.py` (standard library only; runs locally the same way, and as the pre-commit hook `check-absolute-paths`). Its `EXCLUDES` list holds the exceptions, each with its reason: IDE settings folders, the agent log folder, the cell simulation's docker-compose X11 mount, lock files. Main is covered through the merge train.

**Thesis drift.** Every two weeks a scheduled job merges `main` into each thesis branch in a scratch worktree and keeps one issue per thesis, assigned to the student, up to date: conflicts, fast tests failing after the merge, and main commits touching the thesis's files. Facts only. Students choose when to catch up, so they rework their code as rarely as possible; the issue closes itself while the branch is in sync.

**Thesis results** stay out of git: data and results go to the student's bwSyncAndShare folder under a git tag `thesis-<name>-vN`, later into the wbk project folders, or into a repo of their own if others build on them.

**Agents** run ruff and the fast tests locally, open a draft MR, watch its pipeline (`glab ci status`, `glab ci trace`), fix up to about three times, and mark the MR ready only when it is green; otherwise they ask. When the land checks pass, the orchestrator sets the MR to merge through the train (squashed) and GitLab merges it; nobody clicks merge. A red pipeline or an MR dropped from the train goes back to the agent, then to an Opus session, then to the user. A branch catches up with `git merge origin/main`, never a rebase or force-push.

Decided in #163; auto-merge through the merge train, merge commit with squash for agent MRs, and no required approvals decided by the user on 2026-10-22 (#253).

## Considered options

- **Advisory pipeline, or a hard gate for agents only**: students' unfinished milestones could land, and agents would lose a clear "done".
- **Fork per student, or a package per thesis**: a fork is just as big and hides drift; splitting the core into released packages is a large refactor. One repo with a slim clone (partial clone, sparse checkout, LFS on demand) keeps cross-area changes easy.
- **Merged-results pipelines or "rebase required" instead of merge trains**: weaker with many parallel agent MRs.
- **Fast-forward or semi-linear merges (linear `main`)**: both need the source branch rebased, which protected thesis branches with merge commits cannot take; squashing agent MRs gives a readable `main` without it.
- **The maintainer merges every MR, or CODEOWNERS approval for core and architecture paths**: with many agent MRs the user becomes the bottleneck; a green pipeline is the gate, and an escalation reaches the user when it is not.
- **Slow tests in the merge train**: several minutes per train car; they run on `main` after the merge, and a red `main` is reported at once.
- **Own lab GPU runner, SCC project runner or HoreKa**: GPU code is tested locally instead, until a GPU thesis makes a runner worth its upkeep.
- **Ruff on the whole repo after one cleanup MR**: a repo-wide diff would conflict with every open thesis branch. Old files get clean when someone touches them.
- **Coverage reported only, or "total must not drop"**: too weak, or noisy on small MRs.
- **Claude in CI for drift fix hints or MR review**: needs an API key in CI; not now.
- **Per-area test jobs (`rules:changes`)**: the fast suite takes about 3 min, so one job runs it all. Split it by area when it passes 5 min.

## Consequences

- An MR that touches an old, unformatted file formats that whole file; under-tested new code needs tests or a justified `# pragma: no cover`.
- Thesis branches are named by topic (#165). The drift job assigns its issue to the users the branch's protection rule lets push, so protecting a new thesis branch with the student under "Allowed to push and merge" is what makes the issue reach them.
- The drift job needs a project access token with `api` scope as the protected, masked variable `DRIFT_TOKEN`, and a schedule on `main` with `DRIFT_CHECK=1`. The `main-red-issue` job uses the same token unless `MAIN_RED_TOKEN` is set.
- A bad agent MR can now reach `main` without a human look; the slow suite and the `ci::main-red` issue are what catch it, so a red `main` is fixed before new agent work lands.
- The jobs are written inline here; move them into a shared CI component when `crc-disassembly-sim` or a thesis repo needs the same gate.
