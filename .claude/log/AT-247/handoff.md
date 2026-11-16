# Handoff AT-247
Ticket: AT-247
Branch: ticket/247-format-repo-with-ruff
Last commit: a026be5 Hand off AT-247: coverage-gate decision pending (#247)
MR: https://gitlab.kit.edu/kit/wbk/map/imk/sfb1574_kreislauffabrik/assembly-techstack/-/merge_requests/85
Done: format commit 44e7ef5; force-exclude for subtrees; .git-blame-ignore-revs + README line; six F841 fixed by hand (confirmed by maintainer); review done and fixed; both parked decisions decided; branch up to date with main, no conflicts
Open: MR description still says 'Can't merge yet' (update was blocked by the auto-mode classifier); updated text in the job tmp dir mr85.md
Next step: Maintainer switches 'Pipelines must succeed' off, merges !85 with a merge commit (no squash), switches it on again
Test state: fast tests green 475 passed 2 skipped (CI too); lint green; diff-cover red 37% (accepted, one-time bypass)
Parked decisions: 20261020-204552-0c97 (decided), 20261020-205119-76ee (decided)
Continue with: none; merge by maintainer
