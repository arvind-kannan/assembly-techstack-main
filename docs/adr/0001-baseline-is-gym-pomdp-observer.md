# Baseline is gym `pomdp_observer`, imported squashed

The first import of this repo was a snapshot of `WBK-Robotics/pomdp_disassembly_gym` at `main@2c0f3a9` (2026-01-30), about 50 commits behind the work the team actually uses. We replaced it with the gym's `pomdp_observer` branch at `dabba42` (2026-09-17), which contains the POMDP observer, graph extraction, line balancing and the gripping / removing split. Later gym branches (e.g. `removing-capability`) arrive as their own merge requests.

The gym tree was imported as one squashed commit, not with its history: about 1.2 GB of the history reachable from `pomdp_observer` is old test GIFs, which would make every clone huge. Binary assets are stored in Git LFS. The generated preview galleries `tests/preview_multi_*/` (~120 MB of PNGs, referenced by no code) were left out; they remain in the gym. Provenance is the commit message and this ADR; changes flow back to the gym by replaying commits onto a gym branch.

## Considered options

- Full gym history: exact provenance, but ~1.2 GB per clone.
- History filtered of GIFs and figures: every commit hash changes, so it no longer lines up with the gym anyway.
