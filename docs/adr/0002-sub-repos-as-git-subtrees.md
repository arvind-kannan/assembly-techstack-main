# Sub-repos are git subtrees

`crc_diss_assembly_dataclasses/` and `crc_disassembly_sequence_generator/` (and, since #148, `crc_disassembly_sim/`, see below) are git subtrees with their full history, not submodules and not flat copies. In the gym they were submodules; branches silently re-pointed them to personal forks, and several diverging copies of the same files existed side by side. With subtrees everyone clones one repo and sees one version, and changes can still go back upstream with `git subtree push`.

The subtree folders are excluded from Git LFS in `.gitattributes`, because their upstreams do not use LFS and a subtree push would otherwise send LFS pointers.

## Consequences

- Keep commits that touch a subtree folder free of changes elsewhere, so a subtree split produces clean upstream commits.
- The core model and sequence generator subtrees were imported from Hans Wormuth's forks (`hawo2003`, commits pinned by the gym); their upstream is the `WBK-Robotics` repo of the same name.

## Amendment: cell simulation subtree (#148)

`crc_disassembly_sim/` is a third subtree, so the cell-simulation contract tests run against the cell simulation the techstack ships. Its upstream is our KIT GitLab repo `kit/wbk/map/imk/sfb1574_kreislauffabrik/crc-disassembly-sim` (`main`), not the colleague's GitHub `WBK-Robotics/crc_disassembly_sim`. That `main` grew from the thesis snapshot and shares no history with the GitHub `main@19ea74b` first imported here, so #148 removed the old folder and re-added `main@d81ed0f` (#109, #110) with `git subtree add`.

This reverses the LFS exclusion above for this one folder: its upstream uses LFS for its meshes and `.npy` files, so the subtree keeps them in LFS too: the root `.gitattributes` drops the root LFS patterns for the folder, and the subtree's own `crc_disassembly_sim/.gitattributes` decides. A subtree pull needs the upstream's LFS objects locally (`git lfs fetch` in a clone of the upstream, then copy `.git/lfs/objects`); the push to the techstack uploads them to its LFS store.

On Windows the upstream's symlinked robot URDFs check out as stubs that show as modified; never commit them.

## Commands

```sh
# pull upstream changes into a subtree
git subtree pull --prefix=crc_diss_assembly_dataclasses https://github.com/WBK-Robotics/crc_diss_assembly_dataclasses.git main
# push local subtree commits to a branch on the upstream
git subtree push --prefix=crc_diss_assembly_dataclasses https://github.com/WBK-Robotics/crc_diss_assembly_dataclasses.git <branch>
# the cell simulation
git subtree pull --prefix=crc_disassembly_sim https://gitlab.kit.edu/kit/wbk/map/imk/sfb1574_kreislauffabrik/crc-disassembly-sim.git main
```
