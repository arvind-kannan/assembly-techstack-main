# AT-234 evidence: gym outcomes and graphs unchanged

`dump_graphs_and_rollouts.py` dumps the extracted sequence graphs (full and collapsed) and 3 seeds x 40 random env steps with their full `info`. It covers EMO SR V2 and bolt-and-nut.

The bolt-and-nut case has no `product_setup.json`, so the setup is the builder's own: a gripper-only cell, base `part_j_elbow_pipe`, target `part_l_cover_lid`.

Run it from the repo root:

    MPLCONFIGDIR=$TMPDIR/mpl PYTHONPATH=src .venv/bin/python .claude/log/AT-234/evidence/dump_graphs_and_rollouts.py out.json

The dump on main 1200748 (before) and on c7f4c4f (after) had the same sha256, `a98dca288dcd6400d179969917643ef96686fa2f66332ed88b5b49055c6783c1`.
