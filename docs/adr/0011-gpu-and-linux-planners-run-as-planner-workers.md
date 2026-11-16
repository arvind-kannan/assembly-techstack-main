# GPU and Linux planners run as planner workers outside the cell simulation

Light motion planners (`straight_line_checked`, any planner that needs only pybullet) run inside the cell simulation. Planners that need an NVIDIA GPU or Linux (cuMotion, cuRobo, OMPL) each run as a **planner worker**: a separate process with its own environment, which the cell simulation's planner calls per operation over the sim server's length-prefixed framing. It sends a **scene description** (obstacles, held parts from the cell-assembly snapshot, each robot's start joint configuration) and gets back the NC commands of [ADR-0010](0010-cell-simulation-planner-plans-one-operation-and-returns-nc-commands.md). We chose this because students run the stack on low-power laptops without a GPU, so the cell simulation itself must stay light. The planners cannot share one interpreter anyway: the cuMotion wheel is cp314 only, and OMPL's wheels stop at cp313.

Planner workers run in WSL on the GPU workstation, which also runs the whole stack for end-to-end runs and is the only machine with an OPC UA path to the SINUMERIK. Each worker has its own native venv locked with `uv`. The one-time sim config ([ADR-0008](0008-cell-simulation-sits-before-the-real-cell.md)) chooses the planner and, for a planner worker, its address. Dev configs try planners out. After the comparison (#190), the winning planner becomes the workstation's default config.

Decided in #189.

## Considered options

- **All planners in-process** (with optional extras imported lazily): no protocol and no scene translation, but one interpreter for every planner (OMPL built from source for 3.14), and a planner crash would take down the sim server, which on the workstation is connected to the SINUMERIK.
- **Docker image per worker**: reproducible and portable to a cluster, but heavier to set up and debug in WSL, and with a single GPU machine there is little to reproduce. The cuMotion licence would forbid pushing such an image anyway.
- **KIT cluster for the comparison runs**: more throughput, but it needs cluster tooling (Apptainer, job scripts) and adds queue latency. We revisit it only if the comparison outgrows the workstation.

## Consequences

- Students run Windows-native with `straight_line_checked` by default. Running the OMPL planner worker on CPU in WSL is documented as an option for students who work on motion planning.
- The dev machine and CI have no GPU: planner-worker tests carry the `gpu` marker and run on the workstation only, and the seam is tested with fake planners.
- The cuMotion wheel is never committed or redistributed. Each user downloads it after accepting NVIDIA's licence.
