# CRC Disassembly Simulation — multi-robot motion planning
#
# Requirements:
#   - NVIDIA GPU with CUDA 12.6 driver (driver >= 525.85)
#   - Docker with NVIDIA Container Toolkit installed
#     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#
# Build:
#   docker build -t crc_disassembly_sim .
#
# Run (GUI via X11 forwarding on Linux host):
#   docker compose up
# OR without compose:
#   docker run --gpus all --rm -it \
#     -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#     crc_disassembly_sim python3 training/tests/test_cumotion_gui.py

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# ── system dependencies ────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive

# Switch to the KIT/German Ubuntu mirror — archive.ubuntu.com resolves to
# Cloudflare 104.20.28.246 which times out from inside Docker on this network.
# de.archive.ubuntu.com uses a different CDN route and is reliably reachable.
RUN sed -i \
      -e 's|http://archive.ubuntu.com|http://de.archive.ubuntu.com|g' \
      -e 's|http://security.ubuntu.com|http://de.archive.ubuntu.com|g' \
    /etc/apt/sources.list

# Retry up to 5 times, 60s timeout per connection.
RUN printf 'Acquire::Retries "5";\nAcquire::http::Timeout "60";\nAcquire::https::Timeout "60";\n' \
    > /etc/apt/apt.conf.d/80-retries

# Install Python 3.14 via the deadsnakes PPA (Ubuntu 22.04 ships Python 3.10).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.14 \
        python3.14-dev \
        python3.14-venv \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.14 (deadsnakes doesn't ship pip).
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.14

# Make python3.14 the default python3.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.14 1

# ── PyTorch (latest stable — cp314 wheels available via cu126 index) ──────────
# Tries CUDA 12.6 build first; falls back to CPU-only if no GPU wheel exists yet.
RUN python3 -m pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cu126 \
    || python3 -m pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu

# ── cuMotion wheel (Python 3.14, CUDA 12.6) ───────────────────────────────────
# cp314 wheel is expected in cuMotion_wheels/ (not tracked; see its README).
# cumotion_vis is intentionally omitted: open3d has no cp314 wheel on PyPI,
# and PyBullet handles all visualisation in this sim — cumotion_vis is not needed.
COPY cuMotion_wheels/cumotion-1.1.0-cp314-cp314-linux_x86_64.whl /tmp/
RUN python3 -m pip install --no-cache-dir /tmp/cumotion-1.1.0-cp314-cp314-linux_x86_64.whl \
 && rm /tmp/*.whl

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# ── project ───────────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace

ENV PYTHONPATH=/workspace

CMD ["python3", "training/tests/test_cumotion_sim.py"]
