# cuMotion wheels (not tracked)

NVIDIA cuMotion is distributed by NVIDIA under the NVIDIA Isaac ROS Software License and is
not on PyPI. The wheels are therefore **not** stored in this repository; `*.whl` in this folder
is git-ignored so a local copy can sit here for the Docker build without being committed.

1. Download `cumotion-1.1.0-cuda12.6` from https://developer.nvidia.com/isaac/download and
   accept the licence.
2. Copy the wheel for your Python version from its `python_wheels/` folder into this directory
   (the Dockerfile and the native setup use `cumotion-1.1.0-cp314-cp314-linux_x86_64.whl`).
3. Native install: `pip install cuMotion_wheels/cumotion-1.1.0-cp314-cp314-linux_x86_64.whl`
   (`cumotion_vis` is optional and needs `open3d`, which has no cp314 wheel).
4. Docker: `docker build` copies the cp314 wheel from this folder, so step 2 must be done first.

Requirements: NVIDIA GPU (Turing or newer), CUDA 12.6 driver, Python 3.10-3.14.
