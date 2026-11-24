# The test environment: Linux x86_64 (emulated on an ARM Mac), Python 3.11, Blender as a Python module
# (bpy 4.2.0, the Windows kit's version), PyTorch 2.7.1 for the CPU, and the small packages the
# models' read and write code imports, at the Windows kit's versions (requirements-win.txt).
# Built with network; the tests themselves run with --network none.
FROM --platform=linux/amd64 python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
# The libraries bpy links (the earlier Docker kits' list), and zip for the Windows overlay.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libegl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 libxxf86vm1 libxfixes3 libxi6 \
      libxkbcommon0 libsm6 libice6 libusb-1.0-0 libudev1 zip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install bpy==4.2.0 numpy==1.26.4 scipy==1.17.1 trimesh==5.2.2 fast-simplification==0.2.0 \
      python-box==8.0.1 pyyaml==6.0.3 tqdm==4.67.1 omegaconf==2.3.0 addict==2.4.0 pytest==8.3.5 \
      jsonschema==4.23.0
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
RUN useradd --create-home --uid 1000 tester
USER tester
WORKDIR /repo
