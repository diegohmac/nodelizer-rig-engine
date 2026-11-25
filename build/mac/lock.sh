#!/usr/bin/env bash
# Resolves build/mac/requirements-mac.in into requirements-mac.txt: every package for macOS on Apple
# silicon and Python 3.11, wheels only, each with its SHA-256, the versions held to the Windows pins.
# Runs uv in Docker (as the Windows lock was made); the network is used to read PyPI.
#   build/mac/lock.sh <requirements-win.txt>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WIN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
docker run --rm -v "$HERE:/mac" -v "$WIN:/win.txt:ro" python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1 sh -c '
  pip install --quiet uv==0.9.5 &&
  grep -E "^[a-zA-Z0-9_.-]+==" /win.txt | sed "s/ .*//" | sed "s/+cu128//" |
    grep -v -E "^(torch|torchvision|torch-scatter|torch_scatter|torch-cluster|torch_cluster|spconv|cumm)" > /tmp/constraints.txt &&
  cd /mac && uv pip compile requirements-mac.in -c /tmp/constraints.txt \
    --python-platform aarch64-apple-darwin --python-version 3.11 --only-binary :all: \
    --generate-hashes --index-url https://pypi.org/simple --emit-index-url --no-header \
    -o requirements-mac.txt'
