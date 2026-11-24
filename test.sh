#!/usr/bin/env bash
# Runs the whole test set in Docker and exits non zero on any failure.
#
#   ./test.sh                 everything
#   ./test.sh -k names        pytest options are passed on
#
# Steps: build the test image (tests/docker/Dockerfile); make sure the pinned code archives are at
# hand (fetched by build/build_engine.py --fetch-only, the one step with network); build a package
# root from them with no network (no weights: they never go in it); run the tests with no network;
# build the Windows overlay zip (dist/rig-engine-kit.zip).
#
# Reference data (the Windows kit's input models and the earlier results) is read from the folders in
# tests/data.env, or from these variables: RIG_INPUT_HOST, RIG_NATIVE_HOST, RIG_FINAL_HOST.
# RIG_DOWNLOADS_HOST may point at a folder that already holds the pinned archives.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ -f tests/data.env ]; then
  # shellcheck disable=SC1091
  source tests/data.env
fi
IMAGE="rig-engine-test:1"
PLATFORM="linux/amd64"   # bpy 4.2.0 has a Linux wheel for x86_64 only
DOWNLOADS="${RIG_DOWNLOADS_HOST:-$HERE/dist/downloads}"
RUN="$HERE/dist/test-run"
for name in RIG_INPUT_HOST RIG_NATIVE_HOST RIG_FINAL_HOST; do
  if [ -z "${!name:-}" ] || [ ! -d "${!name}" ]; then
    echo "test.sh: set $name (or tests/data.env) to the folder of reference data; see tests/conftest.py" >&2
    exit 2
  fi
done

echo "== image"
docker build --quiet --platform "$PLATFORM" -t "$IMAGE" tests/docker >/dev/null

echo "== pinned code archives"
mkdir -p "$DOWNLOADS"
docker run --rm --platform "$PLATFORM" -v "$HERE:/repo:ro" -v "$DOWNLOADS:/downloads" "$IMAGE" \
  python -I /repo/build/build_engine.py --downloads /downloads --fetch-only

echo "== package root (no network; no weights: they live outside the root)"
# A root with spaces and accents in its path, as Windows user folders often have.
ROOT_IN="/run-dir/Zé root/rig engine 0.1"
rm -rf "$RUN"
mkdir -p "$RUN/scratch"
docker run --rm --platform "$PLATFORM" --network none -v "$HERE:/repo:ro" -v "$DOWNLOADS:/downloads:ro" \
  -v "$RUN:/run-dir" "$IMAGE" \
  python -I /repo/build/build_engine.py --downloads /downloads --root "$ROOT_IN"

echo "== tests (no network)"
status=0
docker run --rm --platform "$PLATFORM" --network none \
  -v "$HERE:/repo:ro" -v "$DOWNLOADS:/downloads:ro" -v "$RUN:/run-dir" \
  -v "$RIG_INPUT_HOST:/data/input:ro" -v "$RIG_NATIVE_HOST:/data/native:ro" -v "$RIG_FINAL_HOST:/data/final:ro" \
  -e RIG_ENGINE="$ROOT_IN/engine" -e RIG_DOWNLOADS=/downloads -e RIG_INPUT=/data/input \
  -e RIG_NATIVE=/data/native -e RIG_FINAL=/data/final -e RIG_SCRATCH=/run-dir/scratch \
  -e PYTHONDONTWRITEBYTECODE=1 -w /repo "$IMAGE" \
  python -m pytest -p no:cacheprovider -q tests "$@" || status=$?

echo "== Windows overlay"
docker run --rm --platform "$PLATFORM" --network none -v "$HERE:/repo:ro" -v "$HERE/dist:/dist" "$IMAGE" \
  python -I /repo/windows-kit/make_kit.py --out /dist/rig-engine-kit.zip || status=$?

if [ "$status" -ne 0 ]; then
  echo "test.sh: FAILED (exit $status)" >&2
else
  echo "test.sh: all passed"
fi
exit "$status"
