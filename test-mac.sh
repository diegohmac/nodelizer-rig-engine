#!/usr/bin/env bash
# The native Mac test: runs the Mac engine on this Mac, every step inside sandbox-exec with
# mac/rig-engine.sb, as the app will. Not part of ./test.sh (Docker), and it needs what only a Mac
# with Apple silicon has:
#   - a built Mac root (build/mac/build_mac.py), in RIG_MAC_ROOT
#   - SkinTokens' weights folder (build/build_engine.py --weights-out, then <folder>/skintokens), in
#     RIG_MAC_WEIGHTS
#   - a model file to rig, in RIG_MAC_INPUT (default: example_tira.glb in RIG_INPUT_HOST)
# The variables may also be set in tests/data.env. It checks the root, probe, the sandbox's refusals
# (network, files outside its folders, other programs, the GPU), UniRig's refusal, then read, rig
# (on the Mac default, in the memory mode this Mac gets) and write on the one model file: three minutes or so.
#
#   ./test-mac.sh            once
#   ./test-mac.sh --twice    rig again in the other memory mode and compare the fingerprints
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ -f tests/data.env ]; then
  # shellcheck disable=SC1091
  source tests/data.env
fi
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "test-mac.sh: needs a Mac with Apple silicon" >&2
  exit 2
fi
ROOT="${RIG_MAC_ROOT:-}"
WEIGHTS="${RIG_MAC_WEIGHTS:-}"
INPUT="${RIG_MAC_INPUT:-${RIG_INPUT_HOST:-}/example_tira.glb}"
if [ -z "$ROOT" ] || [ ! -x "$ROOT/python/bin/python3.11" ]; then
  echo "test-mac.sh: set RIG_MAC_ROOT to a built Mac root (build/mac/build_mac.py)" >&2
  exit 2
fi
if [ -z "$WEIGHTS" ] || [ ! -d "$WEIGHTS/experiments" ]; then
  echo "test-mac.sh: set RIG_MAC_WEIGHTS to SkinTokens' weights folder" >&2
  exit 2
fi
if [ ! -f "$INPUT" ]; then
  echo "test-mac.sh: no model file at $INPUT (set RIG_MAC_INPUT)" >&2
  exit 2
fi

# The checks run with the root's own Python, outside the sandbox; each engine step runs inside it.
exec "$ROOT/python/bin/python3.11" -I -B "$HERE/tests/mac/native.py" --root "$ROOT" --weights "$WEIGHTS" \
  --input "$INPUT" --scratch "$HERE/dist/test-mac" "$@"
