# The rig engine's entry point. The app runs one step per process, for example:
#     python.exe -I -B -u -X utf8 <engine folder>\rigengine.py read --model skintokens ...
# See PROTOCOL.md. (Not named run.py: the Windows kit's Python treats any script called run.py as
# UniRig's and imports PyTorch at start up.)
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _asked_memory(argv):
    """`--memory`'s value, read before the parser runs (as argparse reads it, abbreviations included)."""
    for at, item in enumerate(argv):
        name, _, value = item.partition("=")
        if len(name) >= 4 and "--memory".startswith(name):
            return value or (argv[at + 1] if at + 1 < len(argv) else None)
    return None


if __name__ == "__main__":
    if sys.platform == "darwin" and sys.argv[1:2] == ["rig"]:
        # macOS's allocator keeps freed large blocks for reuse, and generating on the processor frees a
        # slightly larger cache at every token, so the step's footprint doubles. Below 32 GB the step
        # runs lean, with that cache off; the setting is read only when a process starts, so the step
        # starts again in place (same process id, before it writes anything) when its environment
        # doesn't match its mode. PROTOCOL.md, Devices.
        from engine import host

        wanted = host.environment_for(host.memory_mode(_asked_memory(sys.argv[2:])), os.environ)
        if wanted != dict(os.environ):
            try:
                os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], wanted)
            except OSError:
                pass  # runs on in the mode it started in; the card line and result say which
    from engine.cli import main

    main(sys.argv[1:])
