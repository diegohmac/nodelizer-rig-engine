"""The engine's command line: one step per process. See PROTOCOL.md.

    rigengine.py probe [--light] [--work W] [--weights MODEL=FOLDER ...] [--verify-weights]
    rigengine.py read  --model M --input FILE --output W/x.read.npz --work W
    rigengine.py rig   --model M --input W/x.read.npz --output W/x.rig.npz --work W --weights F --seed N
                       [--keep-skeleton] [--standard-names on|off] [--stage skeleton|skin] [--cache C]
                       [--device N]
    rigengine.py write --rig W/x.rig.npz [--input FILE] --output W/x.glb --work W

Progress, the result and errors go to stdout as JSON lines; logs go to stderr. Exit code 0 means a
`result` line was written, 1 an `error` line, 2 an `error` line about the command line itself.
The first line of every step is a `starting` progress line, sent before anything heavy loads.
"""
import argparse
import os
import sys
import traceback

from . import guard, workspace
from .events import EngineError, Events, is_out_of_memory, log, take_stdout

STEPS = ("probe", "read", "rig", "write")
MODELS = ("skintokens", "unirig")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise EngineError("bad_arguments", message)

    def exit(self, status=0, message=None):  # --help: print to the log, never to stdout
        if message:
            print(message, file=sys.stderr)
        raise EngineError("bad_arguments", "help was asked for; nothing was done")


def _on_off(value):
    if value not in ("on", "off"):
        raise argparse.ArgumentTypeError("on or off")
    return value == "on"


def build_parser():
    parser = _Parser(prog="rigengine.py", description="Rig engine steps. See PROTOCOL.md.")
    sub = parser.add_subparsers(dest="step", required=True, parser_class=_Parser)

    probe = sub.add_parser("probe", help="The cards, their free video memory, and the models' files.")
    probe.add_argument("--light", action="store_true", help="the cards and their free memory only")
    probe.add_argument("--work", help="a folder for anything written while probing")
    probe.add_argument("--weights", action="append", default=[], metavar="MODEL=FOLDER",
                       help="a model's weights folder, to check it is complete")
    probe.add_argument("--verify-weights", action="store_true", help="also check every weight file's SHA-256")

    read = sub.add_parser("read", help="Blender reads a model file into plain arrays.")
    read.add_argument("--model", required=True, choices=MODELS)
    read.add_argument("--input", required=True, help="the model file (.glb, .fbx, .obj)")
    read.add_argument("--output", required=True, help="the read file to make, in the work folder, .npz")
    read.add_argument("--work", required=True)

    rig = sub.add_parser("rig", help="The model makes the rig from a read file.")
    rig.add_argument("--model", required=True, choices=MODELS)
    rig.add_argument("--input", required=True, help="the read file")
    rig.add_argument("--output", required=True, help="the rig file to make, in the work folder, .npz")
    rig.add_argument("--work", required=True)
    rig.add_argument("--seed", required=True, type=int)
    rig.add_argument("--keep-skeleton", action="store_true", help="skin the file's own skeleton")
    rig.add_argument("--standard-names", type=_on_off, default=True, metavar="on|off")
    rig.add_argument("--stage", choices=("skeleton", "skin"), help="UniRig only: which of its two passes")
    rig.add_argument("--weights", required=True, help="this model's weights folder")
    rig.add_argument("--cache", help="a folder kept between runs for CUDA's compiled kernels")
    rig.add_argument("--device", type=int, help="the CUDA device, numbered as nvidia-smi numbers them")

    write = sub.add_parser("write", help="Blender writes the rig onto the original model file.")
    write.add_argument("--rig", required=True, help="the rig file")
    write.add_argument("--input", help="the original model file (not for UniRig's skeleton pass)")
    write.add_argument("--output", required=True, help="the file to make, in the work folder")
    write.add_argument("--work", required=True)
    return parser


def run(step, args, events):
    if step == "probe":
        from .steps import probe

        return probe.run(args, events)
    if step == "read":
        from .steps import read

        return read.run(args, events)
    if step == "rig":
        from .steps import rig

        return rig.run(args, events)
    from .steps import write

    return write.run(args, events)


def _remove_blender_temp():
    """Blender removes its session temp folder at a normal exit, which os._exit skips."""
    bpy = sys.modules.get("bpy")
    if bpy is None:
        return
    try:
        import shutil

        folder = bpy.app.tempdir
        if folder and os.path.basename(os.path.normpath(folder)).startswith("blender_"):
            shutil.rmtree(folder, ignore_errors=True)
    except Exception:
        pass


def main(argv):
    sys.dont_write_bytecode = True
    step = argv[0] if argv and argv[0] in STEPS else "unknown"
    events = Events(take_stdout(), step)
    events.starting()  # first, before any library loads: the app can tell "could not start" from a crash
    events.start_heartbeat()
    code = 1
    try:
        try:
            if step == "unknown":
                raise EngineError("bad_arguments", f"the first argument must be one of {', '.join(STEPS)}")
            guard.install(step)
            args = build_parser().parse_args(argv)
            result = run(step, args, events)
            late = guard.loaded_forbidden(step)
            if late:
                raise EngineError("forbidden_module", f"the {step} step loaded {', '.join(late)}", modules=late)
            result["forbidden_modules_loaded"] = []
            events.result(**result)
            code = 0
        except guard.ForbiddenImport as error:
            raise EngineError("forbidden_module", str(error), modules=guard.loaded_forbidden(step))
    except EngineError as error:
        if error.code not in ("bad_arguments", "bad_input"):
            traceback.print_exc()
        events.error(error.code, error.message, **error.details)
        code = 2 if error.code == "bad_arguments" else 1
    except BaseException as error:  # noqa: BLE001 - every failure must end in one error line
        traceback.print_exc()
        if is_out_of_memory(error):
            events.error("out_of_memory", f"{type(error).__name__}: {error}"[:2000])
        else:
            events.error("internal", f"{type(error).__name__}: {error}"[:2000])
        code = 1
    finally:
        events.close()
        workspace.release_cache()
        _remove_blender_temp()
        log(f"{step} finished with exit code {code}")
        for stream in (events.stream, sys.stderr, sys.__stderr__):
            try:
                stream.flush()
            except Exception:
                pass
        # Leave without Python's shutdown: Blender's and CUDA's teardown add nothing once everything is
        # written and flushed, and Blender with other large libraries has crashed there (work-1).
        os._exit(code)
