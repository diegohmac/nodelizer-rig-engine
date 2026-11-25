"""Events on stdout, one JSON object per line. Everything else goes to stderr.

Libraries (and Blender, from C) print to standard output freely. So before anything else runs, the
real stdout is set aside for events only, and file descriptor 1 is pointed at stderr. A print from
any library then lands in the log, and stdout carries nothing but these lines.

Every line is written whole, under a lock, with one write and a flush, so the heartbeat thread can
never split or interleave a line. protocol.schema.json describes every line.
"""
import json
import os
import sys
import threading
import time

from . import ENGINE_VERSION, PROTOCOL_VERSION

# Stable error codes and what they mean. PROTOCOL.md lists the same, with each code's extra fields.
ERROR_CODES = {
    "bad_arguments": "The command line is wrong (unknown option, missing value, a path outside the work folder).",
    "bad_input": "An input file is missing, unreadable, of an unsupported kind, or not what this step expects.",
    "read_failed": "Blender could not read the model file.",
    "no_mesh": "The model file has no mesh to rig.",
    "skeleton_not_kept": "The skeleton could not be kept: the model gave back different bones than the file has.",
    "no_cuda": "No NVIDIA card PyTorch can use: no card, the asked device missing, or a driver too old. On a "
               "Mac: PyTorch cannot use its GPU (MPS).",
    "model_unavailable": "The model doesn't run on this device or computer: UniRig needs an NVIDIA card.",
    "out_of_memory": "The graphics card or the computer ran out of memory.",
    "bad_weights": "A weights file is missing, or could not be loaded as the model's weights.",
    "rig_failed": "The model made no usable rig for this model file.",
    "write_failed": "Blender could not write the rigged file, or the file lacks bones of the rig.",
    "forbidden_module": "A module this step must never load was loaded (Blender in rig, PyTorch in read or write).",
    "internal": "Anything else, a broken engine folder included. A bug; the log has the details.",
}

HEARTBEAT_AFTER = 2.0  # seconds without a line before the heartbeat sends one (PROTOCOL.md promises 5)


class EngineError(Exception):
    def __init__(self, code, message, **details):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def take_stdout():
    """Returns a text stream on the real stdout and sends file descriptor 1 to stderr."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    events_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(events_fd, "w", encoding="utf-8", newline="\n", buffering=1)


class Events:
    def __init__(self, stream, step):
        self.stream = stream
        self.step = step
        self.started = time.monotonic()
        self.stage = "starting"
        self._lock = threading.Lock()
        self._last = self.started
        self._closed = False
        self._heartbeat = None

    def _send(self, kind, fields, final=False):
        line = {"type": kind, "step": self.step, "protocol": PROTOCOL_VERSION}
        line.update(fields)
        text = json.dumps(line, ensure_ascii=True, separators=(",", ":"), default=_plain) + "\n"
        with self._lock:
            if self._closed:
                return
            self.stream.write(text)
            self.stream.flush()
            self._last = time.monotonic()
            if final:
                self._closed = True

    def starting(self):
        """The first line of every step, sent before anything heavy loads."""
        self._send("progress", dict(stage="starting", seconds=self.elapsed(), pid=os.getpid(),
                                    engine_version=ENGINE_VERSION))

    def progress(self, stage, **fields):
        self.stage = stage
        self._send("progress", dict(stage=stage, seconds=self.elapsed(), **fields))

    def result(self, **fields):
        self._send("result", dict(seconds=self.elapsed(), **fields), final=True)

    def error(self, code, message, **fields):
        self._send("error", dict(code=code, message=message, seconds=self.elapsed(), **fields), final=True)

    def elapsed(self):
        return round(time.monotonic() - self.started, 2)

    def start_heartbeat(self):
        """A daemon thread repeating the current stage when nothing was sent for HEARTBEAT_AFTER seconds.
        It runs whenever Python's lock is free; a single library call that holds that lock longer delays it."""
        def beat():
            while not self._closed:
                time.sleep(0.25)
                if time.monotonic() - self._last >= HEARTBEAT_AFTER:
                    self._send("progress", dict(stage=self.stage, seconds=self.elapsed(), heartbeat=True))

        self._heartbeat = threading.Thread(target=beat, name="heartbeat", daemon=True)
        self._heartbeat.start()

    def close(self):
        with self._lock:
            self._closed = True


def _plain(value):
    """JSON for NumPy scalars and paths, without importing NumPy here."""
    for attribute in ("item", "tolist"):
        if hasattr(value, attribute):
            return getattr(value, attribute)()
    return str(value)


def is_out_of_memory(error):
    name = type(error).__name__
    return isinstance(error, MemoryError) or name == "OutOfMemoryError" or "out of memory" in str(error).lower()


def model_failure(error, what):
    """An error from inside a model, as the step's error: out of memory stays itself, the rest is
    rig_failed (the model could not rig this file; another seed may help)."""
    if isinstance(error, EngineError):
        return error
    if type(error).__name__ == "ForbiddenImport":  # engine/guard.py refused a module
        return EngineError("forbidden_module", str(error), modules=[str(error.name or "").partition(".")[0]])
    if is_out_of_memory(error):
        return EngineError("out_of_memory", f"{type(error).__name__}: {error}"[:2000])
    return EngineError("rig_failed", f"{what}: {type(error).__name__}: {error}"[:2000])


def weights_failure(error, what):
    """An error while loading a model's checkpoints: out of memory stays itself, the rest is bad_weights."""
    if isinstance(error, EngineError) or type(error).__name__ == "ForbiddenImport":
        return model_failure(error, what)
    if is_out_of_memory(error):
        return EngineError("out_of_memory", f"{type(error).__name__}: {error}"[:2000])
    return EngineError("bad_weights", f"{what}: {type(error).__name__}: {error}"[:2000])


def log(*parts):
    print("[rig-engine]", *parts, file=sys.stderr, flush=True)
