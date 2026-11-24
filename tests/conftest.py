"""Shared fixtures. The tests run in the Docker image tests/docker/Dockerfile, started by ./test.sh,
which builds an engine folder (no weights) and mounts the reference data read only:

    RIG_ENGINE       the built engine folder
    RIG_DOWNLOADS    the pinned code archives (build/sources.py)
    RIG_INPUT        the Windows kit's input\\ folder (characters and the hand fixed skeleton)
    RIG_NATIVE       the Windows round's results (rig-native-results)
    RIG_FINAL        the last Docker round's results (rig-final-results)
    RIG_SCRATCH      a writable folder

The pytest process itself never imports Blender or PyTorch: anything that needs one runs in a child
process, so the tests keep the engine's own rule.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))


def _env_path(name):
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is not set; run the tests with ./test.sh")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{name}={value} does not exist")
    return path


@pytest.fixture(scope="session")
def engine():
    return _env_path("RIG_ENGINE")


@pytest.fixture(scope="session")
def downloads():
    return _env_path("RIG_DOWNLOADS")


@pytest.fixture(scope="session")
def inputs():
    return _env_path("RIG_INPUT")


@pytest.fixture(scope="session")
def native():
    return _env_path("RIG_NATIVE")


@pytest.fixture(scope="session")
def final():
    return _env_path("RIG_FINAL")


@pytest.fixture(scope="session")
def scratch():
    return _env_path("RIG_SCRATCH")


@pytest.fixture
def work(scratch, request):
    folder = scratch / ("work-" + request.node.name.replace("[", "-").replace("]", "").replace("/", "-"))
    if folder.exists():
        import shutil

        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    return folder


class StepRun:
    def __init__(self, code, stdout, stderr):
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.events = [json.loads(line) for line in stdout.splitlines() if line.strip()]

    @property
    def last(self):
        return self.events[-1] if self.events else None

    @property
    def result(self):
        assert self.code == 0, f"exit {self.code}: {self.last}\n{self.stderr[-4000:]}"
        assert self.last["type"] == "result", self.last
        return self.last

    def __repr__(self):
        return f"StepRun(code={self.code}, last={self.last})"


_VALIDATOR = None


def schema_validator():
    global _VALIDATOR
    if _VALIDATOR is None:
        from jsonschema import Draft202012Validator

        schema = json.loads((REPO / "protocol.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        _VALIDATOR = Draft202012Validator(schema)
    return _VALIDATOR


def check_lines(stdout):
    """stdout carries JSON lines and nothing else; each matches protocol.schema.json; the first is
    `starting`; exactly one `result` or `error`, last."""
    events = []
    for line in stdout.splitlines():
        assert line.strip(), "an empty line on stdout"
        event = json.loads(line)
        errors = sorted(schema_validator().iter_errors(event), key=lambda e: e.path)
        assert not errors, f"line does not match the schema: {line}\n{errors[0].message}"
        events.append(event)
    if events:
        assert events[0]["stage"] == "starting" and "pid" in events[0], events[0]
        finals = [e for e in events if e["type"] in ("result", "error")]
        assert len(finals) == 1 and events[-1] is finals[0], events[-3:]
    return events


def run_step(engine, *args, timeout=1800, env=None):
    """Runs one engine step the way the app does: its own process, isolated Python."""
    command = [sys.executable, "-I", "-B", "-u", "-X", "utf8", str(engine / "rigengine.py"), *map(str, args)]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env)
    check_lines(done.stdout)
    return StepRun(done.returncode, done.stdout, done.stderr)


def run_python(code, *args, timeout=1800, cwd=None):
    """A child Python for anything that loads Blender or PyTorch. Returns its last stdout line as JSON."""
    done = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout, cwd=cwd)
    assert done.returncode == 0, f"child failed ({done.returncode}):\n{done.stdout[-3000:]}\n{done.stderr[-6000:]}"
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1]) if lines else None
