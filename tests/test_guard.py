"""The runtime guard: each step refuses its forbidden modules, loudly."""
import json
import subprocess
import sys

from conftest import REPO, run_python


def test_read_refuses_pytorch():
    result = run_python(
        "import sys, json; sys.path.insert(0, sys.argv[1])\n"
        "from engine import guard\n"
        "guard.install('read')\n"
        "try:\n"
        "    import torch\n"
        "    print(json.dumps({'loaded': True}))\n"
        "except guard.ForbiddenImport as error:\n"
        "    print(json.dumps({'loaded': False, 'message': str(error), 'in_modules': 'torch' in sys.modules}))\n",
        REPO)
    assert result["loaded"] is False and result["in_modules"] is False
    assert "read step may not load torch" in result["message"]


def test_rig_refuses_blender():
    result = run_python(
        "import sys, json; sys.path.insert(0, sys.argv[1])\n"
        "from engine import guard\n"
        "guard.install('rig')\n"
        "try:\n"
        "    import bpy\n"
        "    print(json.dumps({'loaded': True}))\n"
        "except ImportError as error:\n"  # a library's optional import sees it as missing
        "    print(json.dumps({'loaded': False, 'message': str(error)}))\n",
        REPO)
    assert result["loaded"] is False and "rig step may not load bpy" in result["message"]


def test_a_step_fails_when_the_forbidden_module_is_already_loaded():
    """The engine's own entry, with PyTorch loaded before the read step starts: one error line,
    forbidden_module, exit 1."""
    done = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); import torch\n"
         "from engine.cli import main\n"
         "main(['read', '--model', 'skintokens', '--input', '/nope.glb', '--output', '/tmp/x.npz', '--work', '/tmp'])\n",
         str(REPO)], capture_output=True, text=True)
    assert done.returncode == 1, done.stderr
    lines = [json.loads(line) for line in done.stdout.splitlines()]
    assert [line["type"] for line in lines] == ["progress", "error"] and lines[0]["stage"] == "starting"
    assert lines[1]["code"] == "forbidden_module" and lines[1]["modules"] == ["torch"]


def test_guard_lists_are_disjoint():
    sys.path.insert(0, str(REPO))
    from engine import guard

    assert not set(guard.BLENDER) & set(guard.TORCH)
    for step in ("read", "write"):
        assert set(guard.TORCH) <= set(guard.BLOCKED[step])
    for step in ("rig", "probe"):
        assert set(guard.BLENDER) <= set(guard.BLOCKED[step])
