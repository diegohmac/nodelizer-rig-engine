"""The Windows overlay: small, scripts and text only, and its runner calls the engine the way
PROTOCOL.md says (every command line it builds parses, with the folders each step needs)."""
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO))
from engine import cli  # noqa: E402


@pytest.fixture(scope="module")
def kit():
    spec = importlib.util.spec_from_file_location("run_engine_kit", REPO / "windows-kit" / "run_engine_kit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overlay_zip(scratch):
    out = scratch / "kit.zip"
    done = subprocess.run([sys.executable, str(REPO / "windows-kit" / "make_kit.py"), "--out", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
    assert "rig-engine/rigengine.py" in names and "rig-engine/build/build_engine.py" in names
    assert "engine-kit/run_engine_kit.py" in names and "engine-kit/README.md" in names
    assert not [n for n in names if Path(n).suffix.lower() in (".exe", ".dll", ".pyd", ".so", ".whl", ".ckpt")]
    assert not [n for n in names if n.startswith("rig-engine/tests/")]
    assert out.stat().st_size < 512 * 1024


def test_expected_values_are_the_earlier_round():
    expected = json.loads((REPO / "windows-kit" / "expected.json").read_text(encoding="utf-8"))
    tira = expected["skintokens"]["example_tira"]
    assert (tira["points"], tira["tokens"], tira["joints"], tira["skin"]) == (
        "f7e9bd28874e", "12317e1749ed", "d5e5f385049c", "57da8e62d712")
    assert expected["unirig"]["mine_rodin"]["bones"] == 45 and expected["skintokens"]["tira_tail"]["bones"] == 58


@pytest.mark.parametrize("model", ["skintokens", "unirig"])
@pytest.mark.parametrize("character", ["mine_rodin", "tira_tail"])
def test_runner_commands_follow_the_protocol(kit, monkeypatch, tmp_path, model, character):
    monkeypatch.setattr(kit, "OUT", tmp_path)
    calls = []

    def fake_step(label, args, reads, writes, logs, sandbox):
        calls.append((label, [str(a) for a in args], reads, writes))
        return {"label": label, "step": args[0], "ok": True, "last": {}}

    monkeypatch.setattr(kit, "run_step", fake_step)
    kit.run_case(model, character, "a", True)
    steps = [c[1][0] for c in calls]
    if model == "skintokens":
        assert steps == ["read", "rig", "write"]
    elif character == "tira_tail":
        assert steps == ["read", "rig", "write"] and "--keep-skeleton" in calls[1][1]
    else:
        assert steps == ["read", "rig", "write", "read", "rig", "write"]
    for label, args, reads, writes in calls:
        parsed = cli.build_parser().parse_args(args)  # raises on anything the engine would refuse
        assert len(reads) + len(writes) <= 8
        assert reads[0] == kit.ROOT, "the package root (python and engine) is the first grant"
        if parsed.step == "rig":
            assert parsed.seed == kit.SEED and parsed.cache
            assert Path(parsed.weights) == kit.WEIGHTS / model and kit.WEIGHTS / model in reads
            assert kit.CHARACTERS[character].parent not in reads, "the rig step never needs the model file"
        if parsed.step in ("read", "write") and parsed.input and Path(parsed.input) == kit.CHARACTERS[character]:
            assert kit.CHARACTERS[character].parent in reads


def test_runner_end_to_end_without_the_sandbox(kit, monkeypatch, engine, inputs, scratch):
    """The runner as on Windows, minus the launcher and the card: the probes and read run for real, rig
    stops with bad_weights (no weights here), and the report, the diffs and the zip are made."""
    import shutil

    root = scratch / "kit-root"
    if root.exists():
        shutil.rmtree(root)
    (root / "engine-kit").mkdir(parents=True)
    shutil.copytree(REPO, root / "rig-engine", ignore=shutil.ignore_patterns("dist", "__pycache__", "tests"))
    for name in ("run_engine_kit.py", "expected.json", "README.md"):
        shutil.copyfile(REPO / "windows-kit" / name, root / "engine-kit" / name)
    (root / "run-restricted.exe").write_bytes(b"")
    (root / "python").mkdir()
    monkeypatch.setattr(kit, "ROOT", root)
    monkeypatch.setattr(kit, "HERE", root / "engine-kit")
    monkeypatch.setattr(kit, "OUT", root / "engine-kit-results")
    monkeypatch.setattr(kit, "PYTHON", Path(sys.executable))
    monkeypatch.setattr(kit, "LAUNCHER", root / "run-restricted.exe")
    monkeypatch.setattr(kit, "ENGINE", engine)
    monkeypatch.setattr(kit, "ENGINE_JSON", engine.parent / "ENGINE.json")
    monkeypatch.setattr(kit, "WEIGHTS", root / "weights")
    for model in kit.MODELS:
        (root / "weights" / model).mkdir(parents=True)
    monkeypatch.setattr(kit, "CHARACTERS", {"mine_rodin": inputs / "mine_rodin.glb"})
    monkeypatch.setattr(sys, "argv", ["run_engine_kit.py", "--snapshot"])
    assert kit.main() == 0
    (root / "rig-engine" / "engine" / "names.py").write_text("# changed on Windows\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_engine_kit.py", "--no-sandbox", "--only", "skintokens:mine_rodin",
                                      "--runs", "a"])
    assert kit.main() == 1  # the rig step can't run here
    report = json.loads((root / "engine-kit-results" / "report.json").read_text(encoding="utf-8"))
    case = report["cases"][0]
    assert [s["label"] for s in case["steps"]] == ["read", "rig"]
    assert case["steps"][0]["ok"] and case["steps"][0]["forbidden_imported"] == []
    assert case["steps"][0]["stdout_lines_not_json"] == []
    assert "bpy" in (root / "engine-kit-results/logs/skintokens-mine_rodin-a/read.imported.txt").read_text()
    assert case["steps"][1]["error"] == "bad_weights"
    assert all(s["first_line_is_starting"] and not s["schema_problems"] for s in case["steps"])
    assert case["steps"][0]["launch_seconds"] is not None
    probes = report["probes"]
    assert probes["light"]["ok"] and probes["full"]["ok"] and probes["full"]["last"]["models"]["unirig"]["weights"] is False
    assert report["guard_problems"] == []
    assert (root / "engine-kit-results" / "report.md").is_file()
    diffs = list((root / "engine-kit-results" / "diffs").iterdir())
    assert [d.name for d in diffs] == ["rig-engine-engine_names.py.diff"]
    with zipfile.ZipFile(root / "engine-kit-results.zip") as archive:
        names = archive.namelist()
    assert "engine-kit-results/report.md" in names and "engine-kit-results/diffs/rig-engine-engine_names.py.diff" in names
    assert not [n for n in names if "pristine" in n or n.endswith(".npz")]


def test_comparisons(kit, monkeypatch, tmp_path):
    """The report's comparisons, on made up runs: fingerprints against the earlier round, two runs
    against each other, UniRig's skeleton against the earlier one."""
    import numpy as np

    monkeypatch.setattr(kit, "OUT", tmp_path)
    expected = json.loads((REPO / "windows-kit" / "expected.json").read_text(encoding="utf-8"))
    tira = expected["skintokens"]["example_tira"]
    prints = {"rig": {k: tira[k] for k in ("points", "tokens", "joints", "skin")}}
    rodin = expected["unirig"]["mine_rodin"]
    work = tmp_path / "work" / "unirig-mine_rodin-a"
    work.mkdir(parents=True)
    np.savez(work / "b.read.npz", names=np.array(rodin["names"]), parents=np.array(rodin["parents"]),
             joints=np.array(rodin["joints"], dtype=np.float32))
    cases = [
        {"model": "skintokens", "character": "example_tira", "run": run, "ok": True, "bones": 22,
         "fingerprints": prints, "rigged": {"sha256": "x"}, "steps": []} for run in ("a", "b")
    ] + [{"model": "unirig", "character": "mine_rodin", "run": "a", "ok": True, "bones": 45,
          "fingerprints": {}, "steps": []}]
    rows = {(r["model"], r["character"]): r for r in kit.compare(cases, expected)}
    st = rows[("skintokens", "example_tira")]
    assert set(st["vs_earlier_windows"].values()) == {"same"}
    assert st["same_seed_same_fingerprints"] and st["same_seed_same_file"]
    ur = rows[("unirig", "mine_rodin")]["vs_earlier_windows"]
    assert ur["names_equal"] and ur["parents_equal"] and ur["largest_joint_move"] < 1e-5


def test_import_list_parser(kit):
    stderr = ("import time: self [us] | cumulative | imported package\n"
              "import time:       120 |        120 |   torch._C\n"
              "import time:        20 |         20 | bpy\n"
              "something else\n")
    assert kit.imported_top_level(stderr) == {"torch", "bpy"}
