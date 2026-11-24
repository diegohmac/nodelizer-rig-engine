"""The protocol: JSON lines on stdout, stable error codes and exit codes, refused paths, the .npz
layouts, and PROTOCOL.md kept in step with the code."""
import json
import re
import sys

import numpy as np
import pytest

import rigfiles
from conftest import REPO, run_python, run_step

sys.path.insert(0, str(REPO))
from engine import events, layout, plainnpz, workspace  # noqa: E402
from engine.unirig import capture  # noqa: E402


def _error(run, code, exit_code=1):
    assert run.code == exit_code, run
    assert len([e for e in run.events if e["type"] in ("result", "error")]) == 1
    assert run.last["type"] == "error" and run.last["code"] == code, run.last
    assert run.last["code"] in events.ERROR_CODES
    return run.last


def _read_file(path, model="skintokens", **extra):
    arrays = {
        "format": plainnpz.text(layout.READ_FORMAT), "model": plainnpz.text(model),
        "engine_version": plainnpz.text("test"), "source_name": plainnpz.text("x.glb"),
        "has_skeleton": np.bool_(False), "bone_count": np.int64(0),
        "vertices": np.random.rand(10, 3), "faces": np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32),
    }
    if model == "unirig":
        arrays.update(vertex_normals=np.zeros((10, 3), np.float32), face_normals=np.zeros((2, 3), np.float32),
                      armature_bone_count=np.int64(0))
    arrays.update(extra)
    plainnpz.save(path, arrays)
    return path


# --- command line -------------------------------------------------------------------------------

def test_no_step(engine):
    _error(run_step(engine), "bad_arguments", 2)
    _error(run_step(engine, "dance"), "bad_arguments", 2)


def test_help_prints_nothing_on_stdout_but_the_error(engine):
    run = run_step(engine, "read", "--help")
    _error(run, "bad_arguments", 2)
    assert "--model" in run.stderr


def test_missing_option(engine, work):
    _error(run_step(engine, "read", "--model", "skintokens", "--work", work), "bad_arguments", 2)


def test_output_outside_the_work_folder(engine, inputs, work, scratch):
    run = run_step(engine, "read", "--model", "skintokens", "--input", inputs / "mine_rodin.glb",
                   "--output", scratch / "elsewhere.npz", "--work", work)
    _error(run, "bad_arguments", 2)
    run = run_step(engine, "read", "--model", "skintokens", "--input", inputs / "mine_rodin.glb",
                   "--output", work / ".." / "escape.npz", "--work", work)
    _error(run, "bad_arguments", 2)


def test_outputs_are_never_overwritten(engine, inputs, work):
    (work / "taken.npz").write_bytes(b"")
    run = run_step(engine, "read", "--model", "skintokens", "--input", inputs / "mine_rodin.glb",
                   "--output", work / "taken.npz", "--work", work)
    _error(run, "bad_arguments", 2)


def test_relative_and_missing_inputs(engine, work):
    _error(run_step(engine, "read", "--model", "unirig", "--input", "model.glb", "--output", work / "a.npz",
                    "--work", work), "bad_arguments", 2)
    _error(run_step(engine, "read", "--model", "unirig", "--input", work / "missing.glb", "--output",
                    work / "a.npz", "--work", work), "bad_input")
    (work / "notes.txt").write_text("x", encoding="utf-8")
    _error(run_step(engine, "read", "--model", "unirig", "--input", work / "notes.txt", "--output",
                    work / "a.npz", "--work", work), "bad_input")


def _fake_weights(folder, model="skintokens"):
    """Files at the right paths with the wrong bytes: enough to pass the presence check."""
    from engine.weights import WEIGHTS

    for rel in WEIGHTS[model]:
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(b"not the weights")
    return folder


def test_rig_option_checks(engine, work):
    read = _read_file(work / "x.read.npz")
    base = ["rig", "--input", read, "--output", work / "x.rig.npz", "--work", work, "--weights", work / "w"]
    _error(run_step(engine, *base, "--model", "skintokens", "--seed", "-1"), "bad_arguments", 2)
    _error(run_step(engine, *base, "--model", "skintokens", "--seed", str(2**32)), "bad_arguments", 2)
    _error(run_step(engine, *base, "--model", "skintokens", "--seed", "1", "--stage", "skin"), "bad_arguments", 2)
    _error(run_step(engine, *base, "--model", "unirig", "--seed", "1"), "bad_arguments", 2)
    _error(run_step(engine, *base, "--model", "unirig", "--seed", "1", "--stage", "skeleton", "--keep-skeleton"),
           "bad_arguments", 2)
    _error(run_step(engine, *base, "--model", "skintokens", "--seed", "1", "--standard-names", "maybe"),
           "bad_arguments", 2)
    _error(run_step(engine, *base, "--model", "skintokens", "--seed", "1", "--device", "-1"), "bad_arguments", 2)
    no_weights = [a for a in base if a not in ("--weights", work / "w")]
    _error(run_step(engine, *no_weights, "--model", "skintokens", "--seed", "1"), "bad_arguments", 2)


def test_rig_refuses_a_read_file_for_the_other_model(engine, work):
    read = _read_file(work / "x.read.npz", model="unirig")
    error = _error(run_step(engine, "rig", "--model", "skintokens", "--input", read, "--output",
                            work / "x.rig.npz", "--work", work, "--weights", work, "--seed", "1"), "bad_input")
    assert "read the file again" in error["message"]


@pytest.mark.parametrize("model", ["skintokens", "unirig"])
def test_rig_names_the_missing_weights(engine, work, model):
    """Weights live outside the engine folder; a folder without them fails before PyTorch loads."""
    from engine.weights import WEIGHTS

    read = _read_file(work / "x.read.npz", model=model)
    args = ["rig", "--model", model, "--input", read, "--output", work / "x.rig.npz", "--work", work,
            "--weights", work / "empty", "--seed", "1"] + (["--stage", "skin"] if model == "unirig" else [])
    (work / "empty").mkdir()
    run = run_step(engine, *args)
    error = _error(run, "bad_weights")
    assert sorted(error["missing"]) == sorted(WEIGHTS[model])
    assert "loading_libraries" not in [e.get("stage") for e in run.events]
    assert not (work / "x.rig.npz").exists()


def test_rig_without_a_card(engine, work):
    """Weights in place (wrong bytes, which only loading would notice): the card check comes first."""
    read = _read_file(work / "x.read.npz")
    weights = _fake_weights(work / "weights")
    run = run_step(engine, "rig", "--model", "skintokens", "--input", read, "--output", work / "x.rig.npz",
                   "--work", work, "--weights", weights, "--seed", "1", "--device", "3")
    _error(run, "no_cuda")
    assert [e["stage"] for e in run.events if e["type"] == "progress"][:2] == ["starting", "loading_libraries"]


def test_cache_folder_after_a_kill(engine, work):
    """A marker left by a process that is gone means a step was killed: its CUDA cache is emptied and the
    marker removed. A live marker (this test's own process) leaves the cache alone."""
    read = _read_file(work / "x.read.npz")
    weights = _fake_weights(work / "weights")
    cache = work / "cache-kept"
    (cache / "cuda").mkdir(parents=True)
    (cache / "cuda" / "kernel.bin").write_bytes(b"x")
    (cache / ".in-use-999999").write_text("")  # no such process
    args = ["rig", "--model", "skintokens", "--input", read, "--output", work / "x.rig.npz", "--work", work,
            "--weights", weights, "--seed", "1", "--cache", cache]
    _error(run_step(engine, *args), "no_cuda")
    assert not (cache / "cuda" / "kernel.bin").exists() and (cache / "cuda").is_dir()
    assert sorted(p.name for p in cache.iterdir()) == ["cuda"], "markers left behind"
    (cache / "cuda" / "kernel.bin").write_bytes(b"x")
    import os

    (cache / f".in-use-{os.getpid()}").write_text("")
    (cache / ".in-use-999999").write_text("")
    _error(run_step(engine, *args), "no_cuda")
    assert (cache / "cuda" / "kernel.bin").exists(), "a live process's cache was emptied"


def test_write_checks_the_rig_file(engine, inputs, work):
    rig = rigfiles.skintokens_rig_from_read({
        "matrix_local": np.tile(np.eye(4), (2, 1, 1)), "parents": np.array([-1, 0]),
        "joint_names": plainnpz.texts(["a", "b"]), "vertices": np.zeros((3, 3)), "skin": np.ones((3, 2)) / 2})
    rig["skin"] = rig["skin"][:2]  # rows that don't match the vertices
    path = work / "bad.rig.npz"
    plainnpz.save(path, rig)
    _error(run_step(engine, "write", "--rig", path, "--input", inputs / "mine_rodin.glb", "--output",
                    work / "out.glb", "--work", work), "bad_input")


def test_reserved_names_are_refused_as_outputs(engine, inputs, work):
    for name in (".hidden.npz", "tmp", "cache"):
        run = run_step(engine, "read", "--model", "skintokens", "--input", inputs / "mine_rodin.glb",
                       "--output", work / name, "--work", work)
        _error(run, "bad_arguments", 2)


def test_probe_without_a_card(engine, work):
    weights = _fake_weights(work / "weights")
    run = run_step(engine, "probe", "--work", work, "--weights", f"skintokens={weights}",
                   "--weights", f"unirig={work / 'none'}", "--verify-weights")
    result = run.result
    assert result["cuda_available"] is False and result["devices"] == [] and result["default_device"] is None
    assert result["package"]["protocol"] == 1 and result["package"]["python"] == "python/tools/python.exe"
    st, ur = result["models"]["skintokens"], result["models"]["unirig"]
    assert st["code"] and st["weights"] is False and st["needs_free_bytes"] == int(5.5 * 2**30)
    assert all(f["present"] and not f["ok"] for f in st["verified"].values())
    assert ur["weights"] is False and len(ur["missing_weights"]) == 2 and ur["needs_free_bytes"] == int(7.5 * 2**30)


def test_probe_light(engine):
    run = run_step(engine, "probe", "--light")
    result = run.result
    assert result["light"] and "models" not in result and result["devices"] == []


def test_device_choice():
    from engine.steps import torchenv

    cards = [{"index": 0, "name": "a", "total_bytes": 8}, {"index": 1, "name": "b", "total_bytes": 24},
             {"index": 2, "name": "c", "total_bytes": 24}]
    assert torchenv.choose(cards, None) == 1  # most memory, lowest number on a tie
    assert torchenv.choose(cards, 2) == 2
    with pytest.raises(events.EngineError) as caught:
        torchenv.choose(cards, 5)
    assert caught.value.code == "no_cuda"
    with pytest.raises(events.EngineError):
        torchenv.choose([], None)


def test_a_step_needs_no_environment(engine, inputs, work):
    """The app's launcher passes only Windows' own five variables. With nothing at all here, the engine
    sets its own folders, every one inside the work folder."""
    run = run_step(engine, "read", "--model", "unirig", "--input", inputs / "mine_rodin.glb", "--output",
                   work / "a.npz", "--work", work, env={})
    assert run.result["faces"] > 0
    assert (work / "cache" / "home").is_dir() and (work / "tmp").is_dir()


# --- .npz files and layouts ---------------------------------------------------------------------

def test_plain_files_round_trip(tmp_path):
    arrays = {"a": np.arange(5, dtype=np.int32), "b": plainnpz.texts(["Zé", "mixamorig:Hips"]),
              "c": np.bool_(True), "d": plainnpz.text("text"), "e": np.random.rand(3, 4, 4).astype(np.float32)}
    plainnpz.save(tmp_path / "x.npz", arrays)
    back = plainnpz.load(tmp_path / "x.npz")
    for key, value in arrays.items():
        assert back[key].dtype == np.asarray(value).dtype
        np.testing.assert_array_equal(back[key], value)
    assert plainnpz.as_texts(back["b"]) == ["Zé", "mixamorig:Hips"]
    with pytest.raises(TypeError):
        plainnpz.save(tmp_path / "y.npz", {"x": np.array([None, 1], dtype=object)})
    assert not (tmp_path / "y.npz").exists()


def test_read_layout_checks(tmp_path):
    good = plainnpz.load(_read_file(tmp_path / "a.npz"))
    assert layout.check_read(good) == "skintokens"
    bad = dict(good, faces=np.array([[0, 1, 99]], dtype=np.int32))
    with pytest.raises(events.EngineError, match="outside"):
        layout.check_read(bad)
    bad = dict(good, has_skeleton=np.bool_(True))
    with pytest.raises(events.EngineError, match="lacks"):
        layout.check_read(bad)
    bad = dict(good, format=plainnpz.text("something/else"))
    with pytest.raises(events.EngineError):
        layout.check_read(bad)


def test_rig_layout_checks():
    rig = rigfiles.skintokens_rig_from_read({
        "matrix_local": np.tile(np.eye(4), (3, 1, 1)), "parents": np.array([-1, 0, 1]),
        "joint_names": plainnpz.texts(["a", "b", "c"]), "vertices": np.zeros((4, 3)), "skin": np.ones((4, 3)) / 3})
    assert layout.check_rig(rig) == ("skintokens", "final")
    for change, message in [({"parents": np.array([-1, -1, 1], np.int32)}, "one root"),
                            ({"names": plainnpz.texts(["a", "a", "c"])}, "repeated"),
                            ({"skin": np.full((4, 3), np.nan)}, "not numbers"),
                            ({"stage": plainnpz.text("skeleton")}, "stage")]:
        with pytest.raises(events.EngineError, match=message):
            layout.check_rig(dict(rig, **change))


def test_unirig_export_arguments_round_trip():
    recorder = capture.Recorder()
    recorder(object(), path="x.fbx", vertices=np.random.rand(5, 3).astype(np.float32),
             joints=np.random.rand(3, 3), skin=np.random.rand(5, 3), parents=[None, np.int64(0), 1],
             names=np.array(["a", "b", "mixamorig:c"]), faces=np.array([[0, 1, 2]]), group_per_vertex=4,
             tails=np.random.rand(3, 3), use_extrude_bone=False, use_connect_unique_child=False)
    call = recorder.calls[0]
    back = capture.to_call(capture.to_arrays(call))
    assert back["parents"] == [None, 0, 1] and back["names"] == ["a", "b", "mixamorig:c"]
    for key in ("vertices", "joints", "skin", "faces", "tails"):
        assert back[key].dtype == call[key].dtype
        np.testing.assert_array_equal(back[key], call[key])
    for key in capture.OPTION_ARGUMENTS:
        assert back[key] == call[key], key
    with pytest.raises(TypeError):
        recorder(object(), path="x", vertices=None, joints=None, skin=None, parents=[], names=[], colour="red")


def test_unirig_raw_data_without_pickle(engine, tmp_path):
    """The patched RawData saves plain arrays and loads them back as upstream's pickled files did:
    parents an object array of int and None, names a text array."""
    result = run_python(
        "import sys, json, numpy as np\n"
        "sys.path.insert(0, sys.argv[1] + '/models/unirig')\n"
        "from src.data.raw_data import RawData, RawSkin\n"
        "raw = RawData(vertices=np.random.rand(4, 3).astype(np.float32), vertex_normals=None,\n"
        "              faces=np.array([[0, 1, 2]]), face_normals=None, joints=np.random.rand(3, 3).astype(np.float32),\n"
        "              skin=np.random.rand(4, 3).astype(np.float32), parents=[None, 0, 1], names=['a', 'b', 'c'],\n"
        "              matrix_local=None, path=None, cls=None)\n"
        "raw.save(sys.argv[2])\n"
        "back = RawData.load(sys.argv[2])\n"
        "with np.load(sys.argv[2], allow_pickle=False) as data: files = sorted(data.files)\n"
        "print(json.dumps({'parents': list(back.parents), 'pdtype': str(back.parents.dtype),\n"
        "    'names': back.names.tolist(), 'ndtype': back.names.dtype.kind, 'files': files,\n"
        "    'no_skin': back.no_skin.tolist(), 'same': bool((back.joints == raw.joints).all())}))\n",
        engine, tmp_path / "raw_data.npz")
    assert result["parents"] == [None, 0, 1] and result["pdtype"] == "object"
    assert result["names"] == ["a", "b", "c"] and result["ndtype"] == "U"
    assert result["files"] == ["faces", "joints", "names", "parents", "skin", "vertices"]
    assert result["same"] and result["no_skin"] == [False, False, False]


# --- PROTOCOL.md in step with the code ------------------------------------------------------------

def _table_names(section):
    text = (REPO / "PROTOCOL.md").read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(section)}\n(.*?)(?=^## )", text, re.S | re.M)
    assert match, f"PROTOCOL.md has no section {section}"
    return [m for m in re.findall(r"^\| `([^`]+)` \|", match.group(1), re.M)]


def test_protocol_lists_every_error_code():
    assert _table_names("Error codes") == list(events.ERROR_CODES)


def test_protocol_lists_every_environment_variable():
    assert _table_names("Environment") == list(workspace.ENVIRONMENT)


def test_model_errors_become_stable_codes():
    class OutOfMemoryError(RuntimeError):
        pass

    from engine import guard

    assert events.model_failure(OutOfMemoryError("CUDA out of memory"), "x").code == "out_of_memory"
    assert events.model_failure(ValueError("all input arrays must have the same shape"), "x").code == "rig_failed"
    assert events.model_failure(guard.ForbiddenImport("no bpy here"), "x").code == "forbidden_module"
    kept = events.EngineError("skeleton_not_kept", "x")
    assert events.model_failure(kept, "y") is kept


def test_heartbeat_lines_are_whole(tmp_path):
    """The heartbeat thread and the main thread write at once; every line stays whole and valid."""
    import io
    import time

    from conftest import check_lines

    stream = io.StringIO()
    log = events.Events(stream, "rig")
    log.starting()
    log.start_heartbeat()
    log.progress("generating")
    time.sleep(events.HEARTBEAT_AFTER + 1.0)
    for _ in range(2000):
        log.progress("generating")
    time.sleep(events.HEARTBEAT_AFTER + 1.0)
    log.error("rig_failed", "done")
    time.sleep(1.0)
    lines = check_lines(stream.getvalue())
    beats = [e for e in lines if e.get("heartbeat")]
    assert len(beats) >= 2 and all(b["stage"] == "generating" for b in beats)
    assert lines[-1]["type"] == "error", "a heartbeat came after the last line"


def test_events_are_single_json_lines(tmp_path):
    import io

    stream = io.StringIO()
    log = events.Events(stream, "rig")
    log.progress("card", free_bytes=np.int64(5), card="Zé")
    log.error("rig_failed", "two\nlines")
    log.progress("saving")  # nothing after the last line
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2 and all(line.isascii() for line in lines)
    first = json.loads(lines[0])
    assert first["free_bytes"] == 5 and first["card"] == "Zé" and first["protocol"] == 1
    with pytest.raises(ValueError):
        events.EngineError("no_such_code", "x")


def test_protocol_lists_every_weight_file():
    from engine.weights import CONFIGS, WEIGHTS

    text = (REPO / "PROTOCOL.md").read_text(encoding="utf-8")
    section = text.split("## Weights\n", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| (SkinTokens|UniRig) \| `([^`]+)` \| `([0-9a-f]{64})` \|", section, re.M)
    listed = {(m.lower(), path, digest) for m, path, digest in rows}
    want = {(m, p, d) for m, files in WEIGHTS.items() for p, d in files.items()}
    want |= {(m, f"engine/models/{m}/{p}", d) for m, files in CONFIGS.items() for p, d in files.items()}
    assert listed == want


def test_protocol_lists_what_the_app_passes():
    import importlib.util

    spec = importlib.util.spec_from_file_location("kit", REPO / "windows-kit" / "run_engine_kit.py")
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    assert _table_names("Passed by the app") == list(kit.WINDOWS_VARIABLES)
