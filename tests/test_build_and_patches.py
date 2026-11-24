"""The build: pinned inputs, every patch applied exactly once to the pinned archives, nothing unsafe
left in the shipped model code, and the engine folder complete."""
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "build"))
import patches  # noqa: E402
import sources  # noqa: E402


def _unpack(archive, target):
    target.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        members = []
        for member in tar.getmembers():
            parts = Path(member.name).parts[1:]
            if parts:
                member.name = "/".join(parts)
                members.append(member)
        tar.extractall(target, members=members, filter="data")


@pytest.fixture(scope="module")
def fresh(downloads, scratch):
    root = scratch / "fresh-models"
    if root.exists():
        shutil.rmtree(root)
    for archive, where in sources.ARCHIVES.items():
        _unpack(downloads / archive, root / Path(where).name)
    return root


def test_pinned_archives_match(downloads):
    import hashlib

    for name in sources.ARCHIVES:
        digest = hashlib.sha256((downloads / name).read_bytes()).hexdigest()
        assert digest == sources.SOURCES[name][1], name


def test_every_patch_finds_its_text_exactly_once(fresh):
    for patch_id, model, rel, old, new, reason in patches.PATCHES:
        text = (fresh / model / rel).read_text(encoding="utf-8")
        assert text.count(old) == 1, patch_id
        assert old != new and reason, patch_id


def test_patch_ids_unique_and_documented():
    ids = [p[0] for p in patches.PATCHES]
    assert len(ids) == len(set(ids))
    doc = (REPO / "PATCHES.md").read_text(encoding="utf-8")
    for patch_id, *_ in patches.PATCHES:
        assert f"`{patch_id}`" in doc, f"{patch_id} is not in PATCHES.md"
    for model, rel, _ in patches.REMOVALS:
        assert f"`{model}/{rel}`" in doc, f"removal {model}/{rel} is not in PATCHES.md"


def test_patching_refuses_changed_text(fresh, scratch):
    copy = scratch / "changed-models"
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(fresh, copy)
    target = copy / "unirig" / "src" / "inference" / "download.py"
    target.write_text(target.read_text(encoding="utf-8").replace("hf_hub_download(", "hf_hub_download (", 1),
                      encoding="utf-8")
    with pytest.raises(patches.PatchError, match="ur-download-local"):
        patches.apply_patches(copy, log=lambda _line: None)


def test_patching_refuses_twice(fresh, scratch):
    copy = scratch / "twice-models"
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(fresh, copy)
    patches.apply_patches(copy, log=lambda _line: None)
    with pytest.raises(patches.PatchError, match="found 0"):
        patches.apply_patches(copy, log=lambda _line: None)


def test_patching_refuses_windows_line_endings(fresh, scratch):
    copy = scratch / "crlf-models"
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(fresh, copy)
    target = copy / "skintokens" / "src" / "model" / "tokenrig.py"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(patches.PatchError, match="Windows line endings"):
        patches.apply_patches(copy, log=lambda _line: None)


def test_engine_folder_is_complete(engine):
    root = engine.parent
    manifest = json.loads((root / "ENGINE.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(REPO))
    from engine import ENGINE_VERSION, PROTOCOL_VERSION
    from engine.weights import WEIGHTS

    assert (manifest["engine_version"], manifest["protocol"]) == (ENGINE_VERSION, PROTOCOL_VERSION)
    assert manifest["python"] == "python/tools/python.exe" and manifest["entry"] == "engine/rigengine.py"
    assert manifest["patches"] == [p[0] for p in patches.PATCHES]
    assert manifest["skintokens_commit"] == sources.SKINTOKENS_COMMIT
    assert manifest["unirig_commit"] == sources.UNIRIG_COMMIT
    assert manifest["weights"] == WEIGHTS
    for rel in ("rigengine.py", "engine/cli.py", "standin/flash_attn/modules/mha.py", "LICENSE", "PROTOCOL.md",
                "protocol.schema.json", "models/skintokens/models/Qwen3-0.6B/config.json",
                "models/unirig/hf/opt-350m/config.json", "models/skintokens/LICENSE", "models/unirig/LICENSE",
                "models/unirig/src/model/michelangelo/LICENSE"):
        assert (engine / rel).is_file(), rel
    for model, rel, _ in patches.REMOVALS:
        assert not (engine / "models" / model / rel).exists(), rel
    assert not [p for p in engine.rglob("*.ckpt")], "the engine folder holds no weights"


def test_longest_path_inside_the_root(engine):
    """Windows' 259 character limit: the build refuses any path below the root longer than PATH_LIMIT."""
    import build_engine

    manifest = json.loads((engine.parent / "ENGINE.json").read_text(encoding="utf-8"))
    longest = max(len("engine/" + p.relative_to(engine).as_posix()) for p in engine.rglob("*"))
    assert manifest["longest_relative_path"] == longest <= manifest["path_limit"] == build_engine.PATH_LIMIT
    install = "C:\\Users\\" + "u" * 20 + "\\AppData\\Roaming\\Nodelizer\\free-rig\\engine\\" + "v" * 15 + "\\"
    assert len(install) + build_engine.PATH_LIMIT <= 259 - 10


def test_build_stops_on_a_path_too_long(downloads, scratch, monkeypatch):
    import build_engine

    root = scratch / "too-long-root"
    if root.exists():
        shutil.rmtree(root)
    monkeypatch.setattr(build_engine, "PATH_LIMIT", 60)
    with pytest.raises(SystemExit, match="STOP.*characters"):
        build_engine.build_engine(downloads, root, False)
    assert not (root / "engine").exists() and not (root / "ENGINE.json").exists()
    assert not list(root.glob("engine.building-*"))


def test_weights_lists_agree():
    """engine/weights.py (what the engine checks) and build/sources.py (what the build fetches) pin the
    same files, and the build lays them out where the engine looks."""
    import build_engine

    sys.path.insert(0, str(REPO))
    from engine.weights import CONFIGS, WEIGHTS

    for model, names in build_engine.WEIGHT_SOURCES.items():
        laid_out = {sources.SOURCES[n][3].split(f"models/{model}/", 1)[1]: sources.SOURCES[n][1] for n in names}
        assert laid_out == WEIGHTS[model]
    for model, files in CONFIGS.items():
        for rel, digest in files.items():
            assert any(s[3] == f"models/{model}/{rel}" and s[1] == digest for s in sources.SOURCES.values())


def test_weights_build_needs_the_pinned_files(scratch):
    empty = scratch / "no-downloads"
    empty.mkdir(exist_ok=True)
    done = subprocess.run([sys.executable, str(REPO / "build" / "build_engine.py"), "--downloads", str(empty),
                           "--weights-out", str(scratch / "weights-never")], capture_output=True, text=True)
    assert done.returncode != 0 and "Missing" in done.stdout + done.stderr
    assert not (scratch / "weights-never" / "skintokens").exists()


def _shipped_python(engine):
    return [p for p in (engine / "models").rglob("*.py")]


def test_nothing_unsafe_left_in_shipped_code(engine):
    for path in _shipped_python(engine):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(engine)
        assert "allow_pickle=True" not in text, rel
        assert "weights_only=False" not in text, rel
        assert "hf_hub_download(" not in text, rel
        assert "snapshot_download(" not in text, rel
        assert "0.0.0.0" not in text, rel


def test_shipped_code_compiles(engine):
    done = subprocess.run([sys.executable, "-m", "compileall", "-q", "-f", "-x", "__pycache__",
                           str(engine / "engine"), str(engine / "models"), str(engine / "standin"),
                           str(engine / "rigengine.py")], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    for cache in engine.rglob("__pycache__"):  # leave the folder as built
        shutil.rmtree(cache)


def test_repository_code_compiles():
    """Every Python file of ours: build, engine, tests, the Windows kit (compiled in memory, since the
    repository is mounted read only)."""
    files = [p for p in REPO.rglob("*.py") if "dist" not in p.relative_to(REPO).parts]
    assert len(files) > 30
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_build_refuses_a_wrong_checksum(downloads, scratch):
    fake = scratch / "fake-downloads"
    if fake.exists():
        shutil.rmtree(fake)
    fake.mkdir()
    for name in ("skintokens.tar.gz", "qwen3-config.json", "opt-350m-config.json"):
        shutil.copyfile(downloads / name, fake / name)
    (fake / "unirig.tar.gz").write_bytes(b"not the reviewed archive")
    done = subprocess.run([sys.executable, str(REPO / "build" / "build_engine.py"), "--downloads", str(fake),
                           "--root", str(scratch / "never-built")], capture_output=True, text=True)
    assert done.returncode != 0
    assert "Checksum failed for" in done.stdout + done.stderr
    assert not (scratch / "never-built" / "engine").exists()
