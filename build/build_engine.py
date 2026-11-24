# Builds the rig engine from pinned inputs: the engine folder of a package root, and, apart from it,
# one weights folder per model.
#
#   python build/build_engine.py --downloads <folder> --root <package root> [--weights-out <folder>] [--fetch]
#   python build/build_engine.py --downloads <folder> --weights-out <folder>
#   python build/build_engine.py --downloads <folder> --fetch-only
#
# --downloads  a folder holding the pinned files under the names in build/sources.py. The Windows
#              kit's kit\fetch.py leaves exactly these in its downloads\ folder.
# --root       the package root: this makes <root>/engine (code only, no weights) and <root>/ENGINE.json.
#              The root may exist (it also holds python/, placed by the release); engine/ and
#              ENGINE.json must not. The engine folder is built under a temporary name and renamed.
# --weights-out  makes <folder>/skintokens and <folder>/unirig, each with its model's weight files at
#              the paths in engine/weights.py and a WEIGHTS.json listing them. A model folder that exists
#              already is left alone.
# --fetch      download any pinned file that is missing (network; each one checked by SHA-256).
#
# Every input's SHA-256 is checked before it is used. The engine folder gets the two code archives
# (tar "data" filter, top folder dropped), build/patches.py (each an exact, checked text replacement),
# the two config files and the engine's own code. The build fails if any path inside the package
# root would be longer than PATH_LIMIT characters (see PROTOCOL.md, Packaged layout).
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from patches import PATCHES, REMOVALS, apply_patches, apply_removals  # noqa: E402
from sources import ARCHIVES, SKINTOKENS_COMMIT, SOURCES, UNIRIG_COMMIT  # noqa: E402

# Our own files copied into the engine folder.
OWN = ["rigengine.py", "engine", "standin", "LICENSE", "README.md", "PROTOCOL.md", "PATCHES.md", "THIRD_PARTY.md",
       "protocol.schema.json"]
PYTHON = "python/tools/python.exe"  # inside the root: the signed Python of the Windows kit, as it is laid out
# Windows' classic path limit is 259 characters. The app installs the root at
#   C:\Users\<name>\AppData\Roaming\Nodelizer\free-rig\engine\<version>\
# which is 88 characters with a 20 character user name (Windows' longest) and a 15 character version.
# 259 - 88 = 171; 160 leaves 11 for a longer profile folder (C:\Users\name.DOMAIN) or install path.
PATH_LIMIT = 160
WEIGHT_SOURCES = {"skintokens": ("skintokens-grpo_1400.ckpt", "skintokens-skin_vae-last.ckpt"),
                  "unirig": ("unirig-skeleton-model.ckpt", "unirig-skin-model.ckpt")}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, target):
    part = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "rig-engine-build"})
    with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as handle:
        shutil.copyfileobj(response, handle, 1 << 20)
    part.replace(target)


def check_inputs(downloads, names, allow_fetch):
    used = {}
    for name in names:
        url, digest, _kind, _place = SOURCES[name]
        path = downloads / name
        if not path.exists():
            if not allow_fetch:
                raise SystemExit(f"STOP. Missing {path}. Run the kit's fetch.py, or this script with --fetch.")
            print(f"fetching           {name}", flush=True)
            downloads.mkdir(parents=True, exist_ok=True)
            fetch(url, path)
        got = sha256(path)
        if got != digest:
            raise SystemExit(f"STOP. Checksum failed for {path}\n  expected {digest}\n  got      {got}")
        print(f"ok                 {name}", flush=True)
        used[name] = digest
    return used


def unpack(archive, target):
    target.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        members = []
        for member in tar.getmembers():
            parts = Path(member.name).parts[1:]  # drop the archive's top folder
            if not parts:
                continue
            member.name = "/".join(parts)
            members.append(member)
        tar.extractall(target, members=members, filter="data")


def copy_own(out):
    for name in OWN:
        source = REPO / name
        if source.is_dir():
            shutil.copytree(source, out / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, out / name)


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:  # a hard link saves gigabytes when the downloads folder is on the same drive
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def longest_path(folder, prefix):
    """The longest path below folder, as written from the package root (prefix/...)."""
    longest = (len(prefix), prefix)
    for path in folder.rglob("*"):
        rel = prefix + "/" + path.relative_to(folder).as_posix()
        if len(rel) > longest[0]:
            longest = (len(rel), rel)
    return longest


def build_engine(downloads, root, allow_fetch):
    sys.path.insert(0, str(REPO))
    from engine import ENGINE_VERSION, PROTOCOL_VERSION
    from engine.weights import CONFIGS, WEIGHTS

    out, manifest_path = root / "engine", root / "ENGINE.json"
    for path in (out, manifest_path):
        if path.exists():
            raise SystemExit(f"STOP. {path} exists already. Remove it first (it is rebuilt from scratch).")
    code_inputs = [name for name, (_u, _d, kind, _p) in SOURCES.items() if kind == "code"]
    used = check_inputs(downloads, code_inputs, allow_fetch)
    root.mkdir(parents=True, exist_ok=True)
    building = root / f"engine.building-{os.getpid()}"
    try:
        models = building / "models"
        for archive, where in ARCHIVES.items():
            unpack(downloads / archive, building / where)
            print(f"unpacked           {where}", flush=True)
        apply_patches(models, log=lambda line: print(line, flush=True))
        apply_removals(models, log=lambda line: print(line, flush=True))
        for name, (_url, _digest, kind, where) in SOURCES.items():
            if kind == "code" and where is not None:
                link_or_copy(downloads / name, building / where)
                print(f"placed             {where}", flush=True)
        copy_own(building)
        length, path = longest_path(building, "engine")
        if length > PATH_LIMIT:
            raise SystemExit(f"STOP. {path} is {length} characters; the limit is {PATH_LIMIT} (see PATH_LIMIT)")
        manifest = {
            "engine_version": ENGINE_VERSION,
            "protocol": PROTOCOL_VERSION,
            "python": PYTHON,
            "entry": "engine/rigengine.py",
            "skintokens_commit": SKINTOKENS_COMMIT,
            "unirig_commit": UNIRIG_COMMIT,
            "inputs": used,
            "patches": [patch[0] for patch in PATCHES],
            "removed": [f"{model}/{rel}" for model, rel, _ in REMOVALS],
            "weights": WEIGHTS,
            "configs": {model: {f"engine/models/{model}/{rel}": digest for rel, digest in files.items()}
                        for model, files in CONFIGS.items()},
            "longest_relative_path": length,
            "path_limit": PATH_LIMIT,
        }
        building.rename(out)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    print(f"longest path       {length} characters: {path}", flush=True)
    print(f"build done         {out}", flush=True)


def build_weights(downloads, folder, allow_fetch):
    sys.path.insert(0, str(REPO))
    from engine.weights import WEIGHTS

    for model, names in WEIGHT_SOURCES.items():
        target = folder / model
        if target.exists():
            print(f"left alone         {target} (exists)", flush=True)
            continue
        check_inputs(downloads, names, allow_fetch)
        building = folder / f".{model}.building-{os.getpid()}"
        files = {}
        try:
            for name in names:
                rel = SOURCES[name][3].split(f"models/{model}/", 1)[1]
                assert WEIGHTS[model][rel] == SOURCES[name][1], rel
                link_or_copy(downloads / name, building / rel)
                files[rel] = {"sha256": SOURCES[name][1], "bytes": (downloads / name).stat().st_size}
            (building / "WEIGHTS.json").write_text(json.dumps({"model": model, "files": files}, indent=2) + "\n",
                                                   encoding="utf-8")
            building.rename(target)
        except BaseException:
            shutil.rmtree(building, ignore_errors=True)
            raise
        print(f"weights            {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build the rig engine from pinned inputs.")
    parser.add_argument("--downloads", required=True, type=Path)
    parser.add_argument("--root", type=Path, help="the package root to put engine/ and ENGINE.json in")
    parser.add_argument("--weights-out", type=Path, help="where to lay out one weights folder per model")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--fetch-only", action="store_true", help="fetch and check the code inputs, build nothing")
    args = parser.parse_args()
    downloads = args.downloads.resolve()
    if args.fetch_only:
        check_inputs(downloads, [n for n, s in SOURCES.items() if s[2] == "code"], allow_fetch=True)
        return
    if args.root is None and args.weights_out is None:
        parser.error("give --root, --weights-out or both")
    if args.root is not None:
        build_engine(downloads, args.root.resolve(), args.fetch)
    if args.weights_out is not None:
        build_weights(downloads, args.weights_out.resolve(), args.fetch)


if __name__ == "__main__":
    main()
