# Builds a Mac root (darwin-arm64) from pinned inputs: python/ (Astral's python-build-standalone with
# the locked packages), engine/ (SkinTokens only) and ENGINE.json.
#
#   python3 build/mac/build_mac.py --downloads <folder> --cache <folder> --root <new folder>
#
# --downloads  the engine's pinned code inputs (build/sources.py), as for build_engine.py
# --cache      where the pinned Python and the wheels are fetched to and checked (kept between builds)
# --root       made new; it must not exist
# --engine-only  rebuild only engine/ and ENGINE.json in an existing root (after a change to the engine)
#
# The wheels are fetched and installed by pip in Docker (Linux), for macOS on Apple silicon, with every
# file checked against requirements-mac.txt's SHA-256. Installing a wheel only unpacks it, so nothing
# from the packages runs during the build. Afterwards: the console scripts pip writes for the wrong
# platform, pip and setuptools, and the packages in sources_mac.NOT_SHIPPED are removed; every .pth file
# left is listed. The root then runs from any path: python -I -B needs no PYTHONHOME or settings.
import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUILD))

import macho  # noqa: E402
import sources_mac  # noqa: E402

PIP_IMAGE = ("python:3.11.13-slim-bookworm@sha256:"
             "86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1")
# pip takes this as the newest macOS whose wheels it may pick; it also takes older ones.
WHEEL_PLATFORM = "macosx_14_0_arm64"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_python(cache):
    pin = sources_mac.PYTHON
    target = cache / pin["file"]
    if not target.exists():
        print(f"fetching           {pin['file']}", flush=True)
        part = target.with_name(target.name + ".part")
        request = urllib.request.Request(pin["url"], headers={"User-Agent": "rig-engine-build"})
        with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)
        part.replace(target)
    got = sha256(target)
    if got != pin["sha256"]:
        raise SystemExit(f"STOP. Checksum failed for {target}\n  expected {pin['sha256']}\n  got      {got}")
    print(f"ok                 {pin['file']}", flush=True)
    return target


def unpack_python(archive, root):
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")  # the archive's top folder is python/
    if not (root / sources_mac.EXECUTABLE).exists():
        raise SystemExit(f"STOP. {sources_mac.EXECUTABLE} is not in the Python archive")


def docker(*args):
    subprocess.run(["docker", "run", "--rm", *args], check=True)


def install_wheels(cache, site):
    wheels = cache / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    platform = ["--platform", WHEEL_PLATFORM, "--python-version", "3.11", "--implementation", "cp",
                "--abi", "cp311", "--abi", "abi3", "--abi", "none", "--only-binary=:all:"]
    mounts = ["-v", f"{HERE}:/mac:ro", "-v", f"{wheels}:/wheels"]
    print("fetching           the wheels in requirements-mac.txt (pip in Docker, hashes required)", flush=True)
    docker(*mounts, PIP_IMAGE, "pip", "download", "--quiet", "--disable-pip-version-check", "--no-deps",
           "--require-hashes", "-r", "/mac/requirements-mac.txt", "-d", "/wheels", *platform)
    site.mkdir(parents=True, exist_ok=True)
    print("installing         the wheels into python/ (no network)", flush=True)
    docker("--network", "none", *mounts, "-v", f"{site}:/site", PIP_IMAGE, "pip", "install", "--quiet",
           "--disable-pip-version-check", "--no-index", "--find-links", "/wheels", "--no-deps", "--require-hashes",
           "--no-compile", "-r", "/mac/requirements-mac.txt", "--target", "/site", *platform)


def tidy(root, site):
    removed = []
    for path in sorted(site.glob("bin")):  # console scripts written for the platform pip ran on
        shutil.rmtree(path)
        removed.append(str(path.relative_to(root)))
    for name in list(sources_mac.NOT_SHIPPED) + list(sources_mac.PYTHON_TOOLS_REMOVED):
        for path in [site / name, *site.glob(f"{name}-*.dist-info")]:
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path.relative_to(root)))
            elif path.exists():
                path.unlink()
                removed.append(str(path.relative_to(root)))
    for cache in list(root.rglob("__pycache__")):
        shutil.rmtree(cache)
    for path in [*(root / "python" / "bin").glob("pip*")]:  # the Python archive's own pip launchers
        path.unlink()
        removed.append(str(path.relative_to(root)))
    for rel in sources_mac.PYTHON_PARTS_REMOVED:
        path = root / rel
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            raise SystemExit(f"STOP. {rel} is not in Python's archive; check PYTHON_PARTS_REMOVED")
        removed.append(rel)
    links = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_symlink()]
    if links:
        raise SystemExit(f"STOP. Links are left in the root: {links[:10]}")
    return removed


def _minimum_macos(root):
    """ENGINE.json's minimum_macos, after checking that no Mach-O file in the root needs a newer macOS."""
    count, newest = macho.check(root, sources_mac.MIN_MACOS)
    print(f"min macOS          {sources_mac.MIN_MACOS}: {count} Mach-O files, the newest needs "
          f"{'.'.join(map(str, newest))}", flush=True)
    return {"minimum_macos": sources_mac.MIN_MACOS}


def main():
    parser = argparse.ArgumentParser(description="Build a Mac rig engine root from pinned inputs.")
    parser.add_argument("--downloads", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--engine-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.engine_only:
        import build_engine

        shutil.rmtree(root / "engine", ignore_errors=True)
        (root / "ENGINE.json").unlink(missing_ok=True)
        build_engine.build_engine(args.downloads.resolve(), root, False, "darwin-arm64", extra=_minimum_macos(root))
        return
    if root.exists():
        raise SystemExit(f"STOP. {root} exists already. Choose a new folder.")
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    archive = fetch_python(cache)
    building = root.with_name(root.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        unpack_python(archive, building)
        site = building / sources_mac.SITE_PACKAGES
        install_wheels(cache, site)
        removed = tidy(building, site)
        for line in removed:
            print(f"removed            {line}", flush=True)
        import build_engine

        build_engine.build_engine(args.downloads.resolve(), building, False, "darwin-arm64",
                                  extra=_minimum_macos(building))
        pth = sorted(str(p.relative_to(building)) for p in building.rglob("*.pth"))
        print(f".pth files         {pth or 'none'}", flush=True)
        building.rename(root)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    print(f"mac build done     {root}", flush=True)


if __name__ == "__main__":
    main()
