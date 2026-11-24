# Zips the Windows overlay: the engine's source and the kit's runner, scripts and text only.
#
#   python windows-kit/make_kit.py --out dist/rig-engine-kit.zip
#
# The zip unpacks on top of the earlier kit's folder: rig-engine\ and engine-kit\.
import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENGINE_FILES = ["rigengine.py", "engine", "standin", "build", "LICENSE", "README.md", "PROTOCOL.md", "PATCHES.md",
                "THIRD_PARTY.md"]
KIT_FILES = {"README.md": "README.md", "run_engine_kit.py": "run_engine_kit.py", "expected.json": "expected.json"}
ALLOWED_SUFFIXES = {".py", ".md", ".json", ""}
LIMIT = 1024 * 1024  # the overlay is small; anything bigger means something slipped in


def files():
    for name in ENGINE_FILES:
        path = REPO / name
        found = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for item in found:
            if "__pycache__" in item.parts or item.suffix == ".pyc":
                continue
            yield item, "rig-engine/" + item.relative_to(REPO).as_posix()
    for source, target in KIT_FILES.items():
        yield REPO / "windows-kit" / source, "engine-kit/" + target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    part = args.out.with_name(args.out.name + ".part")
    names = []
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, name in files():
            if path.suffix not in ALLOWED_SUFFIXES:
                raise SystemExit(f"STOP. {path} is not a script or text file")
            data = path.read_bytes()
            if data[:2] == b"MZ" or b"\x00" in data:
                raise SystemExit(f"STOP. {path} looks like a program or binary file")
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))  # same bytes in, same zip out
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
            names.append(name)
    if part.stat().st_size > LIMIT:
        raise SystemExit(f"STOP. The overlay is {part.stat().st_size} bytes; it should be scripts and text only")
    part.replace(args.out)
    print(f"{args.out}: {len(names)} files, {args.out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
