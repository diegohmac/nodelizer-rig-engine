# The macOS version each Mach-O file in a folder says it needs: LC_BUILD_VERSION's minos, or the older
# LC_VERSION_MIN_MACOSX, for every architecture in it. Reads the headers itself (no otool needed), so
# the build can refuse a root whose libraries need a newer macOS than ENGINE.json declares.
# Standard library only.
import os
import struct

FAT = {b"\xca\xfe\xba\xbe": ">I", b"\xca\xfe\xba\xbf": ">Q"}  # fat header, 32 and 64 bit offsets
THIN64 = b"\xcf\xfa\xed\xfe"
THIN32 = b"\xce\xfa\xed\xfe"
LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32
PLATFORM_MACOS = 1


def _version(packed):
    return (packed >> 16, (packed >> 8) & 0xFF, packed & 0xFF)


def _thin(data, at):
    """The minimum macOS of one architecture's image starting at `at`, or None if it states none."""
    magic = data[at:at + 4]
    header = 32 if magic == THIN64 else 28
    ncmds = struct.unpack_from("<I", data, at + 16)[0]
    offset = at + header
    found = None
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, offset)
        if cmd == LC_BUILD_VERSION:
            platform, minos = struct.unpack_from("<II", data, offset + 8)
            if platform == PLATFORM_MACOS:
                found = _version(minos)
        elif cmd == LC_VERSION_MIN_MACOSX and found is None:
            found = _version(struct.unpack_from("<I", data, offset + 8)[0])
        offset += size
    return found


def minimums(path):
    """[(cpu type, (major, minor, patch) or None)] for a Mach-O file, or [] for any other file."""
    with open(path, "rb") as handle:
        head = handle.read(4)
        if head not in FAT and head not in (THIN64, THIN32):
            return []
        handle.seek(0)
        data = handle.read()
    if head in (THIN64, THIN32):
        return [(struct.unpack_from("<i", data, 4)[0], _thin(data, 0))]
    wide = FAT[head] == ">Q"
    count = struct.unpack_from(">I", data, 4)[0]
    found = []
    for index in range(count):
        if wide:
            cpu, _sub, offset, _size, _align, _reserved = struct.unpack_from(">iiQQII", data, 8 + index * 32)
        else:
            cpu, _sub, offset, _size, _align = struct.unpack_from(">iiIII", data, 8 + index * 20)
        found.append((cpu, _thin(data, offset)))
    return found


def scan(folder):
    """{relative path: [(cpu type, minimum)]} for every Mach-O file below `folder` (links skipped)."""
    found = {}
    for where, _dirs, names in os.walk(folder):
        for name in names:
            path = os.path.join(where, name)
            if os.path.islink(path):
                continue
            versions = minimums(path)
            if versions:
                found[os.path.relpath(path, folder)] = versions
    return found


def parse(text):
    parts = [int(p) for p in text.split(".")]
    return tuple(parts + [0] * (3 - len(parts)))


def check(folder, declared):
    """Raises SystemExit if a Mach-O file below `folder` needs a newer macOS than `declared` ("14.0").
    Returns (Mach-O file count, the newest minimum found)."""
    limit = parse(declared)
    found = scan(folder)
    newest = max((v for versions in found.values() for _cpu, v in versions if v), default=(0, 0, 0))
    over = sorted(f"{path} needs macOS {'.'.join(map(str, v))}" for path, versions in found.items()
                  for _cpu, v in versions if v and v > limit)
    if over:
        raise SystemExit(f"STOP. {len(over)} file(s) need a newer macOS than {declared}: " + "; ".join(over[:10]))
    return len(found), newest
