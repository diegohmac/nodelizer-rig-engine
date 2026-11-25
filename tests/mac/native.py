# The native Mac test, run by test-mac.sh with the Mac root's own Python (-I). It starts every engine
# step inside sandbox-exec with mac/rig-engine.sb, as the app will, and checks the lines it prints.
# Needs a built Mac root (build/mac/build_mac.py) and SkinTokens' weights folder.
#   python -I native.py --root R --weights W --input FILE --scratch S [--twice] [--report FILE]
# --report writes a JSON report for the release tooling (REPORT_SHAPE below says what is in it).
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "mac" / "rig-engine.sb"
# What --report writes, under stable names (the release tooling reads it):
REPORT_SHAPE = """
schema 1, kind "rig-engine-mac-test", exitCode (0 when every check passed), passed, seconds, twice,
input (the model file's name), seed,
machine {macos, build, model (the chip), memoryBytes},
root {inventory, files, bytes, linksLeftOut, compiledLeftOut} (of --root, as the app hashes it),
engine (ENGINE.json as read),
cases [{step (read, rig or write), model, character, seed, exitCode, ok, wallSeconds, engineSeconds,
        stages {stage: seconds}, capability, mode, modeAsked, peakBytes, needsFreeBytes, bones, names,
        fingerprints {points, tokens, joints, skin}, error {code, message} or null}],
repeats {modes [first, second], matched} or null without --twice,
sandbox [{test, expected, outcome}],
checks [{section, ok, what}] (every printed check, in order)
"""
failures = []
checks = []  # {section, ok, what}, in order, as printed
section = [""]


def heading(text):
    section[0] = text
    print("== " + text, flush=True)


def check(ok, what):
    ok = bool(ok)
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    checks.append({"section": section[0], "ok": ok, "what": what})
    if not ok:
        failures.append(what)


def inventory(root):
    """The root's inventory hash as the app computes it (engine-manifest.ts, inventoryHash): SHA-256 over
    one line per file, `<path>\t<size>\t<sha256>\n`, paths relative with forward slashes, sorted by their
    UTF-8 bytes. Links and compiled Python (__pycache__, .pyc) are left out and counted."""
    entries, links, compiled = [], 0, 0
    for where, dirs, names in os.walk(root):
        for name in list(dirs):
            if os.path.islink(os.path.join(where, name)):
                links += 1
                dirs.remove(name)
        for name in names:
            path = os.path.join(where, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if os.path.islink(path):
                links += 1
                continue
            if "__pycache__" in rel.split("/") or name.endswith(".pyc"):
                compiled += 1
                continue
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(chunk)
            entries.append((rel.encode("utf-8"), os.path.getsize(path), digest.hexdigest()))
    entries.sort()
    total = hashlib.sha256()
    for rel, size, sha in entries:
        total.update(rel + f"\t{size}\t{sha}\n".encode("ascii"))
    return {"inventory": total.hexdigest(), "files": len(entries), "bytes": sum(e[1] for e in entries),
            "linksLeftOut": links, "compiledLeftOut": compiled}


def machine():
    def ask(*command):
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()

    return {"macos": ask("/usr/bin/sw_vers", "-productVersion"), "build": ask("/usr/bin/sw_vers", "-buildVersion"),
            "model": ask("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
            "memoryBytes": int(ask("/usr/sbin/sysctl", "-n", "hw.memsize") or 0)}


def stages(lines):
    """Seconds per stage, from the progress lines (heartbeats left out)."""
    marks = [(line["seconds"], line["stage"]) for line in lines
             if line.get("type") == "progress" and not line.get("heartbeat")]
    end = lines[-1].get("seconds", 0) if lines else 0
    found = {}
    for (at, stage), (after, _) in zip(marks, marks[1:] + [(end, None)]):
        found[stage] = round(found.get(stage, 0) + after - at, 2)
    return found


def sandboxed(root, weights, input_dir, work, cache, args, metal=False):
    """The app's command for one step: Metal's driver for Blender's steps (read, write) only."""
    python = root / "python" / "bin" / "python3.11"
    return ["/usr/bin/sandbox-exec", "-f", str(PROFILE), "-D", f"ROOT={root}", "-D", f"PYTHON={python}",
            "-D", f"WEIGHTS={weights}", "-D", f"INPUT={input_dir}", "-D", f"WORK={work}", "-D", f"CACHE={cache}",
            *(["-D", "METAL=allow"] if metal else []), str(python), "-I", "-B", *args]


def step(folders, *args):
    """One engine step in the sandbox, with an empty environment. Returns (exit code, lines, seconds)."""
    command = sandboxed(*folders, ["-u", "-X", "utf8", str(folders[0] / "engine" / "rigengine.py"), *map(str, args)],
                        metal=args[0] in ("read", "write"))
    started = time.monotonic()
    done = subprocess.run(command, capture_output=True, env={}, cwd=str(folders[3]))
    seconds = time.monotonic() - started
    lines = []
    for raw in done.stdout.decode("ascii", "replace").splitlines():
        try:
            lines.append(json.loads(raw))
        except ValueError:
            lines.append({"type": "not_json", "text": raw})
    if done.returncode not in (0, 1, 2) or not lines:
        sys.stderr.write(done.stderr.decode("utf-8", "replace")[-3000:])
    return done.returncode, lines, seconds


def well_formed(name, lines):
    check(all(line.get("type") in ("progress", "result", "error") for line in lines), f"{name}: every line is JSON")
    check(bool(lines) and lines[0].get("stage") == "starting", f"{name}: the first line is starting")


def snippet(folders, code, metal=False):
    """A few lines of Python in the sandbox, with the root's Python. Returns (exit code, what it printed)."""
    done = subprocess.run(sandboxed(*folders, ["-c", code], metal), capture_output=True, text=True, env={},
                          cwd=str(folders[3]))
    return done.returncode, done.stdout.strip()


DENIED = """
import errno, os, socket, subprocess
HOME, OUTSIDE, INSIDE = {home!r}, {outside!r}, {inside!r}
def attempt(name, action):
    try:
        action()
        print(name, "allowed")
    except OSError as error:
        print(name, "denied" if error.errno in (errno.EPERM, errno.EACCES) else f"other_{{error.errno}}")
def connect():
    with socket.create_connection(("1.1.1.1", 443), timeout=5):
        pass
def local_unix():
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
        s.connect("/var/run/syslog")
def local_tcp():
    with socket.create_connection(("127.0.0.1", 9), timeout=5):
        pass
def listen():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
attempt("internet", connect)
attempt("local_unix_socket", local_unix)
attempt("local_tcp", local_tcp)
attempt("listen", listen)
attempt("read_documents", lambda: os.listdir(os.path.join(HOME, "Documents")))
attempt("read_home", lambda: os.listdir(HOME))
attempt("write_outside", lambda: open(OUTSIDE, "w").close())
attempt("start_program", lambda: subprocess.run(["/bin/echo", "x"], check=True, capture_output=True))
attempt("write_work", lambda: open(INSIDE, "w").close())
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--twice", action="store_true",
                        help="rig again in the other memory mode and compare the fingerprints")
    parser.add_argument("--report", type=Path, help="write a JSON report here (REPORT_SHAPE)")
    args = parser.parse_args()
    report = {"schema": 1, "kind": "rig-engine-mac-test", "twice": bool(args.twice), "input": args.input.name,
              "seed": 12345, "machine": machine(), "root": inventory(args.root.resolve()),
              "engine": json.loads((args.root / "ENGINE.json").read_text()), "cases": [], "sandbox": [],
              "repeats": None}
    started = time.monotonic()
    try:
        run(args, report)
    except Exception as error:  # the report still says what ran and what failed
        check(False, f"the test stopped: {type(error).__name__}: {error}")
    print("== done: " + ("all passed" if not failures else f"{len(failures)} failed"), flush=True)
    code = 1 if failures else 0
    if args.report:
        report.update(exitCode=code, passed=code == 0 and bool(checks), checks=checks,
                      seconds=round(time.monotonic() - started, 1))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return code


def run(args, report):

    # A copy of the root under a path with spaces and accents (an APFS clone: no extra disk space).
    scratch = args.scratch.resolve()
    shutil.rmtree(scratch, ignore_errors=True)
    root = scratch / "Zé root" / "rig engine"
    root.parent.mkdir(parents=True)
    subprocess.run(["/bin/cp", "-cR", str(args.root.resolve()), str(root)], check=True)
    work = scratch / "work ü"
    cache = scratch / "cache"
    work.mkdir()
    cache.mkdir()
    weights = args.weights.resolve()
    source = args.input.resolve()
    folders = (root, weights, work, work, cache)
    with_input = (root, weights, source.parent, work, cache)

    heading("the root")
    manifest = json.loads((root / "ENGINE.json").read_text())
    check(manifest.get("platform") == "darwin-arm64", "ENGINE.json says darwin-arm64")
    check(manifest.get("models") == ["skintokens"], "ENGINE.json lists SkinTokens alone")
    check(manifest.get("python") == "python/bin/python3.11", "ENGINE.json names the Mac Python")
    check(manifest.get("minimum_macos") == "14.0", "ENGINE.json says macOS 14.0 or later")
    site = root / "python" / "lib" / "python3.11" / "site-packages"
    check(not any((site / name).exists() for name in ("pip", "setuptools", "wheel")), "no pip, setuptools or wheel")
    check(not any(root.rglob("__pycache__")), "no __pycache__ folders")

    heading("probe")
    installed = int(subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True,
                                   text=True).stdout)
    mode = "fast" if installed >= 30 * 2**30 else "lean"
    code, lines, seconds = step(folders, "probe", "--light")
    well_formed("probe --light", lines)
    last = lines[-1]
    device = (last.get("devices") or [{}])[0]
    check(code == 0 and last.get("devices_from") == "host" and len(last.get("devices", [])) == 1,
          f"probe --light: one device, from the system ({seconds:.1f} s)")
    check("torch" not in last and device.get("capability") == f"cpu:{mode}",
          f"probe --light: no PyTorch, mode {device.get('capability')}")
    check(last.get("models", {}).get("skintokens", {}).get("needs_free_bytes") == (13 if mode == "fast" else 7) * 2**30,
          "probe --light: SkinTokens' needs for the mode")
    code, lines, seconds = step(folders, "probe", "--weights", f"skintokens={weights}")
    last = lines[-1]
    check(code == 0 and list(last.get("models", {})) == ["skintokens"], f"probe: SkinTokens alone ({seconds:.1f} s)")
    check(last.get("models", {}).get("skintokens", {}).get("weights") is True, "probe: the weights folder is complete")
    check(last.get("forbidden_modules_loaded") == [], "probe: no Blender")

    heading("the sandbox")
    home = Path.home()
    outside = Path(os.environ.get("TMPDIR", "/tmp")) / f"rig-engine-escape-{os.getpid()}"
    _, said = snippet(folders, DENIED.format(home=str(home), outside=str(outside), inside=str(work / "probe.txt")))
    answers = dict(line.split(" ", 1) for line in said.splitlines() if " " in line)
    for name in ("internet", "local_unix_socket", "local_tcp", "listen", "read_documents", "read_home",
                 "write_outside", "start_program"):
        check(answers.get(name) == "denied", f"{name} is refused ({answers.get(name)})")
        report["sandbox"].append({"test": name, "expected": "denied", "outcome": answers.get(name)})
    check(answers.get("write_work") == "allowed", "writing in the work folder works")
    report["sandbox"].append({"test": "write_work", "expected": "allowed", "outcome": answers.get("write_work")})
    check(not outside.exists(), "nothing was written outside")
    report["sandbox"].append({"test": "nothing_outside", "expected": "absent",
                              "outcome": "present" if outside.exists() else "absent"})
    _, said = snippet(folders, "import torch; print(torch.backends.mps.is_available())")
    check(said == "False", f"rig and probe: the GPU (Metal) is out of reach ({said})")
    report["sandbox"].append({"test": "gpu_without_metal", "expected": "denied",
                              "outcome": "denied" if said == "False" else "allowed"})
    code, said = snippet(folders, "import bpy; print('started')", metal=True)
    check(code == 0 and said.endswith("started"), "read and write: Blender's module starts with Metal's driver")
    report["sandbox"].append({"test": "blender_with_metal", "expected": "allowed",
                              "outcome": "allowed" if code == 0 and said.endswith("started") else f"exit {code}"})

    heading("UniRig is refused")
    code, lines, _ = step(with_input, "read", "--model", "unirig", "--input", source, "--output",
                          work / "u.read.npz", "--work", work)
    check(code == 1 and lines[-1].get("code") == "model_unavailable", "read --model unirig: model_unavailable")

    heading("read, rig, write: " + source.name)
    code, lines, seconds = step(with_input, "read", "--model", "skintokens", "--input", source, "--output",
                                work / "a.read.npz", "--work", work)
    well_formed("read", lines)
    check(code == 0, f"read ({seconds:.1f} s)")
    report["cases"].append(case("read", lines, code, seconds, source.name))
    fingerprints = []
    other = "lean" if mode == "fast" else "fast"
    for run, memory_args in enumerate([[], ["--memory", other]][:2 if args.twice else 1]):
        expected = other if memory_args else mode
        code, lines, seconds = step(folders, "rig", "--model", "skintokens", "--input", work / "a.read.npz",
                                    "--output", work / f"a{run}.rig.npz", "--work", work, "--weights", weights,
                                    "--cache", cache, "--seed", "12345", *memory_args)
        well_formed("rig", lines)
        last = lines[-1]
        card = next((line for line in lines if line.get("stage") == "card"), {})
        check(code == 0, f"rig ({seconds:.1f} s; {last.get('code', '')} {last.get('message', '')[:200]})")
        check(card.get("capability") == f"cpu:{expected}", f"rig: the processor, {card.get('capability')}")
        memory = last.get("memory", {})
        check(0 < memory.get("peak_allocated_bytes", 0) <= memory.get("needs_free_bytes", 0),
              f"rig: peak {memory.get('peak_allocated_bytes', 0) / 2**30:.2f} GiB within needs_free_bytes "
              f"{memory.get('needs_free_bytes', 0) / 2**30:.0f} GiB")
        check(last.get("bones", 0) > 0, f"rig: {last.get('bones')} bones, names {last.get('names')}")
        fingerprints.append(last.get("fingerprints"))
        report["cases"].append(dict(case("rig", lines, code, seconds, source.name), mode=expected,
                                    modeAsked=other if memory_args else None))
    if args.twice:
        check(fingerprints[0] == fingerprints[1], f"{mode} and {other}: the same fingerprints {fingerprints[0]}")
        report["repeats"] = {"modes": [mode, other], "matched": fingerprints[0] == fingerprints[1]}
    code, lines, seconds = step(with_input, "write", "--rig", work / "a0.rig.npz", "--input", source, "--output",
                                work / "rigged.glb", "--work", work)
    well_formed("write", lines)
    check(code == 0 and lines[-1].get("bytes", 0) > 0, f"write ({seconds:.1f} s)")
    report["cases"].append(case("write", lines, code, seconds, source.name))
    left = sorted(p.name for p in work.iterdir())
    check(not [name for name in left if name.startswith(".")], f"no step folders left in work ({left})")
    if not failures:
        shutil.rmtree(scratch, ignore_errors=True)


def case(step_name, lines, code, seconds, character):
    """One step of the sequence for the report."""
    last = lines[-1] if lines else {}
    memory = last.get("memory") or {}
    card = next((line for line in lines if line.get("stage") == "card"), {})
    return {"step": step_name, "model": "skintokens", "character": character,
            "seed": 12345 if step_name == "rig" else None,
            "exitCode": code, "ok": code == 0 and last.get("type") == "result", "wallSeconds": round(seconds, 2),
            "engineSeconds": last.get("seconds"), "stages": stages(lines), "capability": card.get("capability"),
            "mode": None, "modeAsked": None,
            "peakBytes": memory.get("peak_allocated_bytes"), "needsFreeBytes": memory.get("needs_free_bytes"),
            "bones": last.get("bones"), "names": last.get("names"), "fingerprints": last.get("fingerprints"),
            "error": ({"code": last.get("code"), "message": last.get("message")}
                      if last.get("type") == "error" else None)}


if __name__ == "__main__":
    sys.exit(main())
