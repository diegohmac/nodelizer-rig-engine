# Runs the rig engine on this Windows computer, every step as its own process inside the app's
# sandbox launcher (run-restricted.exe), and writes a report. See engine-kit\README.md.
#
#   python engine-kit\run_engine_kit.py --snapshot     once, before changing anything (for the diffs)
#   python engine-kit\run_engine_kit.py                every case, twice, then the report and the zip
#   python engine-kit\run_engine_kit.py --only skintokens:mine_rodin --runs a
#   python engine-kit\run_engine_kit.py --report-only  the report and the zip from what has run
#   python engine-kit\run_engine_kit.py --kill-test    only the kill test (it also runs after all cases)
#
# The kit's folder is laid out as the app's package root: python\, engine\ and ENGINE.json, with the
# weights apart in weights\<model> (build_engine.py --root . --weights-out weights).
#
# This script loads neither Blender nor PyTorch. It starts the launcher, which starts the engine.
import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = ROOT / "python" / "tools" / "python.exe"
LAUNCHER = ROOT / "run-restricted.exe"
ENGINE = ROOT / "engine"
ENGINE_JSON = ROOT / "ENGINE.json"
WEIGHTS = ROOT / "weights"
SOURCE = ROOT / "rig-engine"
INPUT = ROOT / "input"
OUT = ROOT / "engine-kit-results"
SEED = 12345
STEP_TIMEOUT = 45 * 60
ZIP_LIMIT = 20 * 1024 * 1024

CHARACTERS = {
    "example_tira": INPUT / "example_tira.glb",
    "mine_rodin": INPUT / "mine_rodin.glb",
    "mine_trellis": INPUT / "mine_trellis.glb",
    "tira_tail": INPUT / "edited" / "example_tira_tail.fbx",  # a skeleton fixed by hand: keep it
}
MODELS = ("skintokens", "unirig")
KEEP = {"tira_tail"}

# Exactly the variables the app's launcher passes (apps/desktop/src/main/free-image/sandbox.ts):
# What Windows needs to start a program in the AppContainer (PROTOCOL.md, Environment). Without
# LOCALAPPDATA the launcher can't start anything (exit 0x4E4C5207). The engine sets the rest itself.
WINDOWS_VARIABLES = ("SystemRoot", "SystemDrive", "windir", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
                     "LOCALAPPDATA")


def windows_environment():
    env = {}
    for name in WINDOWS_VARIABLES:
        value = os.environ.get(name) or os.environ.get(name.upper())
        if value:
            env[name] = value
    return env


_VALIDATOR = []


def schema_errors(event):
    """The engine's own protocol.schema.json, checked with the jsonschema package in the kit's Python."""
    if not _VALIDATOR:
        try:
            from jsonschema import Draft202012Validator

            schema = json.loads((ENGINE / "protocol.schema.json").read_text(encoding="utf-8"))
            _VALIDATOR.append(Draft202012Validator(schema))
        except Exception as error:
            _VALIDATOR.append(None)
            print(f"  (no schema check: {error})", flush=True)
    if _VALIDATOR[0] is None:
        return []
    return [e.message[:200] for e in _VALIDATOR[0].iter_errors(event)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def smi(query):
    try:
        done = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "--id=0"],
                              capture_output=True, text=True, timeout=10)
        return done.stdout.strip().splitlines()[0]
    except Exception:
        return None


class CardWatch:
    """The whole card's memory, sampled twice a second while a step runs (as the kit's run_native.py)."""

    def __init__(self):
        self.before = self._used()
        self.peak = self.before
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _used():
        try:
            return float(smi("memory.used"))
        except (TypeError, ValueError):
            return None

    def _loop(self):
        while not self._stop.wait(0.5):
            used = self._used()
            if used is not None and (self.peak is None or used > self.peak):
                self.peak = used

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=15)

    def extra_gb(self):
        if self.before is None or self.peak is None:
            return None
        return round((self.peak - self.before) / 1024, 2)


def imported_top_level(stderr):
    """Every module the process imported, from -X importtime's lines on stderr: a check of the
    engine's own guard that doesn't trust the engine."""
    names = set()
    for line in stderr.splitlines():
        if line.startswith("import time:") and "|" in line:
            module = line.rsplit("|", 1)[-1].strip()
            if module and not module.startswith("imported"):
                names.add(module.split(".")[0])
    return names


BLENDER = {"bpy", "_bpy", "bpy_extras", "mathutils", "bmesh"}
TORCH = {"torch", "lightning", "transformers", "spconv", "torch_scatter", "torch_cluster", "triton"}


def launch(args, reads, writes, sandbox):
    engine_command = [str(PYTHON), "-I", "-B", "-u", "-X", "utf8", "-X", "importtime",
                      str(ENGINE / "rigengine.py"), *map(str, args)]
    if not sandbox:
        return engine_command
    grants = []
    for folder in reads:
        grants += ["--read", str(folder)]
    for folder in writes:
        grants += ["--write", str(folder)]
    return [str(LAUNCHER), *grants, "--", *engine_command]


def run_step(label, args, reads, writes, logs, sandbox, on_line=None):
    """One engine step. Returns a record for the report. stdout is read line by line as it comes, to time
    the launch (start to the `starting` line, which includes the launcher granting the folders) and the
    longest silence between lines (the heartbeat promises at most 5 seconds)."""
    command = launch(args, reads, writes, sandbox)
    logs.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    lines, errors = [], []
    with CardWatch() as watch:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=windows_environment(), cwd=str(writes[0]) if writes else None)
        timer = threading.Timer(STEP_TIMEOUT, process.kill)
        timer.start()
        reader = threading.Thread(target=lambda: errors.append(process.stderr.read()), daemon=True)
        reader.start()
        try:
            for raw in iter(process.stdout.readline, b""):
                lines.append((time.monotonic() - started, raw.decode("utf-8", "replace").rstrip("\r\n")))
                if on_line is not None:
                    on_line(lines[-1][1], process)
        except ValueError:  # on_line closed the pipe after killing the launcher
            pass
        code = process.wait()
        timer.cancel()
        reader.join(timeout=60)
        if not timer.is_alive() and time.monotonic() - started >= STEP_TIMEOUT:
            code = "timed out"
    seconds = round(time.monotonic() - started, 1)
    stdout = "\n".join(text for _, text in lines) + "\n"
    stderr = (errors[0] if errors else b"").decode("utf-8", "replace")
    imported = imported_top_level(stderr)
    (logs / f"{label}.stdout.txt").write_text(stdout, encoding="utf-8")
    # the log without -X importtime's thousands of lines; the modules it named are listed apart
    (logs / f"{label}.stderr.txt").write_text(
        "\n".join(line for line in stderr.splitlines() if not line.startswith("import time:")) + "\n", encoding="utf-8")
    (logs / f"{label}.imported.txt").write_text("\n".join(sorted(imported)) + "\n", encoding="utf-8")
    events, not_json, invalid = [], [], []
    for _, line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            not_json.append(line[:300])
            continue
        events.append(event)
        invalid += [f"{event.get('type')}/{event.get('stage', event.get('code', ''))}: {e}" for e in schema_errors(event)]
    times = [when for when, line in lines if line.strip()]
    gaps = [b - a for a, b in zip(times, times[1:])]
    last = events[-1] if events else {}
    step = args[0]
    forbidden = sorted(imported & (TORCH if step in ("read", "write") else BLENDER))
    record = {
        "label": label, "step": step, "exit": code if isinstance(code, str) else (code if code < 2**31 else hex(code)),
        "ok": code == 0 and last.get("type") == "result", "seconds": seconds, "engine_seconds": last.get("seconds"),
        "launch_seconds": round(times[0], 2) if times else None,
        "first_line_is_starting": bool(events) and events[0].get("stage") == "starting",
        "longest_silence_seconds": round(max(gaps), 2) if gaps else None,
        "card_extra_gb": watch.extra_gb(), "stdout_lines_not_json": not_json, "schema_problems": invalid[:20],
        "forbidden_imported": forbidden, "forbidden_reported": last.get("forbidden_modules_loaded"),
        "last": last, "card_line": next((e for e in events if e.get("stage") == "card" and not e.get("heartbeat")),
                                         None),
    }
    if not record["ok"]:
        record["error"] = last.get("code") or f"exit {record['exit']} with no error line; see the logs"
    flag = "ok" if record["ok"] else f"FAILED ({record['error']})"
    print(f"    {label:28s} {flag:24s} {seconds:7.1f} s  (launch {record['launch_seconds']} s)", flush=True)
    return record


def run_case(model, character, run, sandbox):
    """read, rig, write (UniRig: twice) for one character, in a fresh work folder."""
    source = CHARACTERS[character]
    keep = character in KEEP
    tag = f"{model}-{character}-{run}"
    work = OUT / "work" / tag
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cache = OUT / "cuda-cache"
    cache.mkdir(parents=True, exist_ok=True)
    logs = OUT / "logs" / tag
    weights = WEIGHTS / model
    steps = []

    def go(label, args, with_input=True, with_cache=False):
        # As the app grants them: the package root (python\ and engine\), the model file's folder for read
        # and write, the model's weights folder for rig; the work folder, and the cache folder for rig.
        reads = [ROOT] + ([source.parent] if with_input else []) + ([weights] if with_cache else [])
        record = run_step(label, args, reads, [work, cache] if with_cache else [work], logs, sandbox)
        steps.append(record)
        return record["ok"]

    final = work / "rigged.glb"
    seed = ["--seed", SEED]
    if model == "skintokens":
        keep_flag = ["--keep-skeleton"] if keep else []
        _ = (go("read", ["read", "--model", model, "--input", source, "--output", work / "a.read.npz", "--work", work])
             and go("rig", ["rig", "--model", model, "--input", work / "a.read.npz", "--output", work / "a.rig.npz",
                            "--work", work, "--weights", weights, "--cache", cache, *seed, *keep_flag], with_input=False, with_cache=True)
             and go("write", ["write", "--rig", work / "a.rig.npz", "--input", source, "--output", final,
                              "--work", work]))
    elif keep:
        _ = (go("read", ["read", "--model", model, "--input", source, "--output", work / "b.read.npz", "--work", work])
             and go("rig_skin", ["rig", "--model", model, "--stage", "skin", "--keep-skeleton", "--input",
                                 work / "b.read.npz", "--output", work / "b.rig.npz", "--work", work,
                                 "--weights", weights, "--cache", cache, *seed], with_input=False, with_cache=True)
             and go("write", ["write", "--rig", work / "b.rig.npz", "--input", source, "--output", final,
                              "--work", work]))
    else:
        skeleton = work / "skeleton.fbx"
        _ = (go("read", ["read", "--model", model, "--input", source, "--output", work / "a.read.npz", "--work", work])
             and go("rig_skeleton", ["rig", "--model", model, "--stage", "skeleton", "--input", work / "a.read.npz",
                                     "--output", work / "a.rig.npz", "--work", work, "--weights", weights, "--cache", cache, *seed],
                    with_input=False, with_cache=True)
             and go("write_skeleton", ["write", "--rig", work / "a.rig.npz", "--output", skeleton, "--work", work],
                    with_input=False)
             and go("read_skeleton", ["read", "--model", model, "--input", skeleton, "--output", work / "b.read.npz",
                                      "--work", work], with_input=False)
             and go("rig_skin", ["rig", "--model", model, "--stage", "skin", "--input", work / "b.read.npz",
                                 "--output", work / "b.rig.npz", "--work", work, "--weights", weights, "--cache", cache, *seed],
                    with_input=False, with_cache=True)
             and go("write", ["write", "--rig", work / "b.rig.npz", "--input", source, "--output", final,
                              "--work", work]))
    case = {"model": model, "character": character, "run": run, "keep_skeleton": keep, "seed": SEED,
            "ok": all(s["ok"] for s in steps) and final.exists(), "steps": steps}
    if final.exists():
        case["rigged"] = {"path": str(final.relative_to(OUT)), "bytes": final.stat().st_size, "sha256": sha256(final)}
    rigs = [s for s in steps if s["step"] == "rig" and s["ok"]]
    if rigs:
        last = rigs[-1]["last"]
        case.update(bones=last.get("bones"), names=last.get("names"), skeleton=last.get("skeleton"),
                    standard_names=last.get("standard_names"), keep_check=last.get("keep_check"),
                    fingerprints={s["label"]: s["last"].get("fingerprints") for s in rigs},
                    memory={s["label"]: s["last"].get("memory") for s in rigs})
    return case


# ---- comparisons ----

def skeleton_arrays(case):
    """UniRig's skeleton as the second round read it back from the skeleton .fbx (the b read file)."""
    import numpy as np

    path = OUT / "work" / f"{case['model']}-{case['character']}-{case['run']}" / "b.read.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {"names": data["names"].tolist(), "parents": data["parents"].tolist(),
                "joints": data["joints"].astype(np.float64)}


def compare(cases, expected):
    import numpy as np

    by_key = {(c["model"], c["character"], c["run"]): c for c in cases}
    rows = []
    for (model, character, run), case in sorted(by_key.items()):
        if run != "a":
            continue
        row = {"model": model, "character": character, "ok": case["ok"], "bones": case.get("bones")}
        other = by_key.get((model, character, "b"))
        if other is not None and case.get("fingerprints") and other.get("fingerprints"):
            row["same_seed_same_fingerprints"] = case["fingerprints"] == other["fingerprints"]
            row["same_seed_same_file"] = (case.get("rigged") or {}).get("sha256") == (other.get("rigged") or {}).get("sha256")
            if not row["same_seed_same_fingerprints"]:
                row["fingerprints_differing"] = {
                    stage: sorted(k for k in (case["fingerprints"].get(stage) or {})
                                  if (case["fingerprints"].get(stage) or {}).get(k) != (other["fingerprints"].get(stage) or {}).get(k))
                    for stage in case["fingerprints"]}
        want = expected.get(model, {}).get(character)
        if want and model == "skintokens" and case.get("fingerprints"):
            got = dict(case["fingerprints"].get("rig") or {}, bones=case.get("bones"))
            row["vs_earlier_windows"] = {k: ("same" if got.get(k) == want[k] else f"DIFFERENT ({got.get(k)} vs {want[k]})")
                                         for k in ("bones", "points", "tokens", "joints", "skin")}
        if want and model == "unirig" and "joints" in want:
            got = skeleton_arrays(case)
            if got is not None:
                same_shape = len(got["names"]) == len(want["names"])
                row["vs_earlier_windows"] = {
                    "bones": f"{len(got['names'])} vs {want['bones']}",
                    "names_equal": got["names"] == want["names"], "parents_equal": got["parents"] == want["parents"],
                    "largest_joint_move": float(np.abs(got["joints"] - np.asarray(want["joints"])).max()) if same_shape else None,
                }
        if want and "names_kept" in want:
            row["names_kept"] = case.get("names") == "kept"
            row["names_kept_expected"] = want["names_kept"]
        row["standard_names"] = case.get("standard_names")
        rows.append(row)
    return rows


def guard_summary(cases):
    problems = []
    for case in cases:
        for step in case["steps"]:
            if step["forbidden_imported"] or (step["ok"] and step["forbidden_reported"] != []):
                problems.append(f"{case['model']} {case['character']} {case['run']} {step['label']}: "
                                f"imported {step['forbidden_imported']}, reported {step['forbidden_reported']}")
            if step["stdout_lines_not_json"]:
                problems.append(f"{case['model']} {case['character']} {case['run']} {step['label']}: "
                                f"{len(step['stdout_lines_not_json'])} stdout lines that are not JSON")
            if step.get("schema_problems"):
                problems.append(f"{case['model']} {case['character']} {case['run']} {step['label']}: lines that "
                                f"don't match protocol.schema.json: {step['schema_problems'][:3]}")
            if step.get("first_line_is_starting") is False:
                problems.append(f"{case['model']} {case['character']} {case['run']} {step['label']}: "
                                "the first line was not `starting`")
            if (step.get("longest_silence_seconds") or 0) > 5:
                problems.append(f"{case['model']} {case['character']} {case['run']} {step['label']}: "
                                f"{step['longest_silence_seconds']} s without a line (the protocol promises 5)")
    return problems


# ---- probes, the kill test, path lengths ----

def run_probes(sandbox):
    work = OUT / "work" / "probe"
    work.mkdir(parents=True, exist_ok=True)
    logs = OUT / "logs" / "probe"
    found = {}
    found["light"] = run_step("probe_light", ["probe", "--light", "--work", work], [ROOT], [work], logs, sandbox)
    weights = [f"{m}={WEIGHTS / m}" for m in MODELS]
    found["full"] = run_step("probe_full", ["probe", "--work", work, "--verify-weights",
                                            *[x for w in weights for x in ("--weights", w)]],
                             [ROOT, *[WEIGHTS / m for m in MODELS]], [work], logs, sandbox)
    return found


def process_left(marker):
    """Engine processes still running (their command line names the engine's entry and the marker)."""
    command = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
               "Where-Object { $_.CommandLine -like '*" + marker + "*' } | Select-Object -ExpandProperty ProcessId")
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True,
                              timeout=60)
        return [int(x) for x in done.stdout.split() if x.strip().isdigit()]
    except Exception as error:
        return [f"could not list processes: {error}"]


def kill_test(sandbox):
    """Starts SkinTokens' rig step on mine_rodin through the launcher, kills the launcher (as the app
    cancels) once generating has begun, and checks that no engine process is left, that the card's
    memory comes back, and what the work and cache folders hold afterwards."""
    print("== kill test", flush=True)
    work = OUT / "work" / "kill-test"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cache = OUT / "cuda-cache"
    cache.mkdir(parents=True, exist_ok=True)
    logs = OUT / "logs" / "kill-test"
    source = CHARACTERS["mine_rodin"]
    read = run_step("read", ["read", "--model", "skintokens", "--input", source, "--output", work / "a.read.npz",
                             "--work", work], [ROOT, source.parent], [work], logs, sandbox)
    report = {"read_ok": read["ok"], "memory_before_mb": CardWatch._used()}
    if not read["ok"]:
        return report
    killed = {}

    def on_line(line, process):
        if not killed and '"stage":"generating"' in line:
            time.sleep(3)  # well inside generation
            killed["memory_generating_mb"] = CardWatch._used()
            killed["at"] = time.monotonic()
            process.kill()  # the launcher: its job object ends the engine with it
            process.stdout.close()  # stop reading; an engine left alive would hold the pipe open

    rig = run_step("rig", ["rig", "--model", "skintokens", "--input", work / "a.read.npz", "--output",
                           work / "a.rig.npz", "--work", work, "--weights", WEIGHTS / "skintokens", "--cache", cache,
                           "--seed", SEED], [ROOT, WEIGHTS / "skintokens"], [work, cache], logs, sandbox, on_line)
    report.update(killed_during_generating=bool(killed), memory_generating_mb=killed.get("memory_generating_mb"),
                  rig_exit=rig["exit"])
    if not killed:
        return report
    freed_after = None
    for _ in range(30):
        used = CardWatch._used()
        if used is not None and report["memory_before_mb"] is not None and used <= report["memory_before_mb"] + 300:
            freed_after = round(time.monotonic() - killed["at"], 1)
            break
        time.sleep(0.5)
    report.update(memory_after_mb=CardWatch._used(), memory_back_within_seconds=freed_after,
                  processes_left=process_left(str(work / "a.rig.npz")),
                  left_in_work=sorted(p.name for p in work.iterdir()),
                  cache_markers=sorted(p.name for p in cache.glob(".in-use-*")))
    return report


def longest_relative(folder):
    best = (0, "")
    for path in folder.rglob("*"):
        rel = path.relative_to(folder.parent).as_posix()
        if len(rel) > best[0]:
            best = (len(rel), rel)
    return {"characters": best[0], "path": best[1]}


def write_report(report):
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Rig engine on Windows, in the sandbox", "",
             f"- Card: {report.get('card')}  Driver: {report.get('driver')}",
             f"- Smart App Control: {report.get('smart_app_control')}",
             f"- Longest paths below the root: engine {report['engine'].get('longest_relative_path')} characters, "
             f"python {report.get('python_longest_path')}", ""]
    for kind, probe in (report.get("probes") or {}).items():
        lines.append(f"- Probe {kind}: {probe.get('seconds')} s (launch {probe.get('launch_seconds')} s), card extra "
                     f"{probe.get('card_extra_gb')} GB: {json.dumps(probe.get('last', {}), default=str)[:1500]}")
    lines += ["", f"- Kill test: {json.dumps(report.get('kill_test'), default=str)}", "",
              "## Steps", "", "| Case | Run | Step | Result | Seconds | Launch s | Longest silence s | Peak allocated GB "
              "| Card extra GB | Bones | Names |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for case in report["cases"]:
        for step in case["steps"]:
            last = step["last"]
            peak = (last.get("memory") or {}).get("peak_allocated_bytes")
            lines.append(f"| {case['model']} {case['character']} | {case['run']} | {step['label']} | "
                         f"{'ok' if step['ok'] else 'FAILED ' + str(step.get('error'))} | {step['seconds']} | "
                         f"{step.get('launch_seconds')} | {step.get('longest_silence_seconds')} | "
                         f"{'' if not peak else round(peak / 2**30, 2)} | {step['card_extra_gb'] or ''} | "
                         f"{last.get('bones', '')} | {last.get('names', '')} |")
    lines += ["", "## Comparisons", ""]
    for row in report["comparisons"]:
        lines.append(f"- **{row['model']} {row['character']}**: " + json.dumps({k: v for k, v in row.items()
                                                                                  if k not in ('model', 'character')}))
    lines += ["", "## Blender and PyTorch kept apart", ""]
    lines += [f"- {p}" for p in report["guard_problems"]] or ["- Every step: nothing forbidden imported "
                                                               "(by -X importtime) or reported, stdout only JSON."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---- snapshot, diffs, zip ----

SNAPSHOT_FOLDERS = ("rig-engine", "engine-kit")


def snapshot():
    target = OUT / "pristine"
    if target.exists():
        print(f"a snapshot exists already: {target}")
        return
    for name in SNAPSHOT_FOLDERS:
        shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
    print(f"snapshot taken: {target}")


def write_diffs():
    pristine = OUT / "pristine"
    diffs = OUT / "diffs"
    if diffs.exists():
        shutil.rmtree(diffs)
    diffs.mkdir(parents=True)
    if not pristine.exists():
        (diffs / "NO_SNAPSHOT.txt").write_text("run_engine_kit.py --snapshot was not run first\n", encoding="utf-8")
        return 0
    count = 0
    for name in SNAPSHOT_FOLDERS:
        before_root, after_root = pristine / name, ROOT / name
        paths = {p.relative_to(before_root) for p in before_root.rglob("*") if p.is_file()}
        paths |= {p.relative_to(after_root) for p in after_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        for rel in sorted(paths):
            a, b = before_root / rel, after_root / rel
            old = a.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if a.exists() else []
            new = b.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if b.exists() else []
            if old != new:
                text = "".join(difflib.unified_diff(old, new, f"a/{name}/{rel.as_posix()}", f"b/{name}/{rel.as_posix()}"))
                (diffs / (f"{name}-" + rel.as_posix().replace("/", "_") + ".diff")).write_text(text, encoding="utf-8")
                count += 1
    return count


def make_zip():
    target = ROOT / "engine-kit-results.zip"
    skipped = []
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if not path.is_file() or "pristine" in path.relative_to(OUT).parts:
                continue
            rel = path.relative_to(OUT)
            if rel.parts[0] == "work" and path.name != "rigged.glb":
                continue  # read and rig files are rebuilt from the logs if needed
            if rel.parts[0] == "cuda-cache":
                continue
            if path.name == "rigged.glb":
                if not rel.parts[1].endswith("-a"):
                    skipped.append(f"{rel} (run b; run a is in)")
                    continue
                if target.stat().st_size + path.stat().st_size > ZIP_LIMIT - 512 * 1024:
                    skipped.append(f"{rel} ({path.stat().st_size // 1024} KB: the zip would pass 20 MB)")
                    continue
            archive.write(path, "engine-kit-results/" + rel.as_posix())
        notes = "\n".join(["Left out of the zip:"] + skipped) + "\n"
        archive.writestr("engine-kit-results/LEFT_OUT.txt", notes)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="model:character pairs, comma separated (default: all)")
    parser.add_argument("--runs", default="a,b", help="a, b, or a,b (two runs with the same seed)")
    parser.add_argument("--no-sandbox", action="store_true", help="for finding a bug only: run without the launcher")
    parser.add_argument("--snapshot", action="store_true", help="keep copies of rig-engine and engine-kit for diffs")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--kill-test", action="store_true", help="only the kill test, then the report")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.snapshot:
        snapshot()
        return 0
    for needed in (PYTHON, LAUNCHER, ENGINE / "rigengine.py", ENGINE_JSON, WEIGHTS / "skintokens", WEIGHTS / "unirig"):
        if not needed.exists():
            raise SystemExit(f"STOP. {needed} is missing. See engine-kit\\README.md.")
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    state = OUT / "cases.json"
    cases = json.loads(state.read_text(encoding="utf-8")) if state.exists() else []
    if args.kill_test:
        (OUT / "kill_test.json").write_text(json.dumps(kill_test(not args.no_sandbox), indent=2), encoding="utf-8")
    elif not args.report_only:
        wanted = [(m, c) for m in MODELS for c in CHARACTERS]
        if args.only:
            wanted = [tuple(pair.split(":")) for pair in args.only.split(",")]
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]
        print("probes ...", flush=True)
        (OUT / "probe.json").write_text(json.dumps(run_probes(not args.no_sandbox), indent=2), encoding="utf-8")
        for model, character in wanted:
            for run in runs:
                print(f"== {model} {character} run {run}", flush=True)
                case = run_case(model, character, run, not args.no_sandbox)
                case["sandbox"] = not args.no_sandbox
                cases = [c for c in cases if (c["model"], c["character"], c["run"]) != (model, character, run)]
                cases.append(case)
                state.write_text(json.dumps(cases, indent=2), encoding="utf-8")
        if not args.only:
            (OUT / "kill_test.json").write_text(json.dumps(kill_test(not args.no_sandbox), indent=2),
                                                encoding="utf-8")
    probe = json.loads((OUT / "probe.json").read_text(encoding="utf-8")) if (OUT / "probe.json").exists() else {}
    sac = None
    try:
        sac = subprocess.run(["reg", "query", r"HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy", "/v",
                              "VerifiedAndReputablePolicyState"], capture_output=True, text=True).stdout.strip()
    except Exception as error:
        sac = f"could not read: {error}"
    report = {"card": f"{smi('name')} with {smi('memory.total')} MB", "driver": smi("driver_version"),
              "smart_app_control": sac, "engine": json.loads(ENGINE_JSON.read_text(encoding="utf-8")),
              "python_longest_path": longest_relative(ROOT / "python") if (ROOT / "python").is_dir() else None,
              "probes": probe, "kill_test": json.loads((OUT / "kill_test.json").read_text(encoding="utf-8"))
              if (OUT / "kill_test.json").exists() else None,
              "cases": cases, "comparisons": compare(cases, expected),
              "guard_problems": guard_summary(cases)}
    write_report(report)
    changed = write_diffs()
    target = make_zip()
    failed = [f"{c['model']} {c['character']} {c['run']}" for c in cases if not c["ok"]]
    print(f"done: {len(cases) - len(failed)} of {len(cases)} cases ok; {changed} changed files as diffs; "
          f"report {OUT / 'report.md'}; zip {target} ({target.stat().st_size // 1024} KB)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
