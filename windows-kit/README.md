# Rig engine on Windows: test kit overlay

The earlier kits ran SkinTokens and UniRig on this computer and answered what an engine has to handle. This overlay brings the engine itself: the same two models, split so that Blender and PyTorch never share a process, every step started through the app's sandbox launcher. The job is to run it, fix what breaks, and send back one zip.

The overlay holds scripts and text only. It is unzipped on top of the earlier kit's folder (the signed one: `python\`, `kit\fetch.py`, `input\`, `run-restricted.exe`, `block-network.ps1`).

| Path | What it is |
| --- | --- |
| `rig-engine\` | The engine's source: `rigengine.py`, `engine\`, `standin\`, `build\`, and its documents (`PROTOCOL.md` says how it works). |
| `engine-kit\run_engine_kit.py` | Runs every case through the launcher and writes the report and the zip. |
| `engine-kit\expected.json` | Results of the earlier Windows round, to compare with. |

After step 2 the kit's folder is laid out as the app will install the engine: the root holds `python\`, `engine\` and `ENGINE.json`, and the weights sit apart in `weights\skintokens` and `weights\unirig`.

## Limits

- **Smart App Control stays as it is.** Don't turn it off or change its mode. If Windows blocks a file, that is a finding: note the file and the event (Event Viewer, Applications and Services Logs, Microsoft, Windows, CodeIntegrity, Operational) and go on.
- **No pip installs, no new program files.** Anything pip installs is unsigned and would be blocked. Everything the engine needs is already in `python\`. Nothing in this overlay is a program file.
- **The models run only with the network blocked** (step 3), and only through `run-restricted.exe`.
- **Only the pinned downloads**, the ones in `kit\fetch.py`.
- **Weights load only with PyTorch's restricted loader.** Don't switch it off to get past an error.
- **Every change is recorded.** Run step 4's snapshot before changing anything; the script then turns every change into a diff.

## Steps

Open PowerShell in the kit's folder (a short path with no spaces, for example `C:\rig-native`).

### 1. Fetch code and weights

```powershell
.\python\tools\python.exe kit\fetch.py
```

It ends with "fetch done" and leaves the pinned files in `downloads\`. If the folder already holds them from the earlier round, it only checks them.

### 2. Build the engine and the weights folders

```powershell
.\python\tools\python.exe rig-engine\build\build_engine.py --downloads downloads --root . --weights-out weights
```

It checks every file's SHA-256, unpacks both models' code into `engine\`, applies every patch (each must find its text exactly once, or it stops), places the two config files, copies the engine's code, checks that no path below the folder passes 160 characters, and writes `ENGINE.json`. Then it lays out `weights\skintokens` and `weights\unirig` (hard links, so no extra 3 GB). It ends with "build done" and two "weights" lines. To build again after a fix, delete `engine\` and `ENGINE.json` first (the weights folders can stay).

### 3. Block the network

In an **administrator** PowerShell, in this folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\block-network.ps1
```

Then, back in the normal PowerShell, `.\python\tools\python.exe kit\check.py offline` must show every internet address blocked.

### 4. Snapshot, then run

```powershell
.\python\tools\python.exe engine-kit\run_engine_kit.py --snapshot
.\python\tools\python.exe engine-kit\run_engine_kit.py
```

The snapshot keeps copies of `rig-engine\` and `engine-kit\` as they arrived. The run probes the card twice (`probe --light`, then a full probe that also reads every weight file and checks its SHA-256), then rigs, with seed 12345, each of `example_tira`, `mine_rodin`, `mine_trellis` (a skeleton is made) and `input\edited\example_tira_tail.fbx` (its own skeleton is kept), with both models, twice. Each `read`, `rig` and `write` is its own process inside `run-restricted.exe`, granted what the app will grant: the root (which here is the whole kit folder) and the character's folder or the model's weights folder (read), the work folder and, for `rig`, a CUDA cache folder (write). It passes the engine exactly the five Windows variables the app's launcher passes (`SystemRoot`, `SystemDrive`, `windir`, `NUMBER_OF_PROCESSORS`, `PROCESSOR_ARCHITECTURE`) and nothing else. At the end a kill test starts SkinTokens' `rig` on mine_rodin, kills the launcher once generating has begun (as the app cancels), and checks that no engine process is left and the card's memory comes back. UniRig takes six steps where SkinTokens takes three (`rig-engine\PROTOCOL.md`, Sequences). Expect 30 to 60 minutes.

Results go to `engine-kit-results\`: `report.md` and `report.json`, a log per step in `logs\`, and the rigged files in `work\`. The report gives, per step, ok or the error code, seconds, peak video memory (PyTorch's own count and the whole card's), bones, and the names outcome; and per character:

- whether the two runs with the same seed gave the same fingerprints (points, tokens, joints, skin) and the same file;
- SkinTokens: the fingerprints against the earlier Windows round (`expected.json`); example_tira should give `f7e9bd28874e`, `12317e1749ed`, `d5e5f385049c`, `57da8e62d712`;
- UniRig: the skeleton against the earlier round's (names, parents, largest joint move);
- whether standard names matched, and whether the kept skeleton's names were kept;
- per step, the launch time (from starting the launcher to the engine's first line, `starting`, which includes the launcher granting the folders) and the longest silence between lines (the heartbeat promises at most 5 seconds);
- every line checked against the engine's `protocol.schema.json`;
- the probes' times and the card memory they took, the weight files' sizes and SHA-256, and the longest path below the root in `python\` and `engine\`;
- the kill test: processes left, card memory before, while generating and after, and what the work and cache folders hold;
- that Blender was never loaded in a `rig` step and PyTorch never in `read` or `write`, checked twice: by the engine's own report, and by Python's `-X importtime` list of everything each process imported.

To run part of it: `--only skintokens:mine_rodin,unirig:tira_tail` and `--runs a` (no kill test then); `--kill-test` runs only the kill test. `--report-only` rebuilds the report and the zip from what has run.

## When something breaks

Most likely places: a path or permission the sandbox refuses (the log says which file), a DLL that won't load (`ImportError: DLL load failed`), a Windows path detail in the engine's code, or a difference between the models' code paths here and in the earlier kit's `st_rig.py` and `run_native.py`.

1. Read the step's log in `engine-kit-results\logs\<case>\`.
2. Find the smallest fix, in `rig-engine\` (then delete `engine\` and build again) or in `engine-kit\`.
3. Run that case again with `--only`.

`--no-sandbox` runs the steps without the launcher, to tell a sandbox problem from an engine problem; report which it was. Don't leave fixes that only work without the sandbox.

Work through failures without waiting, about an hour per problem; then note where it stands and go on.

## What to send back

One zip, `engine-kit-results.zip`, under 20 MB. The script makes it at the end of every run: the report, the logs, a diff per changed file (`diffs\`), and the rigged files of the first run that fit. Add a short `NOTES.md` to `engine-kit-results\` first (then run `--report-only` to zip it in):

- per character and model, ok or not, in a line;
- every fix, and why it was needed;
- anything that failed and wasn't fixed;
- the Smart App Control value, and any Windows prompt or block seen.

## Cleaning up

`unblock-network.ps1` as administrator. The launcher's folder grants stay on the folders and its AppContainer profile `Nodelizer.ImageEngine` stays under `%LOCALAPPDATA%\Packages`; both go when the folder and the profile are deleted.
