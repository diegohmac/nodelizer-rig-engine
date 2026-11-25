# Rig engine

A local auto rigging engine for Nodelizer's desktop app, for Windows with an NVIDIA card and for Macs with Apple silicon. It puts a skeleton and skin weights into a 3D character with one of two models by VAST AI Research: SkinTokens (the default) or UniRig. UniRig needs an NVIDIA card, so a Mac has SkinTokens only, run on the processor. The app runs it as a separate program, through its sandbox (a launcher on Windows, `sandbox-exec` on a Mac), and talks to it as `PROTOCOL.md` describes.

## How it works

Blender (as the `bpy` module, GPL) and PyTorch with NVIDIA's CUDA libraries are never loaded in the same process. Each rig is a few steps, each its own process, started one after another by the app, passing plain files in one work folder:

- `read` loads Blender: the model file in, plain arrays out (the mesh, and the file's skeleton if it has one).
- `rig` loads PyTorch: those arrays in, the rig out (joints, parents, bone names, skin weights).
- `write` loads Blender: the original file and the rig in, a rigged `.glb` out, with the original mesh, materials and textures.

Each step blocks the other side's modules at start and checks again at the end (`engine/guard.py`), and a test walks every import from each step's entry to prove it statically (`tests/test_static_imports.py`). UniRig needs Blender twice in the middle of its run, so it takes two rounds of the three steps (`PROTOCOL.md`, Sequences).

The engine drives the models' own code, so the results are what the models' own pipelines give. Where the split needed it, our drivers stand in for the parts that crossed it: SkinTokens' dataset gets its Asset from the read file instead of Blender, and UniRig's Blender exports are recorded in the rig step and run in the write step. The models' code is changed only by checked text patches (`PATCHES.md`).

Nothing passes between steps as pickle: files are `.npz` with `allow_pickle=False` and JSON. Weights load only with `torch.load(weights_only=True)`. The engine never uses the network, writes only into the work folder, starts no other process and opens no socket, as the launcher requires (one process, no network, only the granted folders). On a Mac below 32 GB the `rig` step starts itself again in place once, same process, to switch off an allocator cache: half the memory, about 2.5 times as long, the same rig (PROTOCOL.md, Devices).

## Rules that aren't obvious from the code

- **One model per process.** Both models call their package `src`; `engine/modelpath.py` puts one model's folder on the path and refuses the other.
- **Same seed, same rig.** SkinTokens is seeded right before its job and its one fetch is seeded as a data loader worker would be, since the sandbox allows no worker process (`engine/seeding.py`). The write step seeds the export with the rig file's seed.
- **The rig step runs in the model's folder.** Both models name their configs relative to it, and their checkpoints too; the drivers take the checkpoints from the weights folder instead. The engine folder may be read only; everything written goes to the work folder.
- **CUDA runs as the models' authors wrote them.** The processor and a Mac's GPU are handled in the drivers, never on the CUDA path (`engine/devices.py`, PROTOCOL.md, Devices), so the Windows rigs don't move.
- **The app passes only six variables Windows needs**, and none on a Mac. The engine sets every folder and flag it needs itself, from `--work` and `--cache` (`engine/workspace.py`).
- **Standard names follow the parent structure only** (`engine/names.py`). Whether the bones also sit where the names say is reported, never used to decide.
- **The entry is `rigengine.py`, never `run.py`.** The Windows kit's Python imports PyTorch at start for any script named `run.py` (UniRig's), which would break the read and write steps.

## Where things live

| Path | What |
| --- | --- |
| `rigengine.py` | The entry point: `probe`, `read`, `rig`, `write`. |
| `engine/cli.py` | Arguments, the JSON lines, error codes, the guard, a clean exit. |
| `engine/steps/` | One module per step: checks its paths and options, then calls the model's driver. |
| `engine/skintokens/`, `engine/unirig/` | The drivers: read, rig and write for each model. |
| `engine/layout.py`, `engine/plainnpz.py` | The read and rig file layouts and their checks; plain `.npz` input and output. |
| `engine/workspace.py` | The work and cache folders, refused paths, and the environment each step sets. |
| `engine/weights.py` | Each model's weight files, their SHA-256, and the free memory each model needs. |
| `engine/devices.py`, `engine/host.py` | Which device and number type a rig runs on; the computer's memory and chip off CUDA. |
| `protocol.schema.json` | A JSON Schema of every line the engine prints. |
| `engine/names.py` | Standard (Mixamo) bone names by parent structure. |
| `standin/flash_attn/` | PyTorch code standing in for flash-attn, which has no Windows build. |
| `build/` | `sources.py` (pinned inputs), `patches.py` (every change to the models' code), `build_engine.py` (makes an engine folder). |
| `build/mac/` | The Mac root: its pinned Python and package lock, and `build_mac.py`, which builds it. |
| `mac/rig-engine.sb` | The sandbox profile the app starts every Mac step with. |
| `tests/`, `test.sh` | The test set, run in Docker. |
| `test-mac.sh`, `tests/mac/` | The native test on a Mac, in its sandbox. |
| `windows-kit/` | The overlay for a test run on the Windows kit computer, and `make_kit.py`, which zips it. |

## Building

The app installs one root folder holding `python\`, `engine\` and `ENGINE.json`, and apart from it one weights folder per model (PROTOCOL.md, Packaged layout and Weights).

```
python build/build_engine.py --downloads <folder with the pinned files> --root <package root> [--weights-out <folder>] [--fetch]
```

The downloads folder holds the files named in `build/sources.py`; the Windows kit's `kit\fetch.py` leaves exactly these in its `downloads\` folder. `--fetch` downloads what is missing, each file checked by SHA-256. `--root` makes `<root>\engine` (never any weights) and `<root>\ENGINE.json`: it unpacks the two code archives, applies every patch, leaves out what nothing uses (`PATCHES.md`), places the two config files, copies the engine's own code, and fails if any path below the root would pass 160 characters. `--weights-out` lays out `<folder>\skintokens` and `<folder>\unirig` from the four weight files (about 3 GB). The release places `python\` in the root.

A Mac root is built whole, `python/` included:

```
python3 build/mac/build_mac.py --downloads <folder with the pinned files> --cache <folder> --root <new folder>
```

It fetches the pinned Python (Astral's python-build-standalone) and every locked wheel into `--cache`, each checked by SHA-256, installs the wheels with pip in Docker for macOS on Apple silicon (installing only unpacks; no package code runs), removes pip, setuptools and wheel and the parts of Python's archive no step uses (`ensurepip`, `venv`, IDLE, Tk, build files; `sources_mac.py`), then makes `engine/` with SkinTokens alone (`build_engine.py --platform darwin-arm64`). The root runs from any folder, spaces and accents included: `python/bin/python3.11 -I -B` needs no settings. `--weights-out <folder> --models skintokens` on `build_engine.py` lays out the Mac's weights. `build/mac/lock.sh` remakes the lock.

## Testing

`./test.sh` runs everything in Docker (an x86_64 Linux image, emulated on an ARM Mac, with bpy 4.2.0 and PyTorch for the CPU) and exits non zero on any failure. It builds a package root from the pinned archives with no network, in a folder with spaces and accents in its path, runs the tests with no network (every line any step prints is checked against `protocol.schema.json`), and makes `dist/rig-engine-kit.zip`. The reference data (the kit's input characters and the earlier rounds' results) comes from the folders in `tests/data.env` (see `tests/data.env.example`).

The rig step itself needs the weights (and for UniRig an NVIDIA card), so it isn't run here: read and write are tested for real with Blender, using rigs the models made on the Windows computer, and everything up to the model (options, files, guards, seeding) is tested without one. The rig step is tested on Windows with `windows-kit/`, and on a Mac with `./test-mac.sh`: it runs natively (the one place outside Docker where Blender's module, PyTorch and SkinTokens run), every step inside `sandbox-exec` with `mac/rig-engine.sb`. It needs a built Mac root and SkinTokens' weights (variables in `tests/data.env.example`), checks the root, probe, every refusal of the sandbox (the GPU included, which only Blender's steps get) and UniRig's refusal, then reads, rigs and writes one character on the Mac default; `--twice` rigs again in the other memory mode and checks the rig is the same, and `--report <file>` writes the results as JSON for the release tooling (`REPORT_SHAPE` in `tests/mac/native.py`).
