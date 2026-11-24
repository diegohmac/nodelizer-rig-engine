# Rig engine

A local auto rigging engine for Nodelizer's desktop app, for Windows with an NVIDIA card. It puts a skeleton and skin weights into a 3D character with one of two models by VAST AI Research: SkinTokens (the default) or UniRig. The app runs it as a separate program, through its sandbox launcher, and talks to it as `PROTOCOL.md` describes.

## How it works

Blender (as the `bpy` module, GPL) and PyTorch with NVIDIA's CUDA libraries are never loaded in the same process. Each rig is a few steps, each its own process, started one after another by the app, passing plain files in one work folder:

- `read` loads Blender: the model file in, plain arrays out (the mesh, and the file's skeleton if it has one).
- `rig` loads PyTorch: those arrays in, the rig out (joints, parents, bone names, skin weights).
- `write` loads Blender: the original file and the rig in, a rigged `.glb` out, with the original mesh, materials and textures.

Each step blocks the other side's modules at start and checks again at the end (`engine/guard.py`), and a test walks every import from each step's entry to prove it statically (`tests/test_static_imports.py`). UniRig needs Blender twice in the middle of its run, so it takes two rounds of the three steps (`PROTOCOL.md`, Sequences).

The engine drives the models' own code, so the results are what the models' own pipelines give. Where the split needed it, our drivers stand in for the parts that crossed it: SkinTokens' dataset gets its Asset from the read file instead of Blender, and UniRig's Blender exports are recorded in the rig step and run in the write step. The models' code is changed only by checked text patches (`PATCHES.md`).

Nothing passes between steps as pickle: files are `.npz` with `allow_pickle=False` and JSON. Weights load only with `torch.load(weights_only=True)`. The engine never uses the network, writes only into the work folder, starts no other process and opens no socket, as the launcher requires (one process, no network, only the granted folders).

## Rules that aren't obvious from the code

- **One model per process.** Both models call their package `src`; `engine/modelpath.py` puts one model's folder on the path and refuses the other.
- **Same seed, same rig.** SkinTokens is seeded right before its job and its one fetch is seeded as a data loader worker would be, since the sandbox allows no worker process (`engine/seeding.py`). The write step seeds the export with the rig file's seed.
- **The rig step runs in the model's folder.** Both models name their configs relative to it, and their checkpoints too; the drivers take the checkpoints from the weights folder instead. The engine folder may be read only; everything written goes to the work folder.
- **The app passes only Windows' own five variables.** The engine sets every folder and flag it needs itself, from `--work` and `--cache` (`engine/workspace.py`).
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
| `engine/weights.py` | Each model's weight files, their SHA-256, and the free video memory each model needs. |
| `protocol.schema.json` | A JSON Schema of every line the engine prints. |
| `engine/names.py` | Standard (Mixamo) bone names by parent structure. |
| `standin/flash_attn/` | PyTorch code standing in for flash-attn, which has no Windows build. |
| `build/` | `sources.py` (pinned inputs), `patches.py` (every change to the models' code), `build_engine.py` (makes an engine folder). |
| `tests/`, `test.sh` | The test set, run in Docker. |
| `windows-kit/` | The overlay for a test run on the Windows kit computer, and `make_kit.py`, which zips it. |

## Building

The app installs one root folder holding `python\`, `engine\` and `ENGINE.json`, and apart from it one weights folder per model (PROTOCOL.md, Packaged layout and Weights).

```
python build/build_engine.py --downloads <folder with the pinned files> --root <package root> [--weights-out <folder>] [--fetch]
```

The downloads folder holds the files named in `build/sources.py`; the Windows kit's `kit\fetch.py` leaves exactly these in its `downloads\` folder. `--fetch` downloads what is missing, each file checked by SHA-256. `--root` makes `<root>\engine` (never any weights) and `<root>\ENGINE.json`: it unpacks the two code archives, applies every patch, leaves out what nothing uses (`PATCHES.md`), places the two config files, copies the engine's own code, and fails if any path below the root would pass 160 characters. `--weights-out` lays out `<folder>\skintokens` and `<folder>\unirig` from the four weight files (about 3 GB). The release places `python\` in the root.

## Testing

`./test.sh` runs everything in Docker (an x86_64 Linux image, emulated on an ARM Mac, with bpy 4.2.0 and PyTorch for the CPU) and exits non zero on any failure. It builds a package root from the pinned archives with no network, in a folder with spaces and accents in its path, runs the tests with no network (every line any step prints is checked against `protocol.schema.json`), and makes `dist/rig-engine-kit.zip`. The reference data (the kit's input characters and the earlier rounds' results) comes from the folders in `tests/data.env` (see `tests/data.env.example`).

The rig step itself needs an NVIDIA card and the weights, so it isn't run here: read and write are tested for real with Blender, using rigs the models made on the Windows computer, and everything up to the model (options, files, guards, seeding) is tested without one. The rig step is tested on Windows with `windows-kit/`.
