# Rig engine protocol, version 1

The rig engine rigs a 3D character with SkinTokens or UniRig. The app runs it as a separate program, one step per process, each started through the app's sandbox launcher (`run-restricted.exe` on Windows). Steps pass plain files to each other in one work folder. Blender (`bpy`) and PyTorch are never loaded in the same process: `read` and `write` load Blender, `rig` and `probe` load PyTorch. `protocol.schema.json` describes every line the engine prints; this file says what the lines mean.

## Packaged layout

One root folder holds everything that runs, so one read grant covers it. The app installs it at `%APPDATA%\Nodelizer\free-rig\engine\<version>\`.

```
<root>\
  ENGINE.json             what this root is (below)
  python\                 the Python environment; its executable is python\tools\python.exe
  engine\                 this engine: rigengine.py, engine\, standin\, models\ (both models' code), docs
```

`ENGINE.json` holds `engine_version`, `protocol` (1), `python` (`python/tools/python.exe`, relative to the root), `entry` (`engine/rigengine.py`), the two models' commits, the SHA-256 of every input the build used, the patch list, the weights each model needs (`weights`: relative path and SHA-256, as in Weights below), and `longest_relative_path` with the `path_limit` it was checked against.

No path below the root is longer than 160 characters: the build fails above that (`build/build_engine.py`, `PATH_LIMIT`). Windows' classic limit is 259 characters; the install path is 88 characters with the longest Windows user name (20) and a 15 character version, which leaves 171; 160 keeps 11 to spare for profile folders like `C:\Users\name.DOMAIN`. This engine folder's longest path is in `ENGINE.json` (85 characters for engine 0.1.0). The Python environment is placed by the release and isn't checked by this build; the Windows kit reports its longest path.

The weights are not in the root: they live in one folder per model, downloaded separately (Weights, below).

## Running a step

```
python\tools\python.exe -I -B -u -X utf8 engine\rigengine.py probe [--light] [--work W] [--weights MODEL=FOLDER ...] [--verify-weights]
python\tools\python.exe -I -B -u -X utf8 engine\rigengine.py read  --model M --input FILE --output W\a.read.npz --work W
python\tools\python.exe -I -B -u -X utf8 engine\rigengine.py rig   --model M --input W\a.read.npz --output W\a.rig.npz --work W
                                                                   --weights F --seed N [--keep-skeleton] [--standard-names on|off]
                                                                   [--stage skeleton|skin] [--cache C] [--device N]
python\tools\python.exe -I -B -u -X utf8 engine\rigengine.py write --rig W\a.rig.npz [--input FILE] --output W\a.glb --work W
```

Paths relative to the root here; every path given to the engine is absolute. `-I` keeps the person's Python settings out (the engine sets its own paths), `-B` writes no `.pyc` files into the root, `-u` sends every line at once, `-X utf8` makes paths with accents safe. The working directory doesn't matter.

## Output

- **stdout** carries JSON lines and nothing else, one object per line, ASCII only. Every line has `type` (`progress`, `result` or `error`), `step`, `protocol` (1) and `seconds` since the step began. Anything a library prints, Blender's C code included, goes to stderr.
- **The first line** of every step is a `progress` line with `stage: "starting"`, `pid` and `engine_version`, sent before anything heavy loads. No `starting` line means the process could not start (the launcher's exit code says why); a `starting` line and then nothing means it crashed.
- **The last line** is one `result` or one `error`. Nothing follows it.
- **Heartbeat.** While a stage runs, a `progress` line repeating the current `stage` with `heartbeat: true` is sent whenever nothing was sent for 2 seconds, so there is a line at least every 5 seconds. It comes from a thread and each line is written whole under a lock, so lines never mix. It needs Python's lock, so a single library call that holds that lock for long could delay it; the Windows kit reports the longest silence per step.
- **stderr** is a log for people. Never parse it.
- **Exit code** 0: the last line is a `result`. 1: the last line is an `error`. 2: an `error` about the command line (`bad_arguments`). Anything else, or no final line, is a crash.

## Stages and how long they take

Times from the Windows rounds on an RTX 5090 with the whole card free; "longest expected" allows for a slow disk, a busy computer and a card short of memory (which can make generating four or five times slower). Kill a step that sends no line for 60 seconds.

| Step | Stage | Usual | Longest expected |
| --- | --- | --- | --- |
| all | `starting` | under 1 s after the process starts | 10 s |
| `probe` | `loading_libraries` | 3 to 5 s (PyTorch) | 60 s |
| `probe` | `verifying_weights` | 1 to 10 s per model (reads every byte) | 120 s |
| `read` | `loading_file` | 1 to 10 s (Blender, mesh, UniRig's simplification) | 120 s |
| `read` | `saving` | under 1 s | 10 s |
| `rig` | `loading_libraries` | 4 to 8 s | 60 s |
| `rig` | `card` | under 1 s | 10 s |
| `rig` | `loading_model` | SkinTokens 1 to 3 s, UniRig 5 to 15 s | 120 s |
| `rig` | `preparing` | SkinTokens 1 to 5 s | 60 s |
| `rig` | `generating` | SkinTokens 10 to 30 s, UniRig's skeleton pass 15 to 30 s | 300 s |
| `rig` | `skinning` | UniRig's skin pass 15 to 35 s | 300 s |
| `rig` | `saving` | under 1 s | 10 s |
| `write` | `writing` | 3 to 15 s | 180 s |

## Steps

### probe

Reports what this computer offers. Never loads Blender, never loads weights.

- `--light`: the cards and their free memory only. Read through NVML, the NVIDIA driver's own library: about 0.2 seconds and no video memory. If NVML doesn't load, PyTorch is asked instead (as below).
- Without `--light`: also loads PyTorch (3 to 5 seconds; CUDA starts, which may hold a few hundred MB on the first card until the process ends; free memory still comes from NVML when it loads) and reports `python`, `torch`, `cuda`, `cuda_available`, and for each model `models.<model>`: `code` (its code is in the engine folder), `needs_free_bytes` (below), `normal_speed_on_default_device`, and with `--weights MODEL=FOLDER` whether that folder holds every weight file (`weights`, `missing_weights`). `--verify-weights` also reads every weight file and compares its SHA-256 (`verified`, per file: `present`, `bytes`, `sha256`, `ok`).

Always: `engine_version`, `protocol`, `light`, `package` (from `ENGINE.json`), `devices` (each card's `index`, `name`, `total_bytes`, `free_bytes`, and `capability` when PyTorch was asked), `devices_from` (`nvml` or `torch`), `driver` (from NVML), `default_device`.

`needs_free_bytes` is the free video memory a model needs to run at normal speed: SkinTokens 5.5 GiB (5905580032), UniRig 7.5 GiB (8053063680). These are the card peaks measured on Windows (SkinTokens 4.6 to 5.1 GiB; UniRig's skeleton pass 7.1 GiB, its skin pass 6.7 GiB) plus 0.4 GiB. With less free, Windows spills into system memory: the rig still comes out, slower (SkinTokens took 4.5 times as long at 5.2 GiB free). The `rig` step reports the same figure in its `card` line and result.

### read

Blender reads the model file (`.glb`, `.fbx`, or `.obj` with its `.mtl` and textures beside it) and the chosen model's own loader turns it into a read file (below). The read file belongs to one model: SkinTokens and UniRig read the same file differently (UniRig simplifies the mesh to 50,000 faces and orders the bones its own way), so `rig` refuses a read file made for the other model. UniRig's own loaders match file endings by case: a `.GLB` is read from a copy in the work folder, and `.OBJ` in upper case is refused (`bad_input`), since a copy would lose the files beside it.

Result: `model`, `output`, `has_skeleton`, `bone_count`, `armatures`, `armature_bones`, `vertices`, `faces`, and `meshes` (SkinTokens).

**How bones are counted.** All counts are Blender's, after its importer has read the file.

- `bone_count`: the bones the model will use. SkinTokens takes every bone of the first armature. UniRig takes the bones it reaches from the root of the first armature's first bone, so a second root's bones are left out (keeping that skeleton then fails with `skeleton_not_kept`).
- `armatures`, `armature_bones`: every armature in the file and all their bones.
- `has_skeleton` is whether the file has an armature.

Where Blender's count can differ from what three.js shows for the same file before a run: for a `.glb`, three.js counts the joints of the skin, while Blender's glTF importer also makes a bone of every node that sits between joints in the hierarchy, and can join several skins into one armature. For an `.fbx`, files from some tools end each chain with an extra end bone (such as `HeadTop_End`); both models read these as bones (Blender's importer keeps leaf bones here), while three.js may or may not show them depending on how the file stores them. An `.obj` has no skeleton. So the count shown before a run is three.js's, and the one after `read` is Blender's; the second is the one the rig uses.

### rig

The model makes the rig from a read file and writes a rig file (below). Before any model loads it sends a `card` progress line: `device`, `card`, `free_bytes`, `total_bytes`, `capability`, `devices` (how many cards PyTorch sees), so the app can warn when `free_bytes` is under the model's `needs_free_bytes`: Windows never reports out of memory on a full card, it spills into system memory and gets slow.

| Option | Meaning |
| --- | --- |
| `--weights F` | Required: this model's weights folder (Weights, below). Checked before PyTorch loads: missing files fail with `bad_weights` and their names. |
| `--seed N` | Required, 0 to 4294967295. Same seed, same computer, same settings: the same rig (SkinTokens bit for bit; UniRig's skeleton bit for bit, its skin within a small run to run noise on the card). "Rig again" is a new seed. |
| `--standard-names on\|off` | Default on. A made skeleton whose parent structure is exactly the Mixamo layout (22 bones, or 52 with fingers) gets the Mixamo names from the models' `configs/skeleton/mixamo.yaml` (`mixamorig:Hips` ...), for either model; any other skeleton keeps `bone_0` ... Whether the named bones also sit where the names say is reported (`placement_warnings`), never used to decide. |
| `--keep-skeleton` | Skin the file's own skeleton, keeping its bone names and parents. SkinTokens gives the bones back at its own positions, up to 1.5 of its quantization steps from the file's. If the file has no skeleton, one is made and the result says so in `keep_skeleton`. If the model can't keep it, the step fails with `skeleton_not_kept`. |
| `--stage skeleton\|skin` | UniRig only, required: which of its two passes (Sequences). |
| `--cache C` | A folder the app keeps between runs, for CUDA's compiled kernels. Without it they go under the work folder and are compiled again each run (a few seconds for UniRig's skin pass). Safe to reuse after a kill (Folders). |
| `--device N` | The card, numbered as `probe`, NVML and `nvidia-smi` number them. Default: the card with the most memory, the lowest number on a tie. A number with no card fails with `no_cuda`. |

Result: `model`, `output`, `seed`, `stage` (`final`, or `skeleton` for UniRig's first pass), `bones`, `skeleton` (`made` or `kept`), `names` (`standard`, `numbered`, `kept`, or `model` for UniRig's first pass), `standard_names` (`asked`, `matched`, `layout`, `placement_warnings`), `keep_skeleton` (a sentence when keeping was asked but the file had no skeleton), `keep_check` (how the kept bones compared), `input_skeleton_ignored` (SkinTokens: the file's own skeleton, not wanted, could not be read and was left out), `fingerprints` (short hashes of the sampled `points`, `tokens`, `joints` and `skin`, the same hashing as the Windows kit's `st_rig.py`, for comparing runs), `memory` (the `card` line's fields, `needs_free_bytes`, `peak_allocated_bytes`, `peak_reserved_bytes`), `cache` (with `--cache`: `stale_markers`, `cuda_cache_emptied`), `python`, `torch`, `cuda`.

### write

Blender writes the rig onto the original model file with the model's own export code and writes a `.glb`: the original mesh, materials and textures (an OBJ's `.mtl` and texture files included), with the new armature and weights (at most 4 bones per vertex). For UniRig's first pass it writes the skeleton `.fbx` instead, from the rig file alone (no `--input`).

Result: `model`, `stage`, `output`, `bytes`, `sha256` (of the output file, so the app can check the file it takes is the one written), `names`, `skeleton`, and for a `.glb` `bones`, `extra_bones`, `meshes`, `materials`, `images`. The step reads its own `.glb` back and fails unless every bone of the rig is in it by name. `extra_bones` is `["neutral_bone"]` when Blender's glTF export added that bone because some vertices have no weight at all; that is reported, not refused.

## Sequences

What the app runs for each choice. `W` is the run's work folder, `IN` the person's model file, `WS` and `WU` the two weights folders. Every step is its own process through the launcher, started after the one before has ended.

| Model, skeleton | Steps |
| --- | --- |
| SkinTokens, make | `read --model skintokens --input IN` → `rig --model skintokens --weights WS` → `write --input IN` |
| SkinTokens, keep | `read` → `rig --weights WS --keep-skeleton` → `write --input IN` |
| UniRig, make | `read --model unirig --input IN` → `rig --stage skeleton --weights WU` → `write --output W\skeleton.fbx` (no `--input`) → `read --model unirig --input W\skeleton.fbx` → `rig --stage skin --weights WU` → `write --input IN` |
| UniRig, keep | `read --model unirig --input IN` → `rig --stage skin --keep-skeleton --weights WU` → `write --input IN` |

UniRig's six steps are its own five (extract, skeleton, extract of the skeleton file, skin, merge) with the same files in between; its skeleton and skin passes export through Blender, which here happens in a `write` step. Pass the same seed to both of UniRig's rig steps. For "rig again", a read file of the same model file and model can be used again; outputs are never overwritten, so use new names or a new work folder.

## Weights

Each model's weights are one folder the app downloads apart from the root, holding these files at these paths relative to it. `rig --weights` takes that folder. The app checks each download by SHA-256; `probe --verify-weights` checks again on request; `rig` checks that every file is there and fails with `bad_weights` if one can't be loaded as the model's weights. `build/build_engine.py --weights-out <folder>` lays out `<folder>\skintokens` and `<folder>\unirig` from the pinned downloads, each with a `WEIGHTS.json` listing its files, their SHA-256 and their size.

| Model | Path in its weights folder | SHA-256 |
| --- | --- | --- |
| SkinTokens | `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt` | `f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692` |
| SkinTokens | `experiments/skin_vae_2_10_32768/last.ckpt` | `4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4` |
| UniRig | `experiments/skeleton/articulation-xl_quantization_256/model.ckpt` | `d8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4` |
| UniRig | `experiments/skin/articulation-xl/model.ckpt` | `9d40cf42fb9d4c10d8b373d9f4f557c6fbbc5ebacae7ae6b3dce5a0d8a18bc33` |

The weights come from huggingface.co/VAST-AI/SkinTokens at revision `79736cad0fd84de384d5eede659b4ebd24effe33` and huggingface.co/VAST-AI/UniRig at revision `36842e2b5947e9e60f89275b83208c8e74071c63` (THIRD_PARTY.md). Their sizes aren't known on the computer this was built on; the Windows kit reports them.

Two small config files fix the networks' shapes. They are not weights and ship inside the root:

| Model | Path below the root | SHA-256 |
| --- | --- | --- |
| SkinTokens | `engine/models/skintokens/models/Qwen3-0.6B/config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| UniRig | `engine/models/unirig/hf/opt-350m/config.json` | `3ec75df0e166476e342cf95226ba775d8155449a7b1e430728b3ca26b425e4ab` |

## Cards

The engine numbers cards as NVML and `nvidia-smi` do: it sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before PyTorch loads, so CUDA's device N is the same card (without it, CUDA puts the fastest card first). `probe` lists the cards with that number; `rig --device N` uses card N; without `--device`, the card with the most memory. Only NVIDIA cards that PyTorch can use count.

## Folders

The engine writes only inside the work folder (and the cache folder when given). Outputs must be inside the work folder, must not exist yet (nothing is overwritten), and are written under a temporary name and renamed into place. Links are refused.

**The input may sit inside the work folder**, for example a `.glb` the app copied there, or an unpacked OBJ bundle in a subfolder. The engine never moves, changes or deletes it. Names the engine reserves in the work folder, which the app keeps its own files clear of: `tmp`, `cache`, anything starting with a dot (each step's own temporary folder, `.read-*`, `.write-*`, `.unirig-*`), and an output's name followed by `.part`. Outputs with these names are refused.

What each step needs from the launcher (`--read` and `--write` folders):

| Step | Read | Write |
| --- | --- | --- |
| `probe` | the root; each weights folder given | the work folder, if given |
| `read` | the root; the folder of the model file | the work folder |
| `rig` | the root; this model's weights folder | the work folder; the cache folder, if given |
| `write` | the root; the folder of the model file | the work folder |

The folder of the model file is needed, not the file alone, because an `.fbx` or `.obj` may name texture and material files beside it. A model file inside the work folder needs no extra grant.

Left in the work folder after a step: its outputs, `tmp\` (empty unless a step was killed) and `cache\` (Blender's, PyTorch's and Hugging Face's, and the folder the engine uses as the home folder; mostly empty). Each step's own temporary folder is removed before it exits; a killed step leaves it. The app empties the work folder between runs.

**The cache folder is safe to reuse after a kill.** It holds `cuda\` (CUDA's compiled kernels) and, while a `rig` step runs, a marker `.in-use-<pid>`. A marker whose process is gone means a step was killed, perhaps while CUDA was writing; the next `rig` step then empties `cuda\` (recompiling costs a few seconds) and removes the marker, unless another live step's marker is there. Its result says so (`cache.cuda_cache_emptied`). One engine process per cache folder at a time is what the app runs; two at once are safe too.

## Passed by the app

The launcher passes exactly the variables Windows needs to start a program and load the graphics driver, as for the image engine (`apps/desktop/src/main/free-image/sandbox.ts`). The engine needs nothing more:

| Variable | Why |
| --- | --- |
| `SystemRoot` | Windows' own folder: system libraries and the graphics driver load from it. |
| `SystemDrive` | The system drive, which some Windows libraries read. |
| `windir` | The same as `SystemRoot`, for older code. |
| `NUMBER_OF_PROCESSORS` | The thread pools of NumPy, OpenMP and PyTorch size themselves by it. |
| `PROCESSOR_ARCHITECTURE` | Python and the libraries read the machine type from it. |
| `LOCALAPPDATA` | Windows works out the AppContainer's own folders from it and won't start a program in one without it (the launcher exits with `0x4E4C5207`). Inside, the engine sees the container's folder, not the person's, and then points it into the work folder. |

No `PATH` is needed: Python finds its own libraries beside `python.exe`, PyTorch adds its folder of CUDA libraries itself (the engine adds it too, for spconv and torch_scatter), and the driver's libraries load from the system folder. No `CUDA_VISIBLE_DEVICES`: `rig --device` chooses the card.

## Environment

The engine sets these itself at the start of every step, before Blender or PyTorch loads, from its own arguments (`--work`, `--cache`). Folders are under the work folder `W`; `CUDA_CACHE_PATH` goes to the cache folder `C` when `--cache` is given. Whatever the app passes for these is replaced. So nothing is written outside the work and cache folders even though the profile folders aren't passed (and aren't reachable inside the sandbox).

| Variable | Value | Why |
| --- | --- | --- |
| `TEMP` | `W\tmp` | Python's and Blender's temporary files, and Windows' GetTempPath. |
| `TMP` | `W\tmp` | The same, for programs that read TMP. |
| `TMPDIR` | `W\tmp` | The same, for libraries written for Linux. |
| `USERPROFILE` | `W\cache\home` | The home folder: anything that writes under `~` lands in the work folder. |
| `HOME` | `W\cache\home` | The same, for libraries that read HOME. |
| `APPDATA` | `W\cache\home\AppData\Roaming` | The default place for programs' settings and caches. |
| `LOCALAPPDATA` | `W\cache\home\AppData\Local` | The same, local. |
| `USERNAME` | `rig-engine`, only when missing | Some libraries ask for the user's name and fail without one. |
| `HF_HOME` | `W\cache\huggingface` | Hugging Face's cache (nothing is downloaded). |
| `HF_HUB_OFFLINE` | `1` | Hugging Face never tries the network. |
| `TRANSFORMERS_OFFLINE` | `1` | transformers never tries the network. |
| `HF_DATASETS_OFFLINE` | `1` | The same for datasets, if anything imports it. |
| `HF_HUB_DISABLE_TELEMETRY` | `1` | No usage reports. |
| `TORCH_HOME` | `W\cache\torch` | PyTorch's cache. |
| `PYTORCH_KERNEL_CACHE_PATH` | `W\cache\torch\kernels` | PyTorch's compiled kernel cache. |
| `TRITON_CACHE_DIR` | `W\cache\triton` | Triton's cache, if anything compiles with it. |
| `CUDA_CACHE_PATH` | `C\cuda`, else `W\cache\cuda` | CUDA's cache of kernels compiled for this card (spconv ships none for new cards); default `%APPDATA%\NVIDIA\ComputeCache`, outside the sandbox. |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | Cards numbered as nvidia-smi numbers them (Cards). |
| `MPLCONFIGDIR` | `W\cache\matplotlib` | matplotlib's settings, if anything imports it. |
| `XDG_CACHE_HOME` | `W\cache\xdg` | Caches of libraries written for Linux. |
| `WANDB_MODE` | `disabled` | No experiment tracking (UniRig can use it). |
| `WANDB_DIR` | `W\cache\wandb` | The same. |
| `BLENDER_USER_CONFIG` | `W\cache\blender\config` | Blender's settings. |
| `BLENDER_USER_SCRIPTS` | `W\cache\blender\scripts` | Blender's add-on folder. |
| `BLENDER_USER_DATAFILES` | `W\cache\blender\datafiles` | Blender's data files. |
| `BLENDER_USER_EXTENSIONS` | `W\cache\blender\extensions` | Blender's extensions. |
| `BLENDER_USER_RESOURCES` | `W\cache\blender\resources` | Blender's user resources. |
| `TQDM_DISABLE` | `1` | No progress bars in the log. |

## Error codes

Every `error` line has `code` and `message` (for the log, not for people). Some codes add fields; no other code carries them (`protocol.schema.json` checks this).

| Code | Meaning | Extra fields |
| --- | --- | --- |
| `bad_arguments` | The command line is wrong: an unknown option or value, a relative path, an output outside the work folder, one that exists already, or one with a reserved name. Exit code 2. | `path` |
| `bad_input` | An input file is missing, unreadable, of an unsupported kind, a link, not a plain array file, or not what this step expects (for example a read file made for the other model). | `path` |
| `read_failed` | Blender could not read the model file. | |
| `no_mesh` | The model file has no mesh with faces. | |
| `skeleton_not_kept` | The skeleton could not be kept: the model gave back other bones than the file has (SkinTokens: a different count, order or parents, or a bone more than 1.5 quantization steps away; UniRig: bones it dropped, such as a second root). | `keep_check` |
| `no_cuda` | No NVIDIA card PyTorch can use: no card, no card with the asked number, a driver too old for this PyTorch, or a card it has no code for. | `torch`, `built_for`, `devices` |
| `out_of_memory` | The card or the computer ran out of memory. On Windows the card itself rarely reports this; see the `card` line. | |
| `bad_weights` | A weight file is missing (`missing` names them), or could not be loaded as the model's weights. | `missing` |
| `rig_failed` | The model made no usable rig for this model file (for example no complete skeleton, or no skin for it). Another seed may help. | |
| `write_failed` | Blender could not write the rigged file, or the file it wrote lacks bones of the rig. | `bones_written`, `bones_expected`, `extra_bones` |
| `forbidden_module` | Blender was loaded in `rig` or `probe`, or PyTorch in `read` or `write`. A bug, never expected. | `modules` (required) |
| `internal` | Anything else, a broken engine folder included (`missing` names its missing files). A bug; the log has the details. | `missing` |

## The read file

A `.npz` file of plain arrays (NumPy, `allow_pickle=False`: numbers, booleans and text only). Every read file has:

| Key | Type | Meaning |
| --- | --- | --- |
| `format` | text | `rig-engine/read/1` |
| `model` | text | `skintokens` or `unirig` |
| `engine_version` | text | the engine that wrote it |
| `source_name` | text | the model file's name |
| `has_skeleton` | bool | the file has an armature |
| `bone_count` | int | the bones the model uses (Steps, read) |

SkinTokens (`BpyParser.load`'s Asset, field for field, each array in the dtype it made): `vertices` (N×3 float64), `faces` (F×3 int32), `vertex_normals` (N×3), `face_normals` (F×3), `vertex_bias` and `face_bias` (P, int64: where each mesh part ends), `mesh_names` (P, text); with a skeleton `joint_names` (J, text), `parents` (J, int32, -1 for the root), `lengths` (J, float32), `matrix_world` (4×4), `matrix_local` (J×4×4), `armature_name` (text), `skin` (N×J), and `matrix_basis` (frames×J×4×4) when the file is animated.

UniRig (its extract step's `raw_data.npz`): `vertices` (N×3 float32), `vertex_normals` (N×3 float32), `faces` (F×3 int64), `face_normals` (F×3 float32), `armature_bone_count` (int); with a skeleton `joints` (J×3 float32), `tails` (J×3 float32), `parents` (J, int64, -1 for the root), `names` (J, text), `matrix_local` (J×4×4 float32), `skin` (N×J float32).

## The rig file

A `.npz` file of plain arrays, like the read file.

| Key | Type | Meaning |
| --- | --- | --- |
| `format` | text | `rig-engine/rig/1` |
| `model` | text | `skintokens` or `unirig` |
| `engine_version` | text | |
| `stage` | text | `final`, or `skeleton` for UniRig's first pass |
| `seed` | int | the run's seed; `write` seeds the export with it, so a rig file always writes the same file |
| `skeleton` | text | `made` or `kept` |
| `names_source` | text | `standard`, `numbered`, `kept`, or `model` |
| `names` | J, text | the final bone names |
| `parents` | J, int32 | each bone's parent, -1 for the one root |
| `joints` | J×3 float | bone heads, in the model's own frame (the character fitted into a unit cube) |
| `vertices` | N×3 float | the model's copy of the mesh, in that frame |
| `skin` | N×J float | weights per vertex of `vertices` (not for UniRig's first pass) |

The model's own frame and mesh are kept because `write` runs the model's own transfer, which fits them onto the original file. SkinTokens adds `matrix_local` (J×4×4 float32, `joints` with identity rotations, as its export reads them). UniRig adds `faces` (F×3), `tails` (J×3, final stage) and `export_options` (text: JSON of the remaining arguments of its `.fbx` export).

## Repeatability

- SkinTokens is seeded right before its one job, after the model has loaded, and runs with `model.eval()`. Its two skin samplers that took their seed from the system take it from NumPy's seeded generator (patches `st-vae-rng-*`). The model file is fetched the way a data loader worker would fetch it (the sandbox allows no worker process). Same seed, card, driver and engine: the same rig bit for bit. Another computer may give another rig for the same seed.
- UniRig is seeded at the start of each pass, as its `run.py` does. The skeleton repeats bit for bit; the skin differs slightly run to run on the card (atomic additions in its sparse convolution).
- `write` seeds NumPy with the rig file's seed: both models' exports draw random numbers (SkinTokens nudges bones that sit on their parent; UniRig's merge picks the axis order from a random subset of points).
