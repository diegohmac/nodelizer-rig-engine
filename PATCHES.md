# Changes to the models' code

The engine folder holds SkinTokens' and UniRig's code at their pinned commits (see THIRD_PARTY.md), changed only by the patches below. `build/patches.py` applies them: each replaces one exact piece of text and the build stops if that text isn't found exactly once, or if a file has Windows line endings, so a patch can never land on code that differs from what was reviewed. A second list of strings must be gone afterwards (`MUST_NOT_CONTAIN`). The reason column here is the `reason` text in `build/patches.py`; a test keeps every id listed.

The rule was to drive the models from our own code (`engine/skintokens`, `engine/unirig`) and patch only where that can't reach. So most of what the Blender and PyTorch split needed is in the drivers, not here.

## Patches

| Patch | File | Why |
| --- | --- | --- |
| `st-qwen-sdpa` | skintokens/src/model/tokenrig.py | Qwen3 runs on PyTorch's own attention, since flash-attn has no Windows build (kit patch 1). |
| `st-ckpt-restricted` | skintokens/src/model/spec.py | Checkpoints open with PyTorch's restricted loader, which cannot run code from the file (kit patch 2). |
| `st-vae-rng-tripo2` | skintokens/src/model/skin_vae/autoencoders/autoencoder_kl_tripo2.py | The skin VAE's point sampler takes its seed from NumPy's seeded generator, not the system (kit patch 4). |
| `st-vae-rng-fsq` | skintokens/src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py | The second skin VAE sampler, the same change (kit patch 4). |
| `st-helper-no-unpickle` | skintokens/src/server/spec.py | The helper's decoder unpickled anything it was sent; nothing in the engine may call it now. |
| `st-helper-no-socket` | skintokens/src/data/datapath.py | The helper loader opened a socket to a local port; the engine never uses the helper. |
| `st-lazy-asset-no-bpy` | skintokens/src/data/datapath.py | The rig step must never import Blender, even by accident; files are read by the read step. |
| `st-debug-export-no-bpy` | skintokens/src/model/tokenrig.py | A debug export imported Blender inside the torch side (tokenrig.py:313). |
| `st-npz-no-pickle` | skintokens/src/data/datapath.py | Its npz loader allowed pickle; plain arrays load the same without it. |
| `st-unirig-npz-no-pickle` | skintokens/src/data/datapath.py | The second npz loader, the same change. |
| `ur-opt-local` | unirig/configs/model/unirig_ar_350m_1024_81920_float32.yaml | OPT's config is read from the engine folder, not the network (kit patch 5, made relative). |
| `ur-opt-sdpa` | unirig/configs/model/unirig_ar_350m_1024_81920_float32.yaml | OPT runs on PyTorch's own attention (kit patch 6). |
| `ur-ptv3-no-flash` | unirig/configs/model/unirig_skin.yaml | The mesh encoder (PTv3) takes its path without flash-attn (kit patch 7). |
| `ur-ptv3-sdpa` | unirig/src/model/pointcept/models/PTv3Object.py | That path uses PyTorch's fused attention, which keeps the skin step's memory where it was (kit patch 8). |
| `ur-ptv3-no-timm` | unirig/src/model/pointcept/models/PTv3Object.py | timm's DropPath, copied: identity at inference, so UniRig needs neither timm nor torchvision (Windows work-3). |
| `ur-voxel-open3d` | unirig/configs/transform/inference_skin_transform.yaml | The skin step voxelizes with open3d, UniRig's other backend; pyrender is not installed (Windows work-1). |
| `ur-download-local` | unirig/src/inference/download.py | download() fetched unpinned weights from the Hub when online; it now always returns the local file. |
| `ur-download-no-hub` | unirig/src/inference/download.py | The Hub client is no longer imported there. |
| `ur-rawdata-helpers` | unirig/src/data/raw_data.py | Plain array save and load for UniRig's intermediate files, so nothing between steps needs pickle. |
| `ur-rawdata-load` | unirig/src/data/raw_data.py | RawData loads without pickle; parents come back as before (an object array of int and None). |
| `ur-rawdata-save` | unirig/src/data/raw_data.py | RawData saves without pickle. |
| `ur-rawskeleton-plain` | unirig/src/data/raw_data.py | RawSkeleton saves and loads without pickle. |
| `ur-rawskin-plain` | unirig/src/data/raw_data.py | RawSkin saves and loads without pickle. |
| `ur-extract-logs` | unirig/src/data/extract.py | UniRig's own extract command wrote logs into its code folder (Windows work-5). |
| `ur-trainer-no-logger` | unirig/run.py | UniRig's own run.py let Lightning write lightning_logs into its code folder (Windows work-5). |
| `ur-merge-no-open3d` | unirig/src/inference/merge.py | merge.py imported open3d without using it, which loads open3d and its viewer stack in the write step. |

The kit's patch 3 (SkinTokens' helper listening on 127.0.0.1) is gone: the helper isn't shipped at all.

## Left out of the engine folder

| Left out | Why |
| --- | --- |
| `skintokens/demo.py` | SkinTokens' web demo: a gradio server and the helper process. |
| `skintokens/bpy_server.py` | Starts the helper, a local server that unpickles what it is sent. |
| `skintokens/src/server/bpy_server.py` | The helper itself. |
| `skintokens/download.py` | Downloads weights from the Hub at the latest revision. |
| `skintokens/assets` | README images. |
| `skintokens/examples` | Example models. |
| `unirig/assets` | README images. |
| `unirig/examples` | Example models. |
| `unirig/blender` | A zipped Blender add-on for VRM files, which the engine does not read. |

## Done in our drivers instead of patches

These change how the models run without touching their code.

- **No Blender in the rig step.** SkinTokens' dataset gets the Asset from the read file through a `LazyAsset` of ours, in place of `BpyLazyAsset`. UniRig's writers end in `Exporter._export_fbx`, which our rig driver replaces in its own process with a recorder; the write step calls the real one with the recorded arguments.
- **UniRig's two exports through Blender** (the skeleton `.fbx`, then the skin `.fbx` that merge reads) happen in write steps, so UniRig is two rounds of read, rig, write (PROTOCOL.md, Sequences).
- **Fast model load.** SkinTokens' model is built inside transformers' `no_init_weights()`: 1 second instead of 35, the same tensors and rigs (Windows work-4).
- **One process.** No data loader workers: SkinTokens' fetch is seeded as worker 0 would be (Windows work-1); UniRig's own command already used none.
- **Restricted checkpoint loading for UniRig.** The driver opens UniRig's checkpoints itself with `torch.load(weights_only=True)` inside `torch.serialization.safe_globals([Box])`, instead of leaving that to Lightning.
- **No log folders.** Lightning gets `logger=False` and a scratch root folder; UniRig's extract helpers log into the read step's scratch folder.
- **Relative paths.** Both models name configs and checkpoints relative to their own folder, so the rig step changes into it (read only); every path it writes is absolute and in the work folder.
