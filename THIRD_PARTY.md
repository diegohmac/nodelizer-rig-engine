# Third party pieces

The engine's own code (`rigengine.py`, `engine/`, `build/`, `tests/`, `windows-kit/`, and the flash-attn stand in in `standin/`) is under the GNU General Public License 3.0 (`LICENSE`).

An engine folder also holds the two models' code, and each model's weights folder its weights, fetched from the pinned sources below and checked by SHA-256 before use (`build/sources.py`). Licence facts come from the Windows kit's table (`dist/rig-engine-test/README.md`, "What the kit runs, exactly"), from licence files and headers inside the archives, and for OPT's config and timm from their own pages at the pinned versions, read on the build computer.

## Michelangelo's encoder, under GPL 3.0, in the rig step

Both models bundle the shape encoder from Michelangelo (github.com/NeuralCarver/Michelangelo), which is under the GNU General Public License 3.0. In SkinTokens it is `src/model/michelangelo/` (the encoder of the mesh the skeleton is generated from); in UniRig it is `src/model/michelangelo/` (the mesh encoder of the skeleton pass and the global encoder of the skin pass), with its licence file at `src/model/michelangelo/LICENSE`. It runs in the `rig` step, in the same process as PyTorch and NVIDIA's CUDA libraries, which come with PyTorch under NVIDIA's own licence. That is accepted. Blender, also GPL 3.0, never shares a process with PyTorch (README.md).

## In the engine folder and the weights folders

| Piece | Publisher | Source | Pinned revision | SHA-256 | Licence |
| --- | --- | --- | --- | --- | --- |
| SkinTokens' code | VAST AI Research | github.com/VAST-AI-Research/SkinTokens | commit `273b691d35989d71cd17ff2895fdc735097b92d1` | archive `f886ce830f8f6ed5a3eabebb9399244812ac17a44d5b51fe8853c381a214e334` | MIT (its `LICENSE`); bundles Michelangelo's encoder, GPL 3.0 (kit table); `src/model/skin_vae/transformers/tripo2_transformer.py` says Apache 2.0 in its header (HunyuanDiT Authors, Qixun Wang and The HuggingFace Team) |
| SkinTokens' weights: `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt` | VAST AI Research | huggingface.co/VAST-AI/SkinTokens | revision `79736cad0fd84de384d5eede659b4ebd24effe33` | `f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692` | MIT (kit table) |
| SkinTokens' weights: `experiments/skin_vae_2_10_32768/last.ckpt` | VAST AI Research | huggingface.co/VAST-AI/SkinTokens | revision `79736cad0fd84de384d5eede659b4ebd24effe33` | `4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4` | MIT (kit table) |
| UniRig's code | VAST AI Research | github.com/VAST-AI-Research/UniRig | commit `6793c6640ff01c8fb389f3993434124bb43d2933` | archive `ebdda5907b520b6ccfa0524f5050c7141475d203289052e9c39024fae039c1a1` | MIT (its `LICENSE`); bundles Michelangelo's encoder, GPL 3.0 (`src/model/michelangelo/LICENSE`), and Pointcept's PTv3, MIT (`src/model/pointcept/LICENSE`) |
| UniRig's weights: `experiments/skeleton/articulation-xl_quantization_256/model.ckpt` | VAST AI Research | huggingface.co/VAST-AI/UniRig | revision `36842e2b5947e9e60f89275b83208c8e74071c63` | `d8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4` | MIT (kit table) |
| UniRig's weights: `experiments/skin/articulation-xl/model.ckpt` | VAST AI Research | huggingface.co/VAST-AI/UniRig | revision `36842e2b5947e9e60f89275b83208c8e74071c63` | `9d40cf42fb9d4c10d8b373d9f4f557c6fbbc5ebacae7ae6b3dce5a0d8a18bc33` | MIT (kit table) |
| Qwen3 0.6B's `config.json` (the network's shape only; no weights) | Alibaba Cloud (Qwen) | huggingface.co/Qwen/Qwen3-0.6B | revision `c1899de289a04d12100db370d81485cdf75e47ca` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` | Apache 2.0 (kit table) |
| OPT 350m's `config.json` (the network's shape only; no weights) | Meta | huggingface.co/facebook/opt-350m | revision `08ab08cc4b72ff5593870b5d527cf4230323703c` | `3ec75df0e166476e342cf95226ba775d8155449a7b1e430728b3ca26b425e4ab` | "other", with "commercial: false": the model card (`README.md`) at that revision says so and names no licence text. It covers OPT's weights; whether it covers this config file (the network's sizes, no weights) is a question for a lawyer. |
| timm's `DropPath`, copied into UniRig's `PTv3Object.py` by patch `ur-ptv3-no-timm` (17 lines) | Ross Wightman and the timm contributors (huggingface/pytorch-image-models) | github.com/huggingface/pytorch-image-models, tag `v1.0.31` (the version in the Windows kit) | 1.0.31 | in the kit's `requirements-win.txt` | Apache 2.0: its `LICENSE` at that tag and its PyPI entry. Apache 2.0 code may be combined into a GPL 3.0 work; the copy says where it came from. |

## Needed to run, not in this repository

The engine runs on the Windows kit's Python environment (`dist/rig-engine-test/python`, every package at the version and checksum in its `requirements-win.txt`). The pieces it loads:

| Piece | Publisher | Source | Revision | Licence |
| --- | --- | --- | --- | --- |
| Python 3.11.9 | Python Software Foundation | nuget.org, package `python` 3.11.9 | package SHA-256 `9283876d58c017e0e846f95b490da3bca0fc0a6ee1134b2870677cfb7eec3c67` | PSF licence (kit table) |
| PyTorch 2.7.1 with CUDA 12.8 | PyTorch Foundation | download.pytorch.org/whl/cu128 | checksums in `requirements-win.txt` | BSD 3, bundles NVIDIA's CUDA libraries under NVIDIA's licence (kit table) |
| Blender as a Python module, `bpy` 4.2.0 | Blender Foundation | pypi.org | checksums in `requirements-win.txt` | GPL 3.0 (kit table) |
| spconv-cu126 2.3.8, cumm-cu126 0.7.11 | Yan Yan (traveller59) | pypi.org | checksums in `requirements-win.txt` | Apache 2.0 (kit table) |
| torch_scatter 2.1.2, torch_cluster 1.6.3 | PyG team | data.pyg.org/whl/torch-2.7.0+cu128 | checksums in the kit's `kit\fetch.py` | MIT (kit table) |
| transformers 4.57.6, lightning 2.6.6, open3d 0.20.0, trimesh, scipy, numpy and the other packages | their PyPI publishers | pypi.org | versions and checksums in `requirements-win.txt` | various open source licences (kit table); not checked one by one here |

The engine needs neither pyrender, nor timm, nor torchvision, nor flash-attn: the patches and the stand in in `standin/flash_attn` replace them.
