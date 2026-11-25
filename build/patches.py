# Every change made to the two models' code, as exact text replacements. Each one must find its text
# exactly once in the file, or the build stops: it can never patch code that differs from what was
# reviewed. PATCHES.md gives the reason for each in one line; `reason` here is the same text.
#
# Standard library only, so the tests can import it without the models' packages.
from pathlib import Path

# --- SkinTokens ----------------------------------------------------------------------------------

_ST_RNG_OLD = "        rng = np.random.default_rng(seed)\n"
_ST_RNG_NEW = ("        rng = np.random.default_rng(np.random.randint(2**31) if seed is None else seed)"
               "  # rig engine: was system entropy when seed is None\n")

# --- UniRig ----------------------------------------------------------------------------------------

_PTV3_ATTN_OLD = """            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
"""
_PTV3_ATTN_NEW = """            if not self.enable_rpe and not self.upcast_softmax and not self.training:
                # rig engine: PyTorch's fused attention, without keeping the (N', H, K, K) matrix
                feat = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
                feat = feat.to(qkv.dtype).transpose(1, 2).reshape(-1, C)
            else:
""" + "".join("    " + line + "\n" for line in _PTV3_ATTN_OLD.splitlines())

_DROP_PATH_OLD = "from timm.models.layers import DropPath\n"
_DROP_PATH_NEW = '''import torch.nn as _nn  # rig engine: timm's DropPath, copied, so UniRig needs neither timm nor torchvision


class DropPath(_nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:  # always the case at inference
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        if keep > 0.0 and self.scale_by_keep:
            mask.div_(keep)
        return x * mask
'''

_DOWNLOAD_OLD = """    try:
        if ckpt_name not in MAP:
            print(f"not found: {ckpt_name}")
            return ckpt_name
        return hf_hub_download(
            repo_id='VAST-AI/UniRig',
            filename=MAP[ckpt_name],
        )
    except Exception as e:
        print(f"Failed to download {ckpt_name}: {e}")
        return ckpt_name"""
_DOWNLOAD_NEW = """    return ckpt_name  # rig engine: always the local, pinned file; never the network"""

_RAW_HELPERS_OLD = "from .order import Order\n"
_RAW_HELPERS_NEW = '''from .order import Order

# rig engine: these files are written and read as plain arrays, never with pickle. A parent of None
# (a root) is stored as -1, a list of names as a text array, and a field that is None is left out.
_RAW_DATA_FIELDS = ('vertices', 'vertex_normals', 'faces', 'face_normals', 'joints', 'skin', 'parents',
                    'names', 'matrix_local', 'tails', 'no_skin', 'path', 'cls')
_RAW_SKELETON_FIELDS = ('joints', 'tails', 'no_skin', 'parents', 'names')
_RAW_SKIN_FIELDS = ('skin', 'vertices', 'joints')


def _plain_save(path: str, fields: dict):
    arrays = {}
    for key, value in fields.items():
        if value is None:
            continue
        if key == 'parents':
            value = np.array([-1 if p is None else int(p) for p in value], dtype=np.int64)
        elif isinstance(value, (list, tuple)):
            if not all(isinstance(x, str) for x in value):
                raise TypeError(f"rig engine: {key} must be a list of names")
            value = np.array(list(value), dtype=np.str_)
        elif isinstance(value, str):
            value = np.array(value, dtype=np.str_)
        else:
            value = np.asarray(value)
        if value.dtype == object:
            raise TypeError(f"rig engine: {key} would need pickle")
        arrays[key] = value
    np.savez(file=path, **arrays)


def _plain_load(path: str, names: tuple) -> dict:
    data = np.load(path, allow_pickle=False)
    d = {name: data[name][()] for name in data.files if name in names}
    for name in names:
        d.setdefault(name, None)
    if d['parents'] is not None:
        d['parents'] = np.array([None if int(p) < 0 else int(p) for p in d['parents']], dtype=object)
    return d
'''

_RAW_DATA_LOAD_OLD = """        data = np.load(path, allow_pickle=True)
        d = {name: data[name][()] for name in data}
"""
_RAW_DATA_LOAD_NEW = """        d = _plain_load(path, _RAW_DATA_FIELDS)  # rig engine: plain arrays, no pickle
"""

_RAW_DATA_SAVE_OLD = (
    '        return RawData(**d)\n'
    '    \n'
    '    def save(self, path: str):\n'
    '        os.makedirs(os.path.dirname(path), exist_ok=True)\n'
    '        np.savez(file=path, **self.__dict__)\n'
)
_RAW_DATA_SAVE_NEW = (
    '        return RawData(**d)\n'
    '    \n'
    '    def save(self, path: str):\n'
    '        os.makedirs(os.path.dirname(path), exist_ok=True)\n'
    '        _plain_save(path, self.__dict__)  # rig engine: plain arrays, no pickle\n'
)

_RAW_SKELETON_OLD = (
    '        data = np.load(path, allow_pickle=True)\n'
    '        return RawSkeleton(**{name: data[name][()] for name in data})\n'
    '    \n'
    '    def save(self, path: str):\n'
    '        os.makedirs(os.path.dirname(path), exist_ok=True)\n'
    '        np.savez(file=path, **self.__dict__)\n'
)
_RAW_SKELETON_NEW = (
    '        return RawSkeleton(**_plain_load(path, _RAW_SKELETON_FIELDS))  # rig engine: no pickle\n'
    '    \n'
    '    def save(self, path: str):\n'
    '        os.makedirs(os.path.dirname(path), exist_ok=True)\n'
    '        _plain_save(path, self.__dict__)  # rig engine: plain arrays, no pickle\n'
)

_RAW_SKIN_OLD = (
    '        data = np.load(path, allow_pickle=True)\n'
    '        return RawSkin(**{name: data[name][()] for name in data})\n'
    '    \n'
    '    def save(self, path: str):\n'
    '        os.makedirs(os.path.dirname(path), exist_ok=True)\n'
    '        np.savez(file=path, **self.__dict__)'  # the last line of the file, with no newline after it
)
_RAW_SKIN_NEW = (
    '        return RawSkin(**_plain_load(path, _RAW_SKIN_FIELDS))  # rig engine: no pickle\n'
    '    \n'
    '    def save(self, path: str):\n'
    '        os.makedirs(os.path.dirname(path), exist_ok=True)\n'
    '        _plain_save(path, self.__dict__)  # rig engine: plain arrays, no pickle\n'
)

# (id, model, file, old text, new text, reason)
PATCHES = [
    # SkinTokens
    ("st-qwen-sdpa", "skintokens", "src/model/tokenrig.py",
     'attn_implementation="flash_attention_2").to(torch.bfloat16)',
     'attn_implementation="sdpa").to(torch.bfloat16)',
     "Qwen3 runs on PyTorch's own attention, since flash-attn has no Windows build (kit patch 1)."),
    ("st-ckpt-restricted", "skintokens", "src/model/spec.py",
     'ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)',
     'ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)',
     "Checkpoints open with PyTorch's restricted loader, which cannot run code from the file (kit patch 2)."),
    ("st-vae-rng-tripo2", "skintokens", "src/model/skin_vae/autoencoders/autoencoder_kl_tripo2.py",
     _ST_RNG_OLD, _ST_RNG_NEW,
     "The skin VAE's point sampler takes its seed from NumPy's seeded generator, not the system (kit patch 4)."),
    ("st-vae-rng-fsq", "skintokens", "src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py",
     _ST_RNG_OLD, _ST_RNG_NEW,
     "The second skin VAE sampler, the same change (kit patch 4)."),
    ("st-helper-no-unpickle", "skintokens", "src/server/spec.py",
     "    return torch.load(io.BytesIO(b), weights_only=False, map_location=map_location)",
     '    raise RuntimeError("rig engine: SkinTokens\' helper protocol is not used; it would unpickle untrusted data")',
     "The helper's decoder unpickled anything it was sent; nothing in the engine may call it now."),
    ("st-helper-no-socket", "skintokens", "src/data/datapath.py",
     '            asset = bytes_to_object(requests.get(f"{BPY_SERVER}/load", data=object_to_bytes(self.path)).content)',
     '            raise RuntimeError("rig engine: SkinTokens\' helper is not used")',
     "The helper loader opened a socket to a local port; the engine never uses the helper."),
    ("st-lazy-asset-no-bpy", "skintokens", "src/data/datapath.py",
     "        from ..rig_package.parser.bpy import BpyParser\n        asset = BpyParser.load(filepath=self.path)\n",
     '        raise RuntimeError("rig engine: the rig step never loads Blender; the read step does this")\n',
     "The rig step must never import Blender, even by accident; files are read by the read step."),
    ("st-debug-export-no-bpy", "skintokens", "src/model/tokenrig.py",
     "        from ..rig_package.parser.bpy import BpyParser\n        BpyParser.export_asset(asset, filepath=path)\n",
     '        raise RuntimeError("rig engine: the rig step never loads Blender")\n',
     "A debug export imported Blender inside the torch side (tokenrig.py:313)."),
    ("st-npz-no-pickle", "skintokens", "src/data/datapath.py",
     "        d = np.load(self.path, allow_pickle=True)\n        asset = Asset(\n",
     "        d = np.load(self.path, allow_pickle=False)  # rig engine: no pickle\n        asset = Asset(\n",
     "Its npz loader allowed pickle; plain arrays load the same without it."),
    ("st-unirig-npz-no-pickle", "skintokens", "src/data/datapath.py",
     "        d = np.load(self.path, allow_pickle=True)\n        parents = bn(",
     "        d = np.load(self.path, allow_pickle=False)  # rig engine: no pickle\n        parents = bn(",
     "The second npz loader, the same change."),
    ("st-flash3-check-cpu", "skintokens", "src/model/michelangelo/utils/misc.py",
     '        self.available = "H100" in get_device_type()\n',
     '        self.available = torch.cuda.is_available() and "H100" in get_device_type()  # rig engine\n',
     "Michelangelo asked CUDA for the card's name when imported, which fails without one (processor, Mac); "
     "with a card the answer is unchanged."),
    # UniRig
    ("ur-opt-local", "unirig", "configs/model/unirig_ar_350m_1024_81920_float32.yaml",
     "pretrained_model_name_or_path: facebook/opt-350m",
     "pretrained_model_name_or_path: ./hf/opt-350m",
     "OPT's config is read from the engine folder, not the network (kit patch 5, made relative)."),
    ("ur-opt-sdpa", "unirig", "configs/model/unirig_ar_350m_1024_81920_float32.yaml",
     "\n  _attn_implementation: flash_attention_2\n",
     "\n  _attn_implementation: sdpa\n",
     "OPT runs on PyTorch's own attention (kit patch 6)."),
    ("ur-ptv3-no-flash", "unirig", "configs/model/unirig_skin.yaml",
     "\n  res_linear: True\n",
     "\n  res_linear: True\n  enable_flash: False  # rig engine: PTv3 without flash-attn\n",
     "The mesh encoder (PTv3) takes its path without flash-attn (kit patch 7)."),
    ("ur-ptv3-sdpa", "unirig", "src/model/pointcept/models/PTv3Object.py",
     _PTV3_ATTN_OLD, _PTV3_ATTN_NEW,
     "That path uses PyTorch's fused attention, which keeps the skin step's memory where it was (kit patch 8)."),
    ("ur-ptv3-no-timm", "unirig", "src/model/pointcept/models/PTv3Object.py",
     _DROP_PATH_OLD, _DROP_PATH_NEW,
     "timm's DropPath, copied: identity at inference, so UniRig needs neither timm nor torchvision (Windows work-3)."),
    ("ur-voxel-open3d", "unirig", "configs/transform/inference_skin_transform.yaml",
     "backend: pyrender # switch to 'open3d' if pyrender does not work",
     "backend: open3d # rig engine: pyrender is not in the environment and needs an OpenGL context",
     "The skin step voxelizes with open3d, UniRig's other backend; pyrender is not installed (Windows work-1)."),
    ("ur-download-local", "unirig", "src/inference/download.py",
     _DOWNLOAD_OLD, _DOWNLOAD_NEW,
     "download() fetched unpinned weights from the Hub when online; it now always returns the local file."),
    ("ur-download-no-hub", "unirig", "src/inference/download.py",
     "from huggingface_hub import hf_hub_download\n",
     "# rig engine: no Hub downloads (see download below)\n",
     "The Hub client is no longer imported there."),
    ("ur-rawdata-helpers", "unirig", "src/data/raw_data.py",
     _RAW_HELPERS_OLD, _RAW_HELPERS_NEW,
     "Plain array save and load for UniRig's intermediate files, so nothing between steps needs pickle."),
    ("ur-rawdata-load", "unirig", "src/data/raw_data.py",
     _RAW_DATA_LOAD_OLD, _RAW_DATA_LOAD_NEW,
     "RawData loads without pickle; parents come back as before (an object array of int and None)."),
    ("ur-rawdata-save", "unirig", "src/data/raw_data.py",
     _RAW_DATA_SAVE_OLD, _RAW_DATA_SAVE_NEW,
     "RawData saves without pickle."),
    ("ur-rawskeleton-plain", "unirig", "src/data/raw_data.py",
     _RAW_SKELETON_OLD, _RAW_SKELETON_NEW,
     "RawSkeleton saves and loads without pickle."),
    ("ur-rawskin-plain", "unirig", "src/data/raw_data.py",
     _RAW_SKIN_OLD, _RAW_SKIN_NEW,
     "RawSkin saves and loads without pickle."),
    ("ur-extract-logs", "unirig", "src/data/extract.py",
     '    log_path = "./logs"\n',
     '    log_path = os.path.join(output_folder, "logs")  # rig engine: never into the code folder\n',
     "UniRig's own extract command wrote logs into its code folder (Windows work-5)."),
    ("ur-trainer-no-logger", "unirig", "run.py",
     "        logger=logger,\n        **trainer_config,\n",
     "        logger=logger if logger is not None else False,  # rig engine: no lightning_logs folder\n"
     "        **trainer_config,\n",
     "UniRig's own run.py let Lightning write lightning_logs into its code folder (Windows work-5)."),
    ("ur-merge-no-open3d", "unirig", "src/inference/merge.py",
     "import open3d as o3d\nimport itertools\n",
     "import itertools  # rig engine: open3d was imported here but never used\n",
     "merge.py imported open3d without using it, which loads open3d and its viewer stack in the write step."),
]

# Files and folders left out of the engine folder: nothing imports them, and some open sockets or
# download from the network. (model, path, reason)
REMOVALS = [
    ("skintokens", "demo.py", "SkinTokens' web demo: a gradio server and the helper process."),
    ("skintokens", "bpy_server.py", "Starts the helper, a local server that unpickles what it is sent."),
    ("skintokens", "src/server/bpy_server.py", "The helper itself."),
    ("skintokens", "download.py", "Downloads weights from the Hub at the latest revision."),
    ("skintokens", "assets", "README images."),
    ("skintokens", "examples", "Example models."),
    ("unirig", "assets", "README images."),
    ("unirig", "examples", "Example models."),
    ("unirig", "blender", "A zipped Blender add-on for VRM files, which the engine does not read."),
]

# After patching, none of these may appear in the named files (a second check that the patches took).
MUST_NOT_CONTAIN = [
    ("skintokens", "src/model/tokenrig.py", "flash_attention_2"),
    ("skintokens", "src/model/tokenrig.py", "parser.bpy"),
    ("skintokens", "src/model/spec.py", "weights_only=False"),
    ("skintokens", "src/server/spec.py", "weights_only=False"),
    ("skintokens", "src/data/datapath.py", "allow_pickle=True"),
    ("skintokens", "src/data/datapath.py", "parser.bpy"),
    ("skintokens", "src/data/datapath.py", "requests.get("),
    ("unirig", "configs/model/unirig_ar_350m_1024_81920_float32.yaml", "flash_attention_2"),
    ("unirig", "configs/model/unirig_ar_350m_1024_81920_float32.yaml", "facebook/opt-350m"),
    ("unirig", "configs/transform/inference_skin_transform.yaml", "backend: pyrender"),
    ("unirig", "src/model/pointcept/models/PTv3Object.py", "from timm"),
    ("unirig", "src/inference/download.py", "hf_hub_download("),
    ("unirig", "src/data/raw_data.py", "allow_pickle=True"),
    ("unirig", "src/data/extract.py", '"./logs"'),
    ("unirig", "src/inference/merge.py", "import open3d"),
]


class PatchError(Exception):
    pass


def apply_patches(models_root, log=print, models=("skintokens", "unirig")):
    """Applies every patch of the given models to the unpacked trees under models_root/<model>/."""
    models_root = Path(models_root)
    for patch_id, model, rel, old, new, _reason in PATCHES:
        if model not in models:
            continue
        path = models_root / model / rel
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raise PatchError(f"{patch_id}: {model}/{rel} is missing")
        text = raw.decode("utf-8")
        if "\r\n" in text:
            raise PatchError(f"{patch_id}: {model}/{rel} has Windows line endings; it is not the reviewed file")
        count = text.count(old)
        if count != 1:
            raise PatchError(f"{patch_id}: expected the text exactly once in {model}/{rel}, found {count}")
        path.write_bytes(text.replace(old, new).encode("utf-8"))
        log(f"patched  {patch_id:28s} {model}/{rel}")
    for model, rel, text in MUST_NOT_CONTAIN:
        if model in models and text in (models_root / model / rel).read_text(encoding="utf-8"):
            raise PatchError(f"{model}/{rel} still contains {text!r}")


def apply_removals(models_root, log=print, models=("skintokens", "unirig")):
    import shutil

    models_root = Path(models_root)
    for model, rel, _reason in REMOVALS:
        if model not in models:
            continue
        path = models_root / model / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            raise PatchError(f"expected {model}/{rel} in the upstream tree")
        log(f"removed  {model}/{rel}")
