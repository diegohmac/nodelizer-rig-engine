"""Folders: the package root, the engine folder (read only), the one work folder a run may write, the
optional cache folder, and the environment.

Every file a step writes goes under the work folder (or the cache folder): outputs, temporary files,
the caches of Blender, PyTorch, Hugging Face and CUDA. The engine sets the environment variables below
itself at start, before Blender or PyTorch loads, from its own arguments, so the app passes only the
few Windows needs to start a program (PROTOCOL.md, Environment). A test keeps that table and this one
in step.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

from .events import EngineError

ENGINE_ROOT = Path(__file__).resolve().parent.parent   # <root>/engine
PACKAGE_ROOT = ENGINE_ROOT.parent                       # <root>, which also holds python/ and ENGINE.json
MODELS = ("skintokens", "unirig")
MODEL_FILE_SUFFIXES = (".glb", ".fbx", ".obj")
# Names in the work folder that belong to the engine; the app keeps its own files clear of them.
RESERVED = ("tmp", "cache", "names starting with a dot", "an output's name followed by .part")


def model_root(model):
    return ENGINE_ROOT / "models" / model


def manifest():
    """ENGINE.json at the package root, or {} when this engine folder sits elsewhere."""
    path = PACKAGE_ROOT / "ENGINE.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# variable: (where it points, why the engine sets it). "work:x" is a folder under the work folder,
# "cache:x" under the cache folder when the app gives one (else under work/cache), "=v" a fixed value,
# "?=v" a value set only when the variable is missing.
ENVIRONMENT = {
    "TEMP": ("work:tmp", "Python's and Blender's temporary files, and Windows' GetTempPath"),
    "TMP": ("work:tmp", "the same, for programs that read TMP"),
    "TMPDIR": ("work:tmp", "the same, for libraries written for Linux"),
    "USERPROFILE": ("work:cache/home", "the home folder: anything that writes under ~ lands in the work folder"),
    "HOME": ("work:cache/home", "the same, for libraries that read HOME"),
    "APPDATA": ("work:cache/home/AppData/Roaming", "default place for programs' settings and caches"),
    "LOCALAPPDATA": ("work:cache/home/AppData/Local", "the same, local"),
    "USERNAME": ("?=rig-engine", "some libraries ask for the user's name (getpass) and fail without one"),
    "HF_HOME": ("work:cache/huggingface", "Hugging Face's cache (nothing is downloaded)"),
    "HF_HUB_OFFLINE": ("=1", "Hugging Face never tries the network"),
    "TRANSFORMERS_OFFLINE": ("=1", "transformers never tries the network"),
    "HF_DATASETS_OFFLINE": ("=1", "the same for datasets, if anything imports it"),
    "HF_HUB_DISABLE_TELEMETRY": ("=1", "no usage reports"),
    "TORCH_HOME": ("work:cache/torch", "PyTorch's cache"),
    "PYTORCH_KERNEL_CACHE_PATH": ("work:cache/torch/kernels", "PyTorch's compiled kernel cache"),
    "TRITON_CACHE_DIR": ("work:cache/triton", "Triton's cache, if anything compiles with it"),
    "CUDA_CACHE_PATH": ("cache:cuda", "CUDA's cache of kernels compiled for this card (spconv has none built "
                                      "for new cards); kept in the cache folder so it survives runs"),
    "CUDA_DEVICE_ORDER": ("=PCI_BUS_ID", "CUDA numbers cards as nvidia-smi and NVML do, so --device means the same"),
    "MPLCONFIGDIR": ("work:cache/matplotlib", "matplotlib's settings, if anything imports it"),
    "XDG_CACHE_HOME": ("work:cache/xdg", "caches of libraries written for Linux"),
    "WANDB_MODE": ("=disabled", "no experiment tracking (UniRig can use it)"),
    "WANDB_DIR": ("work:cache/wandb", "the same"),
    "BLENDER_USER_CONFIG": ("work:cache/blender/config", "Blender's settings"),
    "BLENDER_USER_SCRIPTS": ("work:cache/blender/scripts", "Blender's add-on folder"),
    "BLENDER_USER_DATAFILES": ("work:cache/blender/datafiles", "Blender's data files"),
    "BLENDER_USER_EXTENSIONS": ("work:cache/blender/extensions", "Blender's extensions"),
    "BLENDER_USER_RESOURCES": ("work:cache/blender/resources", "Blender's user resources"),
    "TQDM_DISABLE": ("=1", "no progress bars in the log"),
}


def same_or_inside(path, folder):
    path = os.path.normcase(os.path.abspath(path))
    folder = os.path.normcase(os.path.abspath(folder))
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:  # different drives on Windows
        return False


def work_folder(value):
    path = Path(value)
    if not path.is_absolute():
        raise EngineError("bad_arguments", "the work folder path must be absolute", path=str(value))
    path = path.resolve()
    if not path.is_dir():
        raise EngineError("bad_arguments", "the work folder does not exist", path=str(path))
    return path


def cache_folder(value):
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_dir():
        raise EngineError("bad_arguments", "the cache folder must be an absolute path to a folder", path=str(value))
    return path.resolve()


def set_environment(work, cache=None):
    """Points every temporary and cache location into the work (or cache) folder. Returns what was set."""
    applied = {}
    for name, (rule, _why) in ENVIRONMENT.items():
        if rule.startswith("?="):
            if os.environ.get(name):
                continue
            value = rule[2:]
        elif rule.startswith("="):
            value = rule[1:]
        else:
            kind, rel = rule.split(":", 1)
            base = cache if (kind == "cache" and cache is not None) else work / "cache" if kind == "cache" else work
            folder = base.joinpath(*rel.split("/"))
            folder.mkdir(parents=True, exist_ok=True)
            value = str(folder)
        os.environ[name] = value
        applied[name] = value
    tempfile.tempdir = None  # read TEMP again
    return applied


# ---- the cache folder, safe to reuse after a kill ----

_MARKER = None


def _alive(pid):
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no such process
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim_cache(cache):
    """Marks the cache folder in use by this process. A marker left by a process that is gone means a
    step was killed while it may have been writing; its CUDA cache is then emptied (it only costs a few
    seconds of compiling again). Returns what was found."""
    global _MARKER
    if cache is None:
        return {}
    stale, live = [], []
    for marker in cache.glob(".in-use-*"):
        try:
            pid = int(marker.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        (live if _alive(pid) else stale).append(marker)
    emptied = False
    if stale and not live:
        shutil.rmtree(cache / "cuda", ignore_errors=True)
        emptied = True
    for marker in stale:
        marker.unlink(missing_ok=True)
    _MARKER = cache / f".in-use-{os.getpid()}"
    _MARKER.write_text("", encoding="utf-8")
    (cache / "cuda").mkdir(exist_ok=True)
    return {"stale_markers": len(stale), "cuda_cache_emptied": emptied}


def release_cache():
    if _MARKER is not None:
        try:
            _MARKER.unlink()
        except OSError:
            pass


# ---- inputs and outputs ----

def input_file(value, suffixes, what):
    path = Path(value)
    if not path.is_absolute():
        raise EngineError("bad_arguments", f"the {what} path must be absolute", path=str(value))
    if os.path.islink(path):
        raise EngineError("bad_input", f"the {what} is a link; links are refused", path=str(path))
    if not path.is_file():
        raise EngineError("bad_input", f"the {what} does not exist or is not a file", path=str(path))
    if suffixes and path.suffix.lower() not in suffixes:
        raise EngineError("bad_input", f"the {what} must end with one of {', '.join(suffixes)}", path=str(path))
    return path.resolve()


def output_file(value, work, suffixes, what="output"):
    path = Path(value)
    if not path.is_absolute():
        raise EngineError("bad_arguments", f"the {what} path must be absolute", path=str(value))
    if not path.parent.is_dir():
        raise EngineError("bad_arguments", f"the folder of the {what} does not exist", path=str(path.parent))
    path = path.parent.resolve() / path.name  # resolved as the work folder is (junctions, mapped drives)
    if not same_or_inside(path, work) or path == work or path.name in ("", ".", ".."):
        raise EngineError("bad_arguments", f"the {what} must be inside the work folder", path=str(path))
    if path.suffix.lower() not in suffixes:
        raise EngineError("bad_arguments", f"the {what} must end with one of {', '.join(suffixes)}", path=str(path))
    if path.parent == work and (path.name.startswith(".") or path.name.lower() in ("tmp", "cache")):
        raise EngineError("bad_arguments", f"the {what} uses a name the engine reserves", path=str(path))
    if os.path.lexists(path):
        raise EngineError("bad_arguments", f"the {what} exists already; outputs are never overwritten", path=str(path))
    return path


def scratch_folder(work, name):
    """A fresh folder for one step's own temporary files, inside the work folder."""
    return Path(tempfile.mkdtemp(prefix=f".{name}-", dir=work))


def check_code(model):
    """The model's code and configs are in the engine folder; anything missing is a broken install."""
    from .weights import CONFIGS

    root = model_root(model)
    needed = {"skintokens": ["src/model/tokenrig.py"], "unirig": ["run.py", "src/data/extract.py"]}[model]
    missing = [rel for rel in needed + list(CONFIGS[model]) if not (root / rel).is_file()]
    if missing:
        raise EngineError("internal", f"the engine folder lacks {model}'s code", missing=missing)
    return root
