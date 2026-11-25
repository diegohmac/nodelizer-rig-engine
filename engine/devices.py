"""Which device a rig runs on, and in which number type. Standard library only.

Three kinds: an NVIDIA card through CUDA (Windows, and Linux), the processor, and Apple's GPU through
PyTorch's MPS. `--device` takes `cuda`, `cuda:N` or just `N` (a CUDA card), `cpu` or `mps`; without it
each platform has its default: CUDA on Windows and Linux (the card with the most memory), DEFAULT_MAC
on a Mac.

On CUDA nothing about the models changes: they run as their authors wrote them, under
torch.autocast('cuda', bfloat16), and give the same rigs bit for bit as before this file existed.
Elsewhere the number type is the one measured best for that device (PRECISION; PROTOCOL.md, Devices).
UniRig needs CUDA: anywhere else it is refused up front (`model_unavailable`).
"""
from dataclasses import dataclass

from .events import EngineError

KINDS = ("cuda", "cpu", "mps")
# A Mac's default, from measurements on an M3 Max (PROTOCOL.md, Devices): the processor in float32 gives
# the same rig as the processor anywhere else, run after run, within 6 GB. MPS was faster but its
# skeletons differ from every other device's and it needed 15.6 GB.
DEFAULT_MAC = "cpu"
# The number type each device runs SkinTokens in. CUDA is upstream's own and never changes. MPS in
# float32 ran out of memory; bfloat16 is the one that finished there.
PRECISION = {"cuda": "bfloat16", "cpu": "float32", "mps": "bfloat16"}
PRECISIONS = ("bfloat16", "float32")
# The models each kind of device can run.
MODELS = {"cuda": ("skintokens", "unirig"), "cpu": ("skintokens",), "mps": ("skintokens",)}


@dataclass(frozen=True)
class Device:
    kind: str           # cuda, cpu or mps
    index: object       # the CUDA card's number, or None (CUDA: the card with the most memory)
    precision: str      # bfloat16 or float32
    explicit: bool      # given with --device


def parse(value):
    """--device's value as (kind, index). Raises bad_arguments for anything else."""
    text = str(value).strip().lower()
    if text.isdigit():
        return "cuda", int(text)
    if text in ("cpu", "mps", "cuda"):
        return text, None
    if text.startswith("cuda:") and text[5:].isdigit():
        return "cuda", int(text[5:])
    raise EngineError("bad_arguments", "--device takes cuda, cuda:N, a card's number, cpu or mps")


def choose(value, platform, precision=None):
    """The device for this run: --device if given, else the platform's default."""
    if value is None:
        kind, index, explicit = (DEFAULT_MAC if platform == "darwin" else "cuda"), None, False
    else:
        (kind, index), explicit = parse(value), True
    if kind == "mps" and platform != "darwin":
        raise EngineError("bad_arguments", "the mps device is Apple's GPU, on a Mac only")
    if precision is not None and kind == "cuda" and precision != PRECISION["cuda"]:
        raise EngineError("bad_arguments", "on CUDA the models run in their own bfloat16; --precision is for cpu "
                                           "and mps")
    if precision is not None and precision not in PRECISIONS:
        raise EngineError("bad_arguments", f"--precision takes {' or '.join(PRECISIONS)}")
    return Device(kind, index, precision or PRECISION[kind], explicit)


def require_model(model, kind, platform):
    """UniRig needs an NVIDIA card: refused on a Mac in every step, and on cpu or mps anywhere."""
    if platform == "darwin" and model not in MODELS["mps"]:
        raise EngineError("model_unavailable", f"{model} needs an NVIDIA card; on a Mac the engine has SkinTokens only")
    if kind is not None and model not in MODELS[kind]:
        raise EngineError("model_unavailable", f"{model} needs an NVIDIA card (CUDA); it does not run on {kind}")
