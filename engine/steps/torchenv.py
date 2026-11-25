"""The cards: through NVML (the driver's own library, no CUDA context, no video memory) where it loads,
and through PyTorch for the rig step.

Numbering: the engine sets CUDA_DEVICE_ORDER=PCI_BUS_ID before PyTorch loads, so CUDA's device N is
NVML's and nvidia-smi's device N. Without it CUDA would put the fastest card first.
"""
import ctypes
import os
import sys

from ..events import EngineError


def import_torch():
    # Both models import flash_attn, which has no Windows build. The stand in sits in the engine
    # folder (standin/), and Python runs with -I, so PYTHONPATH can't put it on the path. Appended,
    # never first, so it shadows nothing. Only the steps that load PyTorch get here.
    standin = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "standin")
    if os.path.isdir(standin) and standin not in sys.path:
        sys.path.append(standin)
    import torch

    if os.name == "nt":  # spconv, torch_scatter and torch_cluster link CUDA files that sit in torch\lib
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
    return torch


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def nvml_devices():
    """[{index, name, total_bytes, free_bytes}] and the driver version, or raises OSError."""
    names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                      "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
        else ["libnvidia-ml.so.1"]
    library = None
    for name in names:
        try:
            library = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if library is None:
        raise OSError("NVML is not available")
    if library.nvmlInit_v2() != 0:
        raise OSError("NVML would not start")
    try:
        count = ctypes.c_uint()
        if library.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            raise OSError("NVML could not count the cards")
        devices = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                raise OSError(f"NVML could not open card {index}")
            name = ctypes.create_string_buffer(96)
            library.nvmlDeviceGetName(handle, name, 96)
            memory = _Memory()
            if library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                raise OSError(f"NVML could not read card {index}'s memory")
            devices.append({"index": index, "name": name.value.decode("utf-8", "replace"),
                            "total_bytes": int(memory.total), "free_bytes": int(memory.free)})
        driver = ctypes.create_string_buffer(80)
        library.nvmlSystemGetDriverVersion(driver, 80)
        return devices, driver.value.decode("ascii", "replace")
    finally:
        library.nvmlShutdown()


def torch_devices(torch, with_free=True):
    """The cards PyTorch sees. Reading free memory makes a CUDA context on each card (a few hundred MB
    each while the process lives)."""
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        entry = {"index": index, "name": props.name, "total_bytes": int(props.total_memory),
                 "capability": f"{props.major}.{props.minor}"}
        if with_free:
            free, _total = torch.cuda.mem_get_info(index)
            entry["free_bytes"] = int(free)
        devices.append(entry)
    return devices


def choose(devices, asked):
    """--device as asked, or the card with the most memory (the lowest number on a tie)."""
    if not devices:
        raise EngineError("no_cuda", "PyTorch sees no CUDA card", devices=[])
    if asked is None:
        return max(devices, key=lambda d: (d["total_bytes"], -d["index"]))["index"]
    if not any(d["index"] == asked for d in devices):
        raise EngineError("no_cuda", f"there is no CUDA device {asked}", devices=devices)
    return asked


def card(torch, asked):
    """The card the rig step uses, made current, and its free memory read before any model loads."""
    try:
        if not torch.cuda.is_available():
            raise EngineError("no_cuda", "PyTorch sees no CUDA card", torch=torch.__version__,
                              built_for=getattr(torch.version, "cuda", None))
        devices = torch_devices(torch, with_free=False)
        index = choose(devices, asked)
        torch.cuda.set_device(index)
        free, total = torch.cuda.mem_get_info(index)
    except EngineError:
        raise
    except Exception as error:  # a driver too old for this PyTorch, a card it has no code for, ...
        raise EngineError("no_cuda", f"PyTorch could not use the card: {type(error).__name__}: {error}"[:2000])
    chosen = next(d for d in devices if d["index"] == index)
    return {"device": index, "card": chosen["name"], "free_bytes": int(free), "total_bytes": int(total),
            "capability": chosen["capability"], "devices": len(devices)}


def host_card(torch, kind, mode=None):
    """The `card` facts for the processor or a Mac's GPU, in the CUDA card's shape (PROTOCOL.md, Devices):
    device 0, the chip's name, the computer's memory (a Mac's GPU shares it), `capability` the kind and on
    a Mac its memory mode (`cpu:lean`)."""
    from .. import host

    if kind == "mps" and not torch.backends.mps.is_available():
        raise EngineError("no_cuda", "PyTorch cannot use this Mac's GPU (MPS)", torch=torch.__version__,
                          built_for=None)
    facts = host.facts()
    name = f"{facts['name']} GPU" if kind == "mps" else facts["name"]
    return {"device": 0, "card": name, "free_bytes": facts["free_bytes"], "total_bytes": facts["total_bytes"],
            "capability": f"{kind}:{mode}" if mode else kind, "devices": 1}


class MpsMemory:
    """The most memory Metal held for PyTorch while the step ran, read four times a second (PyTorch keeps
    no peak for MPS)."""

    def __init__(self, torch):
        import threading

        self._torch = torch
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="mps-memory", daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            try:
                self.peak = max(self.peak, int(self._torch.mps.driver_allocated_memory()))
            except Exception:
                return
            if self._stop.wait(0.25):
                return

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak = max(self.peak, int(self._torch.mps.driver_allocated_memory()))
        return self.peak


def versions(torch=None):
    found = {"python": sys.version.split()[0]}
    if torch is not None:
        found.update(torch=torch.__version__, cuda=getattr(torch.version, "cuda", None))
    return found
