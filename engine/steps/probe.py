"""probe: the cards, their free video memory, and whether each model can run. Never loads Blender.

Light (--light): the cards and their free memory, from NVML when it loads (about 0.2 seconds, no video
memory used), else from PyTorch. Full: PyTorch's view of the cards too, each model's code, its weights
folder when given (--weights MODEL=FOLDER) and, with --verify-weights, every weight file's SHA-256.

On a Mac there is one device, the processor, with the computer's memory, and the memory mode a rig
would run in; --light answers without PyTorch. Only SkinTokens is listed (UniRig needs an NVIDIA card),
with its needs for that mode, light or not.
"""
import sys
import time

from .. import ENGINE_VERSION, PROTOCOL_VERSION, devices as device_kinds, weights, workspace
from ..events import EngineError, log
from . import torchenv


def _weights_folders(values):
    folders = {}
    for value in values:
        model, sep, folder = value.partition("=")
        if not sep or model not in workspace.MODELS:
            raise EngineError("bad_arguments", "--weights takes MODEL=FOLDER, with MODEL skintokens or unirig")
        folders[model] = folder
    return folders


def run(args, events):
    if args.work:
        workspace.set_environment(workspace.work_folder(args.work))
    folders = _weights_folders(args.weights)
    info = {"engine_version": ENGINE_VERSION, "protocol": PROTOCOL_VERSION, "light": bool(args.light),
            "package": {k: workspace.manifest().get(k) for k in ("engine_version", "protocol", "python")}}
    if sys.platform == "darwin":
        return _mac(args, events, folders, info)
    started = time.monotonic()
    try:
        devices, driver = torchenv.nvml_devices()
        info.update(devices_from="nvml", driver=driver)
    except OSError as error:
        log(f"NVML: {error}; asking PyTorch")
        devices, driver = None, None
    info["nvml_seconds"] = round(time.monotonic() - started, 2)
    torch = None
    if devices is None or not args.light:
        events.progress("loading_libraries")
        torch = torchenv.import_torch()
        info.update(torchenv.versions(torch), cuda_available=bool(torch.cuda.is_available()))
        if info["cuda_available"]:
            seen = torchenv.torch_devices(torch, with_free=devices is None)
            if devices is None:
                devices, info["devices_from"] = seen, "torch"
            else:  # NVML gave the free memory; PyTorch adds what it knows
                by_index = {d["index"]: d for d in seen}
                for device in devices:
                    device["capability"] = by_index.get(device["index"], {}).get("capability")
                if len(seen) != len(devices):
                    info["torch_devices"] = len(seen)
        elif devices is None:
            devices, info["devices_from"] = [], "torch"
    else:
        info.update(torchenv.versions())
    info["devices"] = devices
    info["default_device"] = torchenv.choose(devices, None) if devices else None
    if args.light:
        return info

    info["models"] = _models(args, events, folders, workspace.MODELS, weights.NEEDS_FREE_BYTES, devices,
                             info["default_device"])
    return info


def _mac(args, events, folders, info):
    """A Mac: one device, the processor, with the computer's memory, from the system (engine/host.py).
    `--light` never loads PyTorch; without it PyTorch is loaded for its version."""
    from .. import host

    mode = host.memory_mode()
    facts = host.facts()
    info.update(devices_from="host", cuda_available=False)
    if args.light:
        info.update(torchenv.versions())
    else:
        events.progress("loading_libraries")
        info.update(torchenv.versions(torchenv.import_torch()))
    kind = device_kinds.DEFAULT_MAC
    info["devices"] = [{"index": 0, "name": facts["name"], "total_bytes": facts["total_bytes"],
                        "free_bytes": facts["free_bytes"], "capability": f"{kind}:{mode}"}]
    info["default_device"] = 0
    needs = {model: weights.needs_host(model, mode) for model in device_kinds.MODELS[kind]}
    info["models"] = _models(args, events, folders, device_kinds.MODELS[kind], needs, info["devices"], 0)
    return info


def _models(args, events, folders, models_here, needs, devices, default_device):
    models = {}
    for model in models_here:
        entry = {"needs_free_bytes": needs[model]}
        try:
            workspace.check_code(model)
            entry["code"] = True
        except EngineError as error:
            entry.update(code=False, missing_code=error.details.get("missing", []))
        if model in folders:
            try:
                path = weights.folder(model, folders[model])
                entry["weights"] = True
                if args.verify_weights:
                    events.progress("verifying_weights", model=model)
                    entry["verified"] = weights.verify(model, path)
                    entry["weights"] = all(f["ok"] for f in entry["verified"].values())
            except EngineError as error:
                entry.update(weights=False, missing_weights=error.details.get("missing", []))
        default = next((d for d in devices if d["index"] == default_device), None)
        if default is not None and default.get("free_bytes") is not None:
            entry["normal_speed_on_default_device"] = default["free_bytes"] >= entry["needs_free_bytes"]
        models[model] = entry
    return models
