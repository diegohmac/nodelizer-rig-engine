"""rig: the model makes the rig from a read file. Loads PyTorch, never Blender.

Free and total memory are reported before any model loads (a `card` progress line): on an NVIDIA card
its video memory, since Windows never raises out of memory on a full card (it spills into system memory
and gets slow); on the processor or a Mac's GPU, the computer's memory (engine/host.py).
"""
import sys

import numpy as np

from .. import ENGINE_VERSION, devices, layout, plainnpz, seeding, weights, workspace
from ..events import EngineError, log
from . import torchenv


def run(args, events):
    try:
        seed = seeding.check(args.seed)
    except ValueError as error:
        raise EngineError("bad_arguments", str(error))
    model = args.model
    if model == "unirig" and args.stage is None:
        raise EngineError("bad_arguments", "UniRig needs --stage skeleton or --stage skin")
    if model == "skintokens" and args.stage is not None:
        raise EngineError("bad_arguments", "--stage is for UniRig only")
    if model == "unirig" and args.keep_skeleton and args.stage != "skin":
        raise EngineError("bad_arguments", "keeping the file's skeleton is UniRig's skin pass alone (--stage skin)")
    chosen = devices.choose(args.device, sys.platform, args.precision)
    devices.require_model(model, chosen.kind, sys.platform)
    if args.memory is not None and sys.platform != "darwin":
        raise EngineError("bad_arguments", "--memory is for a Mac")
    mode = None
    if sys.platform == "darwin":
        from .. import host

        mode = host.running_mode()  # what the entry point started this process in (rigengine.py)
        if args.memory is not None and args.memory != mode:
            log(f"--memory {args.memory} was asked but the step could not start again in it; it runs {mode}")

    work = workspace.work_folder(args.work)
    cache = workspace.cache_folder(args.cache)
    workspace.set_environment(work, cache)
    cache_state = workspace.claim_cache(cache)
    read_path = workspace.input_file(args.input, (".npz",), "read file")
    output = workspace.output_file(args.output, work, (".npz",), "rig file")
    arrays = plainnpz.load(read_path, "read file")
    layout.check_read(arrays, model)
    root = workspace.check_code(model)
    weights_folder = weights.folder(model, args.weights)

    events.progress("loading_libraries")
    torch = torchenv.import_torch()
    if chosen.kind == "cuda":
        card = torchenv.card(torch, chosen.index)
        device = card["device"]
    else:  # the processor or Apple's GPU: the device is named, not numbered
        card = torchenv.host_card(torch, chosen.kind, mode)
        device = chosen
    events.progress("card", **card)
    sampler = torchenv.MpsMemory(torch) if chosen.kind == "mps" else None

    if model == "skintokens":
        from ..skintokens import rig as impl

        rig_arrays, info = impl.rig(root, weights_folder, read_path, arrays, seed, args.keep_skeleton,
                                    args.standard_names, torch, device, events)
    else:
        from ..unirig import rig as impl

        rig_arrays, info = impl.rig(root, weights_folder, read_path, arrays, seed, args.stage, args.keep_skeleton,
                                    args.standard_names, torch, device, work, events)

    rig_arrays.update(
        format=plainnpz.text(layout.RIG_FORMAT),
        model=plainnpz.text(model),
        engine_version=plainnpz.text(ENGINE_VERSION),
        seed=np.int64(seed),
    )
    events.progress("saving")
    plainnpz.save(output, rig_arrays)
    layout.check_rig(plainnpz.load(output, "rig file"))
    if chosen.kind == "cuda":
        info["memory"] = dict(card, needs_free_bytes=weights.NEEDS_FREE_BYTES[model],
                              peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                              peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
    else:  # PROTOCOL.md, Devices: what the two peaks mean off CUDA
        from .. import host

        peak = host.peak_footprint()
        info["memory"] = dict(card, needs_free_bytes=weights.needs_host(model, mode),
                              peak_allocated_bytes=peak,
                              peak_reserved_bytes=sampler.stop() if sampler is not None else peak)
    if cache is not None:
        info["cache"] = cache_state
    return dict(model=model, output=str(output), seed=seed, **torchenv.versions(torch), **info)
