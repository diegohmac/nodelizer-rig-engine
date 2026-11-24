"""rig: the model makes the rig from a read file. Loads PyTorch, never Blender.

Free and total video memory are reported before any model loads (a `card` progress line), since
Windows never raises out of memory on a full card: it spills into system memory and gets slow.
"""
import numpy as np

from .. import ENGINE_VERSION, layout, plainnpz, seeding, weights, workspace
from ..events import EngineError
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
    if args.device is not None and args.device < 0:
        raise EngineError("bad_arguments", "--device is a card's number, 0 or more")

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
    card = torchenv.card(torch, args.device)
    events.progress("card", **card)
    device = card["device"]

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
    info["memory"] = dict(card, needs_free_bytes=weights.NEEDS_FREE_BYTES[model],
                          peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                          peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
    if cache is not None:
        info["cache"] = cache_state
    return dict(model=model, output=str(output), seed=seed, **torchenv.versions(torch), **info)
