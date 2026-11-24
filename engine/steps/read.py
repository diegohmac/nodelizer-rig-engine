"""read: Blender reads the model file; the chosen model's own loader turns it into plain arrays.

Loads Blender, never PyTorch. The arrays are exactly what that model's loader returns
(SkinTokens' BpyParser.load, or UniRig's extract with its simplification to 50,000 faces), so the
rig step sees what upstream code would have seen.
"""
from .. import ENGINE_VERSION, layout, plainnpz, workspace


def run(args, events):
    work = workspace.work_folder(args.work)
    workspace.set_environment(work)
    source = workspace.input_file(args.input, workspace.MODEL_FILE_SUFFIXES, "model file")
    output = workspace.output_file(args.output, work, (".npz",), "read file")
    workspace.check_code(args.model)
    if args.model == "skintokens":
        from ..skintokens import read as impl
    else:
        from ..unirig import read as impl
    events.progress("loading_file")
    arrays, info = impl.read(source, work, events)
    arrays.update(
        format=plainnpz.text(layout.READ_FORMAT),
        model=plainnpz.text(args.model),
        engine_version=plainnpz.text(ENGINE_VERSION),
        source_name=plainnpz.text(source.name),
    )
    events.progress("saving")
    plainnpz.save(output, arrays)
    layout.check_read(plainnpz.load(output, "read file"), args.model)
    return dict(model=args.model, output=str(output), **info)
