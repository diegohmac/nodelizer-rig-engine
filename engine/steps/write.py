"""write: Blender puts the rig onto the original model file and writes the rigged file.

Loads Blender, never PyTorch. It runs the model's own export with the arrays the rig step kept, so the
result is what the model's own pipeline gives: SkinTokens' transfer_rigging, or UniRig's skin export
followed by its merge. Original mesh, materials and textures come from the original file.
"""
import sys

from .. import devices, glbinfo, layout, plainnpz, weights, workspace
from ..events import EngineError


def run(args, events):
    work = workspace.work_folder(args.work)
    workspace.set_environment(work)
    rig_path = workspace.input_file(args.rig, (".npz",), "rig file")
    arrays = plainnpz.load(rig_path, "rig file")
    model, stage = layout.check_rig(arrays)
    devices.require_model(model, None, sys.platform)
    if stage == "final":
        if not args.input:
            raise EngineError("bad_arguments", "--input, the original model file, is needed")
        source = workspace.input_file(args.input, workspace.MODEL_FILE_SUFFIXES, "model file")
        output = workspace.output_file(args.output, work, (".glb",))
    else:
        if args.input:
            raise EngineError("bad_arguments", "UniRig's skeleton pass writes from the rig file alone; drop --input")
        source = None
        output = workspace.output_file(args.output, work, (".fbx",))
    workspace.check_code(model)
    if model == "skintokens":
        from ..skintokens import write as impl
    else:
        from ..unirig import write as impl
    events.progress("writing")
    info = impl.write(arrays, stage, source, output, work, events)
    try:
        result = dict(model=model, stage=stage, output=str(output), bytes=output.stat().st_size,
                      sha256=weights.sha256(output), **info)
        if output.suffix.lower() == ".glb":
            result.update(_check_glb(output, plainnpz.as_texts(arrays["names"])))
    except BaseException:
        output.unlink(missing_ok=True)  # a file that failed its check is never left for the app
        raise
    return result


def _check_glb(path, names):
    """Every bone of the rig is in the file, under its name. Blender's glTF export adds one bone of its
    own, "neutral_bone", when some vertices have no weight at all; that is reported, not refused."""
    summary = glbinfo.summary(path)
    joints = summary["joints"]
    extra = sorted(set(joints) - set(names))
    if summary["skins"] != 1 or sorted(set(names) - set(joints)) or len(joints) != len(set(joints)) \
            or extra not in ([], ["neutral_bone"]):
        raise EngineError("write_failed", "the written file does not hold the rig's bones",
                          bones_written=len(joints), bones_expected=len(names), extra_bones=extra)
    return dict(bones=len(names), extra_bones=extra, meshes=summary["meshes"], materials=summary["materials"],
                images=summary["images"])
