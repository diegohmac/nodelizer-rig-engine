"""UniRig's read: its extract step for one file, as extract_builtin runs it. Blender only.

The arrays are what extract saves as raw_data.npz (mesh simplified to 50,000 faces, normals by
trimesh, the skeleton in UniRig's own bone order), written as plain arrays.
"""
import shutil

import numpy as np

from .. import modelpath, plainnpz, workspace
from ..events import EngineError
from . import FACES_TARGET, loadable


def read(source, work, events):
    modelpath.use("unirig")
    from src.data import log as unirig_log
    from src.data.extract import clean_bpy, get_arranged_bones, load, process_armature, process_mesh, save_raw_data

    scratch = workspace.scratch_folder(work, "read")
    try:
        # extract's helpers write warnings to a log file; it lives in the scratch folder
        unirig_log.new_log(str(scratch), "extract")
        unirig_log.new_entry(str(source))
        target = loadable(source, scratch)
        clean_bpy()
        try:
            armature = load(str(target))
        except Exception as error:
            raise EngineError("read_failed", f"{type(error).__name__}: {error}"[:2000])
        import bpy

        armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
        bone_count = len(armature.data.bones) if armature is not None else 0
        try:
            arranged = get_arranged_bones(armature) if armature is not None else None
            vertices, faces, skin = process_mesh(arranged)
        except ValueError as error:
            if "at least one array" in str(error):
                raise EngineError("no_mesh", "the file has no mesh with faces")
            raise EngineError("read_failed", f"{type(error).__name__}: {error}"[:2000])
        except Exception as error:
            raise EngineError("read_failed", f"{type(error).__name__}: {error}"[:2000])
        if armature is not None:
            joints, tails, parents, names, matrix_local = process_armature(armature, arranged)
        else:
            joints = tails = parents = names = matrix_local = None
        raw = scratch / "raw_data.npz"
        save_raw_data(path=str(raw), vertices=vertices, faces=faces - 1, skin=skin, joints=joints, tails=tails,
                      parents=parents, names=names, matrix_local=matrix_local, target_count=FACES_TARGET)
        arrays = plainnpz.load(raw, "UniRig's raw data")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for key in ("path", "cls", "no_skin"):  # set again when UniRig loads the file
        arrays.pop(key, None)
    has_skeleton = "joints" in arrays
    arrays["has_skeleton"] = np.bool_(has_skeleton)
    arrays["bone_count"] = np.int64(len(arrays["joints"]) if has_skeleton else 0)
    arrays["armature_bone_count"] = np.int64(bone_count)
    info = dict(has_skeleton=has_skeleton, bone_count=int(arrays["bone_count"]), armatures=len(armatures),
                armature_bones=bone_count, vertices=len(arrays["vertices"]), faces=len(arrays["faces"]))
    return arrays, info
