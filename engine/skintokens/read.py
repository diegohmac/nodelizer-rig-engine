"""SkinTokens' read: BpyParser.load, as its helper and demo.py run it. Blender only."""
import numpy as np

from .. import modelpath
from ..events import EngineError
from . import asset_io


def read(source, work, events):
    modelpath.use("skintokens")
    from src.rig_package.parser.bpy import BpyParser

    try:
        asset = BpyParser.load(filepath=str(source))
    except ValueError as error:
        if "at least one array" in str(error):
            raise EngineError("no_mesh", "the file has no mesh")
        raise EngineError("read_failed", f"{type(error).__name__}: {error}"[:2000])
    except Exception as error:
        raise EngineError("read_failed", f"{type(error).__name__}: {error}"[:2000])
    if asset.vertices is None or len(asset.vertices) == 0 or asset.faces is None or len(asset.faces) == 0:
        raise EngineError("no_mesh", "the file has no mesh with faces")
    import bpy

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    arrays = asset_io.to_arrays(asset)
    has_skeleton = asset.parents is not None
    bones = int(len(asset.parents)) if has_skeleton else 0  # BpyParser takes the first armature's bones
    arrays["has_skeleton"] = np.bool_(has_skeleton)
    arrays["bone_count"] = np.int64(bones)
    info = dict(has_skeleton=has_skeleton, bone_count=bones, armatures=len(armatures),
                armature_bones=sum(len(obj.data.bones) for obj in armatures), vertices=int(asset.N),
                faces=int(asset.F), meshes=int(asset.P))
    return arrays, info
