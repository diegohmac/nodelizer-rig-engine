"""UniRig, driven from our own code: read (Blender), rig (PyTorch, two passes), write (Blender).

Upstream runs five processes: extract, skeleton, extract again (of the skeleton .fbx the skeleton
step wrote), skin, merge. Two of those steps export through Blender in the middle of a PyTorch
process. Here every Blender part is a read or write step, so a full UniRig rig is two rounds:

    read  (input)        ->  rig --stage skeleton  ->  write  (skeleton .fbx, upstream's export)
    read  (skeleton .fbx) ->  rig --stage skin      ->  write  (skin .fbx export, then merge onto input)

which is upstream's five steps, in order, with the same files in between. Keeping the file's own
skeleton is the second round alone, on the person's file.
"""
import shutil

from ..events import EngineError

FACES_TARGET = 50000  # the kit's --faces_target_count, as UniRig's launch scripts use it
# UniRig's loaders (extract.load, merge.load) match file endings by case
LOADABLE = (".glb", ".obj", ".fbx", ".FBX")


def loadable(source, scratch):
    """The model file under a name UniRig's loaders take. A .GLB is copied (it holds everything in
    one file); other upper case endings would lose the files beside them, so they are refused."""
    if source.suffix in LOADABLE:
        return source
    if source.suffix.lower() == ".glb":
        copy = scratch / "model.glb"
        shutil.copyfile(source, copy)
        return copy
    raise EngineError("bad_input", f"UniRig reads {source.suffix} files only with a lower case ending",
                      path=str(source))
