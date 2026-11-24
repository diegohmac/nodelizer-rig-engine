"""UniRig's write. Blender only.

Skeleton pass: Exporter._export_fbx with the recorded arguments, as UniRig's skeleton writer calls it,
which writes the skeleton .fbx (the simplified mesh and the made bones). The next round reads it.

Skin pass: the same call as UniRig's skin writer (the skin .fbx, in a scratch folder), then UniRig's
merge (src/inference/merge.py, transfer) of that file onto the original model file, which exports the
rigged .glb. That is upstream's skin export and merge step, in one process.
"""
import os
import random
import shutil

import numpy as np

from .. import modelpath, plainnpz, workspace
from ..events import EngineError
from . import capture, loadable


def write(arrays, stage, source, output, work, events):
    modelpath.use("unirig")
    from src.data.exporter import Exporter

    seed = int(arrays["seed"][()])
    call = capture.to_call(arrays)
    scratch = workspace.scratch_folder(work, "write")
    try:
        if stage == "skeleton":
            made = scratch / "skeleton.fbx"
            try:
                Exporter()._export_fbx(path=str(made), **call)
            except Exception as error:
                raise EngineError("write_failed", f"{type(error).__name__}: {error}"[:2000])
        else:
            from src.inference.merge import transfer

            skin_file = scratch / "skin.fbx"
            made = scratch / "rigged.glb"
            target = loadable(source, scratch)
            try:
                Exporter()._export_fbx(path=str(skin_file), **call)
                # merge picks the axis order with a random subset of points; seeded, the file repeats
                random.seed(seed)
                np.random.seed(seed)
                transfer(source=str(skin_file), target=str(target), output=str(made), add_root=False)
            except Exception as error:
                raise EngineError("write_failed", f"{type(error).__name__}: {error}"[:2000])
        if not made.is_file() or made.stat().st_size == 0:
            raise EngineError("write_failed", "Blender wrote no file; the log has UniRig's message")
        os.replace(made, output)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {"names": plainnpz.as_text(arrays["names_source"]), "skeleton": plainnpz.as_text(arrays["skeleton"])}
