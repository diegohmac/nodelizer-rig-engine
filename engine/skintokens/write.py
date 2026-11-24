"""SkinTokens' write: transfer_rigging, as its helper ran it for st_rig.py and demo.py. Blender only.

The source Asset is rebuilt from the rig file with the arrays predict_step made (vertices,
matrix_local, parents, skin) and the final bone names. transfer_rigging then loads the original file,
maps the joints onto it with the similarity transform it computes, carries the skin over by nearest
vertex, keeps the 4 largest weights per vertex and exports a .glb.
"""
import os
import random
import shutil

import numpy as np

from .. import modelpath, plainnpz, workspace
from ..events import EngineError


def write(arrays, stage, source, output, work, events):
    modelpath.use("skintokens")
    from src.rig_package.info.asset import Asset
    from src.rig_package.parser.bpy import transfer_rigging

    seed = int(arrays["seed"][()])
    asset = Asset(
        vertices=np.array(arrays["vertices"]),
        matrix_local=np.array(arrays["matrix_local"]),
        parents=np.array(arrays["parents"]),
        joint_names=plainnpz.as_texts(arrays["names"]),
        skin=np.array(arrays["skin"]),
    )
    # transfer_rigging nudges bones that sit on their parent with np.random; seeded, the file repeats.
    random.seed(seed)
    np.random.seed(seed)
    scratch = workspace.scratch_folder(work, "write")
    try:
        made = scratch / ("rigged" + output.suffix.lower())
        try:
            transfer_rigging(source_asset=asset, target_path=str(source), export_path=str(made), group_per_vertex=4)
        except Exception as error:
            raise EngineError("write_failed", f"{type(error).__name__}: {error}"[:2000])
        if not made.is_file() or made.stat().st_size == 0:
            raise EngineError("write_failed", "Blender wrote no file")
        os.replace(made, output)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {"names": plainnpz.as_text(arrays["names_source"]), "skeleton": plainnpz.as_text(arrays["skeleton"])}
