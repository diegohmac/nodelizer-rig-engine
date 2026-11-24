"""Rig files made from existing rigs, so the write step can be tested without a graphics card: a rig
read back from a rigged file is turned into exactly the arrays the rig step would have kept."""
import json

import numpy as np

from conftest import REPO

import sys

sys.path.insert(0, str(REPO))
from engine import ENGINE_VERSION, layout, plainnpz  # noqa: E402


def _common(model, stage, names, parents, joints, vertices, skin, seed, skeleton, names_source):
    arrays = {
        "format": plainnpz.text(layout.RIG_FORMAT),
        "model": plainnpz.text(model),
        "engine_version": plainnpz.text(ENGINE_VERSION),
        "stage": plainnpz.text(stage),
        "seed": np.int64(seed),
        "skeleton": plainnpz.text(skeleton),
        "names_source": plainnpz.text(names_source),
        "names": plainnpz.texts(names),
        "parents": np.asarray(parents, dtype=np.int32),
        "joints": np.asarray(joints),
        "vertices": np.asarray(vertices),
    }
    if skin is not None:
        arrays["skin"] = np.asarray(skin)
    return arrays


def skintokens_rig_from_read(read, names=None, seed=12345, skeleton="made", names_source="numbered"):
    """SkinTokens' rig arrays, as predict_step would make them, from a SkinTokens read of a rigged file:
    vertices, matrix_local with the joints (float32, rotation left as identity, as Asset.from_data does)
    and the per vertex skin."""
    joints = np.asarray(read["matrix_local"])[:, :3, 3]
    matrix_local = np.zeros((len(joints), 4, 4), dtype=np.float32)
    matrix_local[...] = np.eye(4)
    matrix_local[:, :3, 3] = joints
    names = list(names) if names is not None else plainnpz.as_texts(read["joint_names"])
    arrays = _common("skintokens", "final", names, read["parents"], matrix_local[:, :3, 3], read["vertices"],
                     read["skin"], seed, skeleton, names_source)
    arrays["matrix_local"] = matrix_local
    return arrays


def skintokens_rig_onto(target, rigged, **kwargs):
    """As skintokens_rig_from_read, but on the target's own vertices (as predict_step's asset is, so
    transfer_rigging maps the joints with an exact fit), the skin taken over by nearest vertex."""
    from scipy.spatial import cKDTree

    arrays = skintokens_rig_from_read(rigged, **kwargs)
    _, nearest = cKDTree(np.asarray(rigged["vertices"])).query(np.asarray(target["vertices"]))
    arrays["vertices"] = np.asarray(target["vertices"])
    arrays["skin"] = np.asarray(rigged["skin"])[nearest]
    return arrays


def unirig_rig_from_read(read, stage, names=None, seed=12345, skeleton="made", names_source="numbered",
                         options=None):
    """UniRig's recorded _export_fbx arguments from a UniRig read of a skeleton or skin .fbx. The skin
    pass passes tails and the skin; the skeleton pass neither."""
    names = list(names) if names is not None else plainnpz.as_texts(read["names"])
    skin = read["skin"] if stage == "final" else None
    if stage == "final" and skin is not None:
        skin = skin / np.maximum(skin.sum(1, keepdims=True), 1e-9)
    arrays = _common("unirig", stage, names, read["parents"], read["joints"], read["vertices"], skin, seed,
                     skeleton, names_source)
    arrays["faces"] = np.asarray(read["faces"])
    if stage == "final":
        arrays["tails"] = np.asarray(read["tails"])
        default = dict(extrude_size=0.03, group_per_vertex=4, add_root=False, do_not_normalize=False,
                       use_extrude_bone=False, use_connect_unique_child=False, extrude_from_parent=True)
    else:
        default = dict(extrude_size=0.03, group_per_vertex=-1, add_root=False, do_not_normalize=False,
                       use_extrude_bone=True, use_connect_unique_child=True, extrude_from_parent=True)
    default.update(options or {})
    arrays["export_options"] = plainnpz.text(json.dumps(default, sort_keys=True))
    return arrays


def save(path, arrays):
    layout.check_rig(arrays)
    plainnpz.save(path, arrays)
    return path


def load(path):
    return plainnpz.load(path)
