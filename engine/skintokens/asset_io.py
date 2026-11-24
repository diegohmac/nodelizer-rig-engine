"""SkinTokens' Asset to and from plain arrays, field for field, keeping each array's dtype.

The read step saves exactly what BpyParser.load returned; the rig step builds the same Asset back.
Names become text arrays and come back as Python lists of str, as BpyParser makes them; the armature
name comes back as a str. Nothing here imports Blender or PyTorch.
"""
import numpy as np

from .. import plainnpz

FIELDS = ("vertices", "faces", "vertex_normals", "face_normals", "vertex_bias", "face_bias", "mesh_names",
          "joint_names", "parents", "lengths", "matrix_world", "matrix_local", "matrix_basis", "armature_name",
          "skin")
SKELETON_FIELDS = ("joint_names", "parents", "lengths", "matrix_local", "matrix_basis", "armature_name", "skin")


def to_arrays(asset):
    arrays = {}
    for field in FIELDS:
        value = getattr(asset, field)
        if value is None:
            continue
        if field == "joint_names":
            value = plainnpz.texts(value)
        elif field == "armature_name":
            value = plainnpz.text(value)
        else:
            value = np.asarray(value)
            if value.dtype == object:
                raise TypeError(f"{field} would need pickle")
        arrays[field] = value
    return arrays


def asset_fields(arrays, with_skeleton=True):
    """Keyword arguments for Asset(...), from a read file's arrays."""
    fields = {}
    for field in FIELDS:
        if field not in arrays or (not with_skeleton and field in SKELETON_FIELDS):
            continue
        value = arrays[field]
        if field == "joint_names":
            value = plainnpz.as_texts(value)
        elif field == "armature_name":
            value = plainnpz.as_text(value)
        else:
            value = np.array(value)  # a writable copy
        fields[field] = value
    return fields
