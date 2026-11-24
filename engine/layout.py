"""The two .npz layouts that pass between steps, and their checks. PROTOCOL.md documents them.

A read file (`rig-engine/read/1`) holds what Blender read from the model file, in exactly the form
the chosen model's own loader produced it, so the rig step sees the same arrays upstream code would
have seen. A rig file (`rig-engine/rig/1`) holds the made rig and exactly what the write step needs to
run the model's own export.
"""
import numpy as np

from .events import EngineError

READ_FORMAT = "rig-engine/read/1"
RIG_FORMAT = "rig-engine/rig/1"

# kind letters: f float, i signed int, u unsigned, b bool, U text. (key, kinds, ndim, required)
_READ_COMMON = [
    ("format", "U", 0, True), ("model", "U", 0, True), ("engine_version", "U", 0, True),
    ("source_name", "U", 0, True), ("has_skeleton", "b", 0, True), ("bone_count", "iu", 0, True),
]
_READ_SKINTOKENS = [  # SkinTokens' Asset, as BpyParser.load makes it
    ("vertices", "f", 2, True), ("faces", "iu", 2, True), ("vertex_normals", "f", 2, False),
    ("face_normals", "f", 2, False), ("vertex_bias", "iu", 1, False), ("face_bias", "iu", 1, False),
    ("mesh_names", "U", 1, False), ("joint_names", "U", 1, False), ("parents", "iu", 1, False),
    ("lengths", "f", 1, False), ("matrix_world", "f", 2, False), ("matrix_local", "f", 3, False),
    ("matrix_basis", "f", 4, False), ("armature_name", "U", 0, False), ("skin", "f", 2, False),
]
_READ_UNIRIG = [  # UniRig's RawData, as its extract step makes it
    ("vertices", "f", 2, True), ("vertex_normals", "f", 2, True), ("faces", "iu", 2, True),
    ("face_normals", "f", 2, True), ("joints", "f", 2, False), ("tails", "f", 2, False),
    ("skin", "f", 2, False), ("parents", "iu", 1, False), ("names", "U", 1, False),
    ("matrix_local", "f", 3, False), ("armature_bone_count", "iu", 0, True),
]

_RIG_COMMON = [
    ("format", "U", 0, True), ("model", "U", 0, True), ("engine_version", "U", 0, True),
    ("stage", "U", 0, True), ("seed", "iu", 0, True), ("skeleton", "U", 0, True),
    ("names_source", "U", 0, True), ("names", "U", 1, True), ("parents", "iu", 1, True),
    ("joints", "f", 2, True), ("vertices", "f", 2, True), ("skin", "f", 2, False),
]
_RIG_SKINTOKENS = [("matrix_local", "f", 3, True)]
_RIG_UNIRIG = [("faces", "iu", 2, True), ("tails", "f", 2, False), ("export_options", "U", 0, True)]

STAGES = ("final", "skeleton")
SKELETON = ("made", "kept")
NAMES_SOURCES = ("kept", "standard", "numbered", "model")


def _check_fields(arrays, fields, what):
    for key, kinds, ndim, required in fields:
        if key not in arrays:
            if required:
                raise EngineError("bad_input", f"the {what} lacks {key}")
            continue
        value = arrays[key]
        if value.dtype.kind not in kinds or value.ndim != ndim:
            raise EngineError("bad_input", f"the {what} has {key} as {value.dtype}, {value.ndim} dimensions; "
                              f"expected kind {kinds}, {ndim} dimensions")


def _text(arrays, key):
    return str(arrays[key][()])


def check_read(arrays, model=None):
    what = "read file"
    _check_fields(arrays, _READ_COMMON, what)
    if _text(arrays, "format") != READ_FORMAT:
        raise EngineError("bad_input", f"the {what} is not {READ_FORMAT}")
    found = _text(arrays, "model")
    if found not in ("skintokens", "unirig"):
        raise EngineError("bad_input", f"the {what} names an unknown model {found!r}")
    if model is not None and found != model:
        raise EngineError("bad_input", f"the {what} was read for {found}, not {model}; read the file again")
    _check_fields(arrays, _READ_SKINTOKENS if found == "skintokens" else _READ_UNIRIG, what)
    vertices, faces = arrays["vertices"], arrays["faces"]
    if vertices.shape[1:] != (3,) or faces.shape[1:] != (3,) or len(vertices) == 0 or len(faces) == 0:
        raise EngineError("bad_input", f"the {what} has no usable mesh")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise EngineError("bad_input", f"the {what} has faces pointing outside its vertices")
    names_key = "joint_names" if found == "skintokens" else "names"
    if bool(arrays["has_skeleton"][()]):
        for key in (names_key, "parents"):
            if key not in arrays:
                raise EngineError("bad_input", f"the {what} says it has a skeleton but lacks {key}")
        if len(arrays[names_key]) != len(arrays["parents"]):
            raise EngineError("bad_input", f"the {what} has {len(arrays[names_key])} names for "
                              f"{len(arrays['parents'])} bones")
    return found


def check_rig(arrays):
    what = "rig file"
    _check_fields(arrays, _RIG_COMMON, what)
    if _text(arrays, "format") != RIG_FORMAT:
        raise EngineError("bad_input", f"the {what} is not {RIG_FORMAT}")
    model = _text(arrays, "model")
    if model not in ("skintokens", "unirig"):
        raise EngineError("bad_input", f"the {what} names an unknown model {model!r}")
    _check_fields(arrays, _RIG_SKINTOKENS if model == "skintokens" else _RIG_UNIRIG, what)
    stage = _text(arrays, "stage")
    if stage not in STAGES or (stage == "skeleton" and model != "unirig"):
        raise EngineError("bad_input", f"the {what} has an unknown stage {stage!r}")
    if _text(arrays, "skeleton") not in SKELETON or _text(arrays, "names_source") not in NAMES_SOURCES:
        raise EngineError("bad_input", f"the {what} has unknown skeleton or names_source values")
    names, parents, joints = arrays["names"], arrays["parents"], arrays["joints"]
    bones = len(names)
    if bones == 0 or len(parents) != bones or joints.shape != (bones, 3):
        raise EngineError("bad_input", f"the {what} has inconsistent bones")
    if len(set(names.tolist())) != bones:
        raise EngineError("bad_input", f"the {what} has repeated bone names")
    if (parents < -1).any() or (parents >= bones).any() or int((parents == -1).sum()) != 1:
        raise EngineError("bad_input", f"the {what} needs exactly one root and parents inside the skeleton")
    vertices = arrays["vertices"]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise EngineError("bad_input", f"the {what} has no vertices")
    if stage == "final":
        if "skin" not in arrays or arrays["skin"].shape != (len(vertices), bones):
            raise EngineError("bad_input", f"the {what} needs skin weights of {len(vertices)} by {bones}")
        if not np.isfinite(arrays["skin"]).all():
            raise EngineError("bad_input", f"the {what} has skin weights that are not numbers")
    if model == "skintokens" and arrays["matrix_local"].shape != (bones, 4, 4):
        raise EngineError("bad_input", f"the {what} has a matrix_local of the wrong shape")
    if model == "unirig":
        faces = arrays["faces"]
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(vertices):
            raise EngineError("bad_input", f"the {what} has faces pointing outside its vertices")
        if "tails" in arrays and arrays["tails"].shape != (bones, 3):
            raise EngineError("bad_input", f"the {what} has tails of the wrong shape")
    return model, stage
