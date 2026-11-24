"""UniRig's Blender exports, as plain arrays.

UniRig's writers end in Exporter._export_fbx(...), which builds the armature in Blender and writes an
.fbx. In the rig step that one method is replaced by a recorder, so UniRig's own writer code runs
unchanged (the skin writer's resampling included) and only the Blender part is deferred. The write
step then calls the real _export_fbx with exactly the recorded arguments.

Arrays keep their dtype. Parents (a list with None for the root) are stored as int32 with -1, names
as text; the scalar options as one JSON text.
"""
import json

import numpy as np

from .. import plainnpz

ARRAY_ARGUMENTS = ("vertices", "joints", "skin", "faces", "tails")
OPTION_ARGUMENTS = ("extrude_size", "group_per_vertex", "add_root", "do_not_normalize", "use_extrude_bone",
                    "use_connect_unique_child", "extrude_from_parent")
# _export_fbx's own defaults, so an argument a caller left out is recorded with its value
DEFAULTS = dict(faces=None, extrude_size=0.03, group_per_vertex=-1, add_root=False, do_not_normalize=False,
                use_extrude_bone=True, use_connect_unique_child=True, extrude_from_parent=True, tails=None)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, exporter, path, vertices, joints, skin, parents, names, **options):
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"_export_fbx got arguments the engine does not know: {sorted(unknown)}")
        call = dict(DEFAULTS)
        call.update(options)
        call.update(path=path, vertices=vertices, joints=joints, skin=skin, parents=list(parents), names=list(names))
        self.calls.append(call)


def to_arrays(call):
    arrays = {}
    for key in ARRAY_ARGUMENTS:
        if call.get(key) is not None:
            arrays[key] = np.asarray(call[key])
    arrays["parents"] = np.array([-1 if p is None else int(p) for p in call["parents"]], dtype=np.int32)
    arrays["names"] = plainnpz.texts(call["names"])
    options = {}
    for key in OPTION_ARGUMENTS:
        value = call[key]
        options[key] = value.item() if hasattr(value, "item") else value
    arrays["export_options"] = plainnpz.text(json.dumps(options, sort_keys=True))
    return arrays


def to_call(arrays):
    """The keyword arguments for Exporter._export_fbx, without `path`."""
    call = {key: (np.array(arrays[key]) if key in arrays else None) for key in ARRAY_ARGUMENTS}
    call["parents"] = [None if int(p) < 0 else int(p) for p in arrays["parents"]]
    call["names"] = plainnpz.as_texts(arrays["names"])
    call.update(plainnpz.as_json(arrays["export_options"]))
    return call
