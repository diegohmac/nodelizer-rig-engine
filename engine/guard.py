"""Keeps Blender and PyTorch out of each other's process.

Blender (bpy, GPL) and PyTorch with NVIDIA's CUDA files must never be loaded in the same process. Each
step installs a blocker at its very start that refuses to import the modules it must not load, and at
the end checks sys.modules once more. Either way a forbidden module fails the step loudly
(`forbidden_module`). An optional import inside a library (`try: import torch`) just sees the module
as missing, which is what we want.
"""
import importlib.abc
import sys

BLENDER = ("bpy", "_bpy", "bpy_extras", "bpy_types", "mathutils", "bmesh", "gpu", "addon_utils")
TORCH = ("torch", "torchvision", "torchaudio", "lightning", "pytorch_lightning", "lightning_fabric",
         "transformers", "diffusers", "triton", "spconv", "cumm", "torch_scatter", "torch_cluster",
         "flash_attn", "flash_attn_interface")
# Not forbidden by policy, only never needed by read and write: kept out so those steps stay small.
UNNEEDED_IN_BLENDER_STEPS = ("open3d",)

FORBIDDEN = {
    "read": TORCH,
    "write": TORCH,
    "rig": BLENDER,
    "probe": BLENDER,
}
BLOCKED = {
    "read": TORCH + UNNEEDED_IN_BLENDER_STEPS,
    "write": TORCH + UNNEEDED_IN_BLENDER_STEPS,
    "rig": BLENDER,
    "probe": BLENDER,
}


class ForbiddenImport(ModuleNotFoundError):
    pass


class _Blocker(importlib.abc.MetaPathFinder):
    def __init__(self, step, names):
        self.step = step
        self.names = frozenset(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in self.names:
            raise ForbiddenImport(f"rig engine: the {self.step} step may not load {fullname}", name=fullname)
        return None


def install(step):
    """Blocks this step's forbidden modules for the rest of the process. Fails if one is loaded already."""
    loaded = loaded_forbidden(step)
    if loaded:
        raise ForbiddenImport(f"rig engine: {', '.join(loaded)} loaded before the {step} step began")
    sys.meta_path.insert(0, _Blocker(step, BLOCKED[step]))


def loaded_forbidden(step):
    names = set(FORBIDDEN[step])
    return sorted({name.partition(".")[0] for name in sys.modules} & names)
