"""Puts one model's code folder first on sys.path. Both models call their package `src`, so a process
only ever works with one of them."""
import os
import sys

from . import workspace


def use(model):
    root = str(workspace.model_root(model))
    loaded = sys.modules.get("src")
    if loaded is not None:
        places = [os.path.abspath(p) for p in list(getattr(loaded, "__path__", []))]
        if not all(p.startswith(root) for p in places):
            raise RuntimeError("another model's code is loaded in this process already")
    others = {str(workspace.model_root(m)) for m in workspace.MODELS if m != model}
    sys.path[:] = [p for p in sys.path if p not in others]
    if root not in sys.path:
        sys.path.insert(0, root)
    return workspace.model_root(model)
