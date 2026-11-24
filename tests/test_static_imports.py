"""A static proof of the split: walking the imports from each step's entry, through our code and the
models' code in the built engine folder, no module reachable from the rig step imports Blender and
nothing reachable from read or write imports PyTorch.

In our own code every import counts, wherever it is (a function's import runs when the step does). In
the models' code only imports that run when a module loads count: module level, including inside if
and try blocks and class bodies, but not inside functions. Third party packages are not followed; the
runtime guard (engine/guard.py, test_guard.py) covers them.
"""
import ast
from pathlib import Path

import pytest

BLENDER = {"bpy", "_bpy", "bpy_extras", "mathutils", "bmesh", "gpu"}
TORCH = {"torch", "torchvision", "lightning", "pytorch_lightning", "transformers", "diffusers", "triton", "spconv",
         "torch_scatter", "torch_cluster", "flash_attn", "flash_attn_interface"}


class Walker:
    def __init__(self, engine):
        self.engine = engine
        self.roots = {"skintokens": engine / "models/skintokens", "unirig": engine / "models/unirig"}

    def find(self, name, context):
        """(file, is_package) for a module, or None when it is third party."""
        parts = name.split(".")
        if parts[0] in ("engine",):
            base = self.engine
        elif parts[0] == "src" and context in self.roots:
            base = self.roots[context]
        elif parts[0] == "flash_attn" and context is not None:
            base = self.engine / "standin"
        else:
            return None
        path = base.joinpath(*parts)
        if (path / "__init__.py").is_file():
            return path / "__init__.py", True
        if path.with_suffix(".py").is_file():
            return path.with_suffix(".py"), False
        if path.is_dir():  # a namespace package
            return None, True
        return None

    @staticmethod
    def _statements(tree, every_level):
        if every_level:
            yield from ast.walk(tree)
            return
        stack = list(tree.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            yield node
            stack.extend(ast.iter_child_nodes(node))

    def imports(self, name, path, is_package, every_level):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        package = name if is_package else name.rpartition(".")[0]
        found = []
        for node in self._statements(tree, every_level):
            if isinstance(node, ast.Import):
                found += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")
                    base = base[: len(base) - (node.level - 1)]
                    module = ".".join(base + ([node.module] if node.module else []))
                else:
                    module = node.module
                found.append(module)
                found += [f"{module}.{alias.name}" for alias in node.names]
        return found

    def reach(self, entries):
        """Third party top level names reachable from the entries: [(module, context, every_level)]."""
        seen, external, queue = set(), set(), list(entries)
        while queue:
            name, context, every_level = queue.pop()
            if name.startswith("engine.skintokens"):
                context = "skintokens"
            elif name.startswith("engine.unirig"):
                context = "unirig"
            key = (name, context)
            if key in seen:
                continue
            seen.add(key)
            # a parent package's __init__ runs first
            parts = name.split(".")
            for i in range(1, len(parts)):
                queue.append((".".join(parts[:i]), context, False))
            hit = self.find(name, context)
            if hit is None:
                if self.find(parts[0], context) is None:
                    external.add(parts[0])
                continue
            path, is_package = hit
            if path is None:
                continue
            # Our code: every import counts, except in engine.cli, whose run() imports the one step
            # module this process runs; that module is an entry of its own.
            ours = name.startswith("engine") and name != "engine.cli"
            for module in self.imports(name, path, is_package, every_level or ours):
                queue.append((module, context, False))
        return external


def _entries(step):
    return [("engine.cli", None, False), (f"engine.steps.{step}", None, True)]


def test_cli_imports_each_step_only_inside_run(engine):
    """The walk treats engine.cli's function level imports as per step; check that they are exactly the
    four step modules, each in its own branch of run()."""
    tree = ast.parse((engine / "engine" / "cli.py").read_text(encoding="utf-8"))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    inside = [alias.name for node in ast.walk(run) if isinstance(node, ast.ImportFrom) for alias in node.names]
    assert sorted(inside) == ["probe", "read", "rig", "write"]
    elsewhere = [node for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name != "run"
                 for node in ast.walk(fn) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all(alias.name in ("shutil",) for node in elsewhere for alias in node.names), "cli imports more"


@pytest.fixture(scope="module")
def walker(engine):
    return Walker(engine)


@pytest.mark.parametrize("step", ["read", "write"])
def test_blender_steps_never_reach_pytorch(walker, step):
    external = walker.reach(_entries(step))
    assert "bpy" in external, "the walk should find Blender here; it found nothing"
    assert not external & TORCH, sorted(external & TORCH)


@pytest.mark.parametrize("step", ["rig", "probe"])
def test_pytorch_steps_never_reach_blender(walker, step):
    external = walker.reach(_entries(step))
    assert "torch" in external, "the walk should find PyTorch here; it found nothing"
    assert not external & BLENDER, sorted(external & BLENDER)


def test_the_walk_sees_function_level_imports_in_our_code(walker):
    # steps.rig imports the model drivers inside run(); they must be walked, or the proof is empty
    external = walker.reach(_entries("rig"))
    assert {"lightning", "transformers"} <= external
