"""The rig step can't run without a graphics card and the weights, so its drivers are checked against
the models' code statically: every name they import from a model exists there, and every keyword they
pass to a model's function or class is one it takes. (Everything else in the drivers runs in the
read and write tests, or on Windows.)"""
import ast
from pathlib import Path

import pytest

DRIVERS = {
    "skintokens": ["read.py", "rig.py", "write.py", "asset_io.py"],
    "unirig": ["read.py", "rig.py", "write.py", "capture.py"],
}


def _module_file(root, dotted):
    path = root.joinpath(*dotted.split("."))
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    return None


def _top_level_names(tree):
    names = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names[(alias.asname or alias.name).split(".")[0]] = node
        elif isinstance(node, (ast.Try, ast.If)):
            for inner in ast.walk(node):
                if isinstance(inner, (ast.FunctionDef, ast.ClassDef)):
                    names[inner.name] = inner
                elif isinstance(inner, (ast.Import, ast.ImportFrom)):
                    for alias in inner.names:
                        names[(alias.asname or alias.name).split(".")[0]] = inner
    return names


def _parameters(node):
    """(names, takes **kwargs) for a function, a class's __init__, or a dataclass's fields."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        return names, args.kwarg is not None
    if isinstance(node, ast.ClassDef):
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return _parameters(item)
        fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign)
                  and isinstance(item.target, ast.Name)}
        if any("dataclass" in ast.unparse(d) for d in node.decorator_list):
            return fields, False
    return None


def _imports_from_model(tree):
    """{local name: (module, name)} for `from src.x import name` anywhere in a driver."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
            for alias in node.names:
                found[alias.asname or alias.name] = (node.module, alias.name)
    return found


def _cases(engine):
    for model, files in DRIVERS.items():
        for name in files:
            yield model, engine / "engine" / model / name


@pytest.mark.parametrize("model,driver", [pytest.param(m, n, id=f"{m}-{n}") for m, files in DRIVERS.items()
                                          for n in files])
def test_driver_imports_and_keywords(engine, model, driver):
    root = engine / "models" / model
    tree = ast.parse((engine / "engine" / model / driver).read_text(encoding="utf-8"))
    imported = _imports_from_model(tree)
    definitions = {}
    for local, (module, name) in imported.items():
        path = _module_file(root, module)
        assert path is not None, f"{driver}: {module} does not exist in {model}"
        names = _top_level_names(ast.parse(path.read_text(encoding="utf-8")))
        submodule = _module_file(root, f"{module}.{name}")
        assert name in names or submodule is not None, f"{driver}: {module} has no {name}"
        if name in names:
            definitions[local] = names[name]
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = None
        if isinstance(node.func, ast.Name) and node.func.id in definitions:
            target = definitions[node.func.id]
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                and node.func.value.id in definitions and isinstance(definitions[node.func.value.id], ast.ClassDef):
            target = next((item for item in definitions[node.func.value.id].body
                           if isinstance(item, ast.FunctionDef) and item.name == node.func.attr), None)
            assert target is not None, f"{driver}: {node.func.value.id} has no method {node.func.attr}"
        if target is None:
            continue
        signature = _parameters(target)
        if signature is None:
            continue
        names, open_ended = signature
        passed = {k.arg for k in node.keywords if k.arg is not None}
        if not open_ended:
            assert passed <= names, f"{driver} line {node.lineno}: {sorted(passed - names)} not taken"
        checked += 1
    if driver in ("rig.py", "write.py", "read.py"):
        assert checked, f"{driver}: no call into the model's code was checked"
