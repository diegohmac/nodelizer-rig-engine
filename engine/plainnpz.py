"""Plain .npz files: arrays of numbers, booleans and text only, never pickle.

Saving refuses object arrays (they would need pickle), and loading passes allow_pickle=False, so a
file between steps can only ever hold data. A file is written under a temporary name and renamed into
place, so a reader never sees half of one.
"""
import json
import os
from pathlib import Path

import numpy as np

from .events import EngineError


def text(value):
    return np.array(str(value), dtype=np.str_)


def texts(values):
    values = [str(v) for v in values]
    return np.array(values, dtype=np.str_) if values else np.zeros((0,), dtype="<U1")


def as_text(array):
    return str(np.asarray(array)[()])


def as_texts(array):
    return [str(v) for v in np.asarray(array).tolist()]


def as_json(array):
    return json.loads(as_text(array))


def save(path, arrays):
    path = Path(path)
    clean = {}
    for key, value in arrays.items():
        if value is None:
            continue
        value = np.asarray(value)
        if value.dtype == object:
            raise TypeError(f"{key} is an object array; it would need pickle")
        if value.dtype.kind not in "biufUS":
            raise TypeError(f"{key} has dtype {value.dtype}, which a plain file cannot hold")
        clean[key] = value
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as handle:
        np.savez(handle, **clean)
    os.replace(part, path)


def load(path, what="input"):
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    except ValueError as error:  # an object array: the file asks for pickle
        raise EngineError("bad_input", f"the {what} is not a plain array file: {error}", path=str(path))
    except Exception as error:
        raise EngineError("bad_input", f"the {what} could not be read: {type(error).__name__}: {error}",
                          path=str(path))
