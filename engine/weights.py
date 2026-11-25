"""The models' weights: outside the engine folder, one folder per model, passed with --weights.

Each folder holds its files at these paths, relative to it, checked by SHA-256 (the app checks its
downloads; `probe --verify-weights` checks again on request). The two small config files that fix the
networks' shapes ship inside the engine folder instead (CONFIGS, relative to the model's code folder).
build/sources.py pins the same files for the build; a test keeps the two lists equal.
"""
import hashlib
from pathlib import Path

from .events import EngineError

WEIGHTS = {
    "skintokens": {
        "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt":
            "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692",
        "experiments/skin_vae_2_10_32768/last.ckpt":
            "4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4",
    },
    "unirig": {
        "experiments/skeleton/articulation-xl_quantization_256/model.ckpt":
            "d8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4",
        "experiments/skin/articulation-xl/model.ckpt":
            "9d40cf42fb9d4c10d8b373d9f4f557c6fbbc5ebacae7ae6b3dce5a0d8a18bc33",
    },
}
CONFIGS = {
    "skintokens": {"models/Qwen3-0.6B/config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"},
    "unirig": {"hf/opt-350m/config.json": "3ec75df0e166476e342cf95226ba775d8155449a7b1e430728b3ca26b425e4ab"},
}

GIB = 2**30
# Free video memory each model needs to run at normal speed: the card peaks measured on Windows (RTX 5090,
# dist/rig-native-results/work-4: SkinTokens 4.6 to 5.1 GiB, UniRig's skeleton pass 7.1 GiB and skin pass
# 6.7 GiB) plus 0.4 GiB. With less, Windows spills into system memory: the rig still comes out, slower
# (SkinTokens took 4.5 times as long at 5.2 GiB free).
NEEDS_FREE_BYTES = {"skintokens": int(5.5 * GIB), "unirig": int(7.5 * GIB)}
# The same on the processor, where the model uses the computer's own memory, for each of a Mac's memory
# modes (PROTOCOL.md, Devices): the largest peak footprint measured on a Mac in float32 plus a margin,
# as macOS needs room besides. Lean: 5.7 GiB (a kept skeleton of 58 bones; 4.9 to 5.2 GiB for made
# ones) plus 1.3 GiB. Fast: 11.3 GiB (a kept skeleton of 58 bones; 10.3 GiB for
# mine_rodin's 52 made ones) plus 1.7 GiB. With less available macOS swaps: the rig still comes out,
# slower. Elsewhere (Linux) the lean figure.
NEEDS_FREE_BYTES_HOST = {"lean": {"skintokens": int(7 * GIB)}, "fast": {"skintokens": int(13 * GIB)}}


def needs_host(model, mode):
    return NEEDS_FREE_BYTES_HOST[mode or "lean"][model]


def folder(model, value):
    """The model's weights folder, with every file there; bad_weights names what is missing."""
    if value is None:
        raise EngineError("bad_weights", f"--weights, the folder of {model}'s weights, is needed",
                          missing=sorted(WEIGHTS[model]))
    path = Path(value)
    if not path.is_absolute():
        raise EngineError("bad_arguments", "the weights path must be absolute", path=str(value))
    path = path.resolve()
    missing = [rel for rel in WEIGHTS[model] if not (path / rel).is_file()]
    if missing:
        raise EngineError("bad_weights", f"{model}'s weights are missing from {path}", missing=missing)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(model, path):
    """Each file's size and whether its SHA-256 is the pinned one (reads every byte: seconds per GB)."""
    report = {}
    for rel, want in WEIGHTS[model].items():
        file = Path(path) / rel
        if not file.is_file():
            report[rel] = {"present": False, "ok": False}
            continue
        got = sha256(file)
        report[rel] = {"present": True, "bytes": file.stat().st_size, "ok": got == want, "sha256": got}
    return report
