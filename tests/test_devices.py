"""Devices: which device and number type a rig runs on, UniRig's refusal off CUDA, and the processor's
`card` line. CUDA's choice must stay what it was before devices existed (the Windows fingerprints rest
on it). The processor runs here up to its `card` line; test-mac.sh runs it whole on a Mac."""
import sys

import pytest

from conftest import REPO, run_step
from test_protocol import _error, _fake_weights, _read_file

sys.path.insert(0, str(REPO))
from engine import devices, events, host  # noqa: E402


@pytest.mark.parametrize("value, parsed", [("0", ("cuda", 0)), ("3", ("cuda", 3)), ("cuda", ("cuda", None)),
                                           ("cuda:1", ("cuda", 1)), ("CPU", ("cpu", None)), (" mps ", ("mps", None))])
def test_device_values(value, parsed):
    assert devices.parse(value) == parsed


@pytest.mark.parametrize("value", ["-1", "gpu", "cuda:", "cuda:x", "metal", ""])
def test_bad_device_values(value):
    with pytest.raises(events.EngineError) as caught:
        devices.parse(value)
    assert caught.value.code == "bad_arguments"


def test_cuda_is_unchanged():
    """Windows and Linux: CUDA, the card with the most memory unless a number is given, in the models'
    own bfloat16; --precision can't change it."""
    for platform in ("win32", "linux"):
        assert devices.choose(None, platform) == devices.Device("cuda", None, "bfloat16", False)
        assert devices.choose("2", platform) == devices.Device("cuda", 2, "bfloat16", True)
        assert devices.choose("cuda", platform, "bfloat16").precision == "bfloat16"
        with pytest.raises(events.EngineError) as caught:
            devices.choose(None, platform, "float32")
        assert caught.value.code == "bad_arguments"


def test_mac_default_and_number_types():
    assert devices.choose(None, "darwin") == devices.Device("cpu", None, "float32", False)
    assert devices.choose("mps", "darwin").precision == "bfloat16"
    assert devices.choose("cpu", "linux", "bfloat16") == devices.Device("cpu", None, "bfloat16", True)
    with pytest.raises(events.EngineError) as caught:
        devices.choose("mps", "linux")
    assert caught.value.code == "bad_arguments"


def test_unirig_needs_cuda():
    devices.require_model("unirig", "cuda", "win32")
    devices.require_model("skintokens", "cpu", "darwin")
    devices.require_model("skintokens", None, "darwin")
    for kind, platform in ((None, "darwin"), ("cuda", "darwin"), ("cpu", "linux"), ("mps", "darwin")):
        with pytest.raises(events.EngineError) as caught:
            devices.require_model("unirig", kind, platform)
        assert caught.value.code == "model_unavailable"


def test_computer_memory():
    facts = host.facts()
    assert facts["name"] and 0 < facts["free_bytes"] <= facts["total_bytes"]
    assert host.peak_footprint() > 0


def test_unirig_on_the_processor_is_refused(engine, work):
    read = _read_file(work / "x.read.npz", model="unirig")
    run = run_step(engine, "rig", "--model", "unirig", "--stage", "skeleton", "--input", read, "--output",
                   work / "x.rig.npz", "--work", work, "--weights", work / "none", "--seed", "1", "--device", "cpu")
    _error(run, "model_unavailable")
    assert "loading_libraries" not in [e.get("stage") for e in run.events]


def test_bad_device_and_precision_on_the_command_line(engine, work):
    read = _read_file(work / "x.read.npz")
    base = ["rig", "--model", "skintokens", "--input", read, "--output", work / "x.rig.npz", "--work", work,
            "--weights", _fake_weights(work / "weights"), "--seed", "1"]
    _error(run_step(engine, *base, "--device", "mps"), "bad_arguments", 2)
    _error(run_step(engine, *base, "--device", "gpu"), "bad_arguments", 2)
    _error(run_step(engine, *base, "--precision", "float32"), "bad_arguments", 2)
    _error(run_step(engine, *base, "--device", "cpu", "--precision", "float16"), "bad_arguments", 2)


def test_processor_card_line(engine, work):
    """On the processor the `card` line keeps the CUDA shape: device 0, one device, capability cpu, the
    computer's memory. The test image has PyTorch but not SkinTokens' own packages (transformers,
    lightning), so the step ends with an error once SkinTokens' code is imported, after the card line."""
    read = _read_file(work / "x.read.npz")
    run = run_step(engine, "rig", "--model", "skintokens", "--input", read, "--output", work / "x.rig.npz",
                   "--work", work, "--weights", _fake_weights(work / "weights"), "--seed", "1", "--device", "cpu")
    assert run.code == 1 and run.last["type"] == "error"
    card = next(e for e in run.events if e.get("stage") == "card")
    assert card["device"] == 0 and card["devices"] == 1 and card["capability"] == "cpu"  # no memory mode off a Mac
    assert card["card"] and 0 < card["free_bytes"] <= card["total_bytes"]
    stages = [e["stage"] for e in run.events if e["type"] == "progress" and not e.get("heartbeat")]
    assert stages[:3] == ["starting", "loading_libraries", "card"]


GIB = 2**30


@pytest.mark.parametrize("installed, mode", [(16 * GIB, "lean"), (24 * GIB, "lean"), (int(29.9 * GIB), "lean"),
                                             (30 * GIB, "fast"), (32 * GIB, "fast"), (36 * GIB, "fast")])
def test_mac_memory_mode_by_installed_memory(installed, mode):
    assert host.memory_mode(None, installed) == mode
    assert host.memory_mode("lean", 64 * GIB) == "lean" and host.memory_mode("fast", 8 * GIB) == "fast"


def test_mac_memory_mode_environment():
    lean = host.environment_for("lean", {"A": "1"})
    assert lean == {"A": "1", "MallocLargeCache": "0"} and host.running_mode(lean) == "lean"
    fast = host.environment_for("fast", lean)
    assert fast == {"A": "1"} and host.running_mode(fast) == "fast"
    from engine import weights

    assert weights.needs_host("skintokens", "lean") == 7 * GIB
    assert weights.needs_host("skintokens", "fast") == 13 * GIB
    assert weights.needs_host("skintokens", None) == 7 * GIB  # Linux: the lean figure


def test_memory_read_before_the_parser():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rigengine_entry", REPO / "rigengine.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    assert entry._asked_memory(["--model", "skintokens", "--memory", "lean"]) == "lean"
    assert entry._asked_memory(["--memory=fast"]) == "fast"
    assert entry._asked_memory(["--mem", "lean"]) == "lean"  # argparse takes the abbreviation too
    assert entry._asked_memory(["--model", "skintokens", "--seed", "1"]) is None


def test_memory_option_is_for_a_mac(engine, work):
    read = _read_file(work / "x.read.npz")
    run = run_step(engine, "rig", "--model", "skintokens", "--input", read, "--output", work / "x.rig.npz",
                   "--work", work, "--weights", _fake_weights(work / "weights"), "--seed", "1", "--device", "cpu",
                   "--memory", "lean")
    _error(run, "bad_arguments", 2)


def _macho(minos, platform=1):
    """A thin arm64 Mach-O header with one LC_BUILD_VERSION."""
    import struct

    major, minor = minos
    command = struct.pack("<IIIIII", 0x32, 24, platform, (major << 16) | (minor << 8), 0, 0)
    return struct.pack("<IiiIIIII", 0xFEEDFACF, 0x0100000C, 0, 6, 1, len(command), 0, 0) + command


def test_minimum_macos_check(tmp_path):
    import struct

    sys.path.insert(0, str(REPO / "build" / "mac"))
    import macho

    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "a.dylib").write_bytes(_macho((11, 2)))
    thin = _macho((14, 0))
    fat = struct.pack(">II", 0xCAFEBABE, 1) + struct.pack(">iiIII", 0x0100000C, 0, 4096, len(thin), 12)
    (tmp_path / "lib" / "b.so").write_bytes(fat + b"\0" * (4096 - len(fat)) + thin)
    (tmp_path / "text.py").write_text("print(1)")
    assert macho.minimums(tmp_path / "lib" / "a.dylib") == [(0x0100000C, (11, 2, 0))]
    assert macho.check(tmp_path, "14.0") == (2, (14, 0, 0))
    with pytest.raises(SystemExit) as caught:
        macho.check(tmp_path, "13.0")
    assert "lib/b.so needs macOS 14.0.0" in str(caught.value)
