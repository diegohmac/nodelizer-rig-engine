"""The CUDA path, pinned without a card. The Windows rigs (windows-kit/expected.json) rest on SkinTokens'
driver calling the model exactly as before devices existed, so these tests run the driver and the rig
step against a stand in for PyTorch and for SkinTokens' own modules that records every call, and check
the calls, their arguments and their order. The stand ins live only in this test's sys.modules: no real
PyTorch is imported here."""
import sys
import types

import numpy as np
import pytest

from conftest import REPO

sys.path.insert(0, str(REPO))
from engine import devices, events, seeding  # noqa: E402

calls = []


class FakeTensor:
    """A tensor: `.to(target)` is recorded; `detach` lets the fingerprint hash it."""

    def __init__(self, value, name):
        self.value, self.name = np.asarray(value, dtype=np.float32), name

    def to(self, target):
        calls.append(("to", self.name, target))
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _fake_torch():
    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.autocast = object()  # never replaced on CUDA
    cuda = types.SimpleNamespace()
    cuda.synchronize = lambda *a: calls.append(("cuda.synchronize",) + a)
    cuda.reset_peak_memory_stats = lambda *a: calls.append(("cuda.reset_peak_memory_stats",) + a)
    cuda.max_memory_allocated = lambda *a: calls.append(("cuda.max_memory_allocated",) + a) or 11
    cuda.max_memory_reserved = lambda *a: calls.append(("cuda.max_memory_reserved",) + a) or 12
    torch.cuda = cuda

    def mps_call(*a):
        raise AssertionError("an MPS call on the CUDA path")

    torch.mps = types.SimpleNamespace(get_rng_state=mps_call, set_rng_state=mps_call, synchronize=mps_call)
    return torch


class FakeResult:
    def __init__(self):
        joints = np.zeros((2, 3))
        self.detokenize_output = types.SimpleNamespace(tokens=[1, 2, 3], joints=joints, parents=[None, 0])
        self.skin_pred = np.ones((4, 2)) / 2
        self.asset = types.SimpleNamespace(parents=[-1, 0], joints=joints, vertices=np.zeros((4, 3)),
                                           matrix_local=np.tile(np.eye(4), (2, 1, 1)), skin=np.ones((4, 2)) / 2)


class FakeModel:
    tokenizer_config = {"t": 1}
    transform_config = {"predict_transform": {"p": 1}}
    _process_fn = None

    def eval(self):
        calls.append(("model.eval",))
        return self

    def float(self):
        raise AssertionError("model.float() on the CUDA path")

    def predict_step(self, batch, **kwargs):
        calls.append(("model.predict_step", sorted(k for k in batch if k != "generate_kwargs"),
                      dict(batch["generate_kwargs"]), kwargs))
        return {"results": [FakeResult()]}


def _fake_skintokens(monkeypatch):
    """SkinTokens' modules the driver imports, each recording what it is asked."""
    modules = {}

    def module(name, **attrs):
        made = types.ModuleType(name)
        made.__dict__.update(attrs)
        modules[name] = made
        return made

    class ModelSpec:
        @classmethod
        def load_from_system_checkpoint(cls, path, *args, **kwargs):
            return path

    class NoInit:
        def __enter__(self):
            calls.append(("no_init_weights",))

        def __exit__(self, *exc):
            return False

    def get_model(*args, **kwargs):
        calls.append(("get_model", args, kwargs))
        return FakeModel()

    class Transform:
        @staticmethod
        def parse(**kwargs):
            return "transform"

    class LazyAsset:
        pass

    class Part:
        def __init__(self):
            self.datapath = types.SimpleNamespace(loader=None)

    class DatasetConfig:
        @staticmethod
        def parse(**kwargs):
            calls.append(("DatasetConfig.parse", {k: kwargs[k] for k in ("shuffle", "batch_size", "num_workers")}))
            return types.SimpleNamespace(split_by_cls=lambda: {"articulation": Part()})

    class RigDatasetModule:
        def __init__(self, **kwargs):
            pass

        def predict_dataloader(self):
            batch = {"vertices": FakeTensor(np.arange(6).reshape(2, 3), "vertices"), "cls": ["articulation"],
                     "skeleton_tokens": FakeTensor([1], "skeleton_tokens"), "skeleton_mask": FakeTensor([1], "mask")}
            return {"articulation": iter([batch])}

    for name in ("src", "src.data", "src.model", "src.server", "src.tokenizer", "src.rig_package",
                 "src.rig_package.info", "transformers"):
        module(name)
    module("transformers.modeling_utils", no_init_weights=NoInit)
    module("src.data.transform", Transform=Transform)
    module("src.model.spec", ModelSpec=ModelSpec)
    module("src.server.spec", get_model=get_model)
    module("src.tokenizer.parse", get_tokenizer=lambda **k: types.SimpleNamespace(continuous_range=(0, 1),
                                                                               num_discrete=256))
    module("src.data.datapath", LazyAsset=LazyAsset)
    module("src.data.dataset", DatasetConfig=DatasetConfig, RigDatasetModule=RigDatasetModule)
    module("src.rig_package.info.asset", Asset=lambda **k: types.SimpleNamespace(**k))
    for name, made in modules.items():
        monkeypatch.setitem(sys.modules, name, made)


@pytest.fixture
def driver(monkeypatch, tmp_path):
    from engine import modelpath
    from engine.skintokens import rig as driver

    calls.clear()
    torch = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    _fake_skintokens(monkeypatch)
    monkeypatch.setattr(modelpath, "use", lambda model: calls.append(("modelpath.use", model)))
    monkeypatch.setattr(driver.os, "chdir", lambda path: calls.append(("chdir", str(path))))
    monkeypatch.setattr(seeding, "seed_all", lambda seed, t=None: calls.append(("seed_all", seed, t is torch)))
    monkeypatch.setattr(seeding, "fetch_like_a_worker",
                        lambda batches, t: calls.append(("fetch_like_a_worker", t is torch)) or next(batches))
    return driver, torch, tmp_path


class Recorder:
    def progress(self, stage, **fields):
        calls.append(("progress", stage))


def _arrays():
    return {"has_skeleton": np.bool_(False), "vertices": np.zeros((4, 3)), "faces": np.zeros((2, 3), np.int32),
            "vertex_normals": np.zeros((4, 3)), "face_normals": np.zeros((2, 3)),
            "vertex_bias": np.array([4]), "face_bias": np.array([2]), "mesh_names": np.array(["m"])}


def test_skintokens_on_cuda_calls_the_model_as_before(driver):
    """Device 0 (a card's number): the call sequence st_rig.py and engine 0.1.0 make, nothing else."""
    rig, torch, folder = driver
    autocast = torch.autocast
    rig_arrays, info = rig.rig(folder, folder / "weights", folder / "a.read.npz", _arrays(), 12345,
                               False, True, torch, 0, Recorder())
    names = [c[0] for c in calls]
    assert names == ["modelpath.use", "chdir", "progress", "no_init_weights", "get_model", "model.eval",
                     "cuda.synchronize", "cuda.reset_peak_memory_stats", "progress", "seed_all",
                     "DatasetConfig.parse", "fetch_like_a_worker", "to", "to", "to", "progress",
                     "model.predict_step", "cuda.synchronize"], names
    assert ("progress", "loading_model") == calls[2] and ("progress", "preparing") == calls[8]
    assert ("progress", "generating") == calls[15]
    assert calls[4] == ("get_model", (rig.CKPT,), {"hf_path": None})  # no device, no number type
    assert calls[7] == ("cuda.reset_peak_memory_stats", 0)
    assert calls[9] == ("seed_all", 12345, True)
    assert calls[10] == ("DatasetConfig.parse", {"shuffle": False, "batch_size": 1, "num_workers": 0})
    assert {c[2] for c in calls if c[0] == "to"} == {"cuda"}
    predict = calls[16]
    assert predict[2] == rig.GENERATE == dict(max_length=2048, top_k=5, top_p=0.95, temperature=1.0,
                                              repetition_penalty=2.0, num_return_sequences=1, num_beams=10,
                                              do_sample=True)
    assert predict[3] == {"make_asset": True} and "skeleton_tokens" not in predict[1]
    assert torch.autocast is autocast, "the autocast was replaced on CUDA"
    assert ("progress", "skinning") not in calls
    assert info["bones"] == 2 and rig_arrays["skin"].shape == (4, 2)


def test_rig_step_asks_for_the_card_as_before(monkeypatch, work):
    """The rig step's device choice off a Mac (Linux here, Windows alike): no --device is the card with
    the most memory (torchenv.card(torch, None)); N and cuda:N are card N. Stopped at the card."""
    from engine.steps import rig as step
    from engine.steps import torchenv
    from test_protocol import _fake_weights, _read_file

    asked = []

    def card(torch, index):
        asked.append(index)
        raise events.EngineError("no_cuda", "stopped here", torch="fake", built_for=None)

    monkeypatch.setattr(torchenv, "import_torch", lambda: _fake_torch())
    monkeypatch.setattr(torchenv, "card", card)
    monkeypatch.setattr(torchenv, "host_card", lambda *a: pytest.fail("host_card on the CUDA path"))
    monkeypatch.setattr(step.workspace, "set_environment", lambda *a: None)  # keep this process's own
    monkeypatch.setattr(step.workspace, "check_code", lambda model: work)  # the repo has no model code
    read = _read_file(work / "x.read.npz")
    for value, index in ((None, None), ("2", 2), ("cuda:1", 1), ("cuda", None)):
        args = step_args(read, work, _fake_weights(work / "weights"), value)
        with pytest.raises(events.EngineError) as caught:
            step.run(args, Recorder())
        assert caught.value.code == "no_cuda"
        assert asked[-1] == index
    assert devices.choose(None, "win32").precision == "bfloat16"


def step_args(read, work, weights, device):
    from engine.cli import build_parser

    argv = ["rig", "--model", "skintokens", "--input", str(read), "--output", str(work / "x.rig.npz"),
            "--work", str(work), "--weights", str(weights), "--seed", "1"]
    if device is not None:
        argv += ["--device", device]
    return build_parser().parse_args(argv)
