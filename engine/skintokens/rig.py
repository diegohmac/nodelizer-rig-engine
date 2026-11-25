"""SkinTokens' rig step. PyTorch only.

It follows the Windows kit's st_rig.py (helper off, seeded per job), which gave the same rigs bit for
bit as SkinTokens' own helper runs: the same model load (inside transformers' no_init_weights, which
skips a 35 second random fill the checkpoint overwrites anyway), model.eval(), the same dataset and
transform, the fetch seeded as a data loader worker would be, demo.py's generation settings, and
predict_step(make_asset=True). The only change in the data path is where the Asset comes from: the
read file, holding exactly what BpyParser.load returned, instead of Blender in this process.

On CUDA (`device` is the card's number) every call is as it was. On the processor or a Mac's GPU
(`device` is an engine.devices.Device), _adapt points SkinTokens' own torch.autocast('cuda', ...) at that
device first, and the model loads there in the device's number type (engine/devices.py, PRECISION).
"""
import os

import numpy as np

from .. import fingerprint, modelpath, names, plainnpz, seeding
from ..events import EngineError, log, model_failure, weights_failure
from . import asset_io

CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
CLASS = "articulation"  # the class st_rig.py and demo.py give
# demo.py's generation settings
GENERATE = dict(max_length=2048, top_k=5, top_p=0.95, temperature=1.0, repetition_penalty=2.0,
                num_return_sequences=1, num_beams=10, do_sample=True)


def _on_cuda(device):
    return not hasattr(device, "kind")


def _adapt(torch, device, events):
    """Off CUDA, before any of SkinTokens' modules load. SkinTokens wraps its work in
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16), which does nothing without CUDA and would
    leave its bfloat16 weights meeting float32 inputs. Here that autocast runs on the device instead:
    in bfloat16 as written, or switched off for float32, where the model is turned to float32 after it
    loads and the skin VAE's one hand made cast of its query to bfloat16 (skin_vae_model.py) follows the
    keys' type instead. SkinTokens' skin is decoded after its tokens; a `skinning` line says when."""
    kind, precision = device.kind, device.precision
    upstream = torch.autocast

    class DeviceAutocast(upstream):
        def __init__(self, device_type, *args, **kwargs):
            if device_type == "cuda":
                device_type = kind
                if precision == "float32":
                    kwargs["enabled"] = False
            super().__init__(device_type, *args, **kwargs)

    torch.autocast = DeviceAutocast
    import src.model.skin_vae_model as vae
    import src.model.tokenrig as tokenrig

    if precision == "float32":
        inner = vae.flash_attn_func
        vae.flash_attn_func = lambda q, k, v, *args, **kwargs: inner(q.to(k.dtype), k, v, *args, **kwargs)
    decode = tokenrig.decode

    def decode_reporting(*args, **kwargs):
        events.progress("skinning")
        return decode(*args, **kwargs)

    tokenrig.decode = decode_reporting


def _synchronize(torch, device):
    if _on_cuda(device):
        torch.cuda.synchronize()
    elif device.kind == "mps":
        torch.mps.synchronize()


def _load_model(torch, weights_folder, events, device):
    from transformers.modeling_utils import no_init_weights

    from src.data.transform import Transform
    from src.model.spec import ModelSpec
    from src.server.spec import get_model
    from src.tokenizer.parse import get_tokenizer

    # Both checkpoints load through ModelSpec.load_from_system_checkpoint: the main one by its path,
    # the skin VAE by the path inside the main one's settings, relative to SkinTokens' folder upstream.
    # Relative paths are taken from the weights folder instead.
    upstream = ModelSpec.__dict__["load_from_system_checkpoint"].__func__

    def from_weights(cls, checkpoint_path, *args, **kwargs):
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = str(weights_folder / checkpoint_path)
        return upstream(cls, checkpoint_path, *args, **kwargs)

    ModelSpec.load_from_system_checkpoint = classmethod(from_weights)
    events.progress("loading_model")
    try:
        with no_init_weights():
            if _on_cuda(device):
                model = get_model(CKPT, hf_path=None)
            else:
                model = get_model(CKPT, hf_path=None, device=device.kind)
                if device.precision == "float32":
                    model = model.float()
    except Exception as error:
        raise weights_failure(error, "SkinTokens' weights would not load")
    model.eval()  # demo.py never does this; without it dropout stays on and runs don't repeat
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])
    _synchronize(torch, device)
    return model, tokenizer, transform


def _fetch(read_path, arrays, with_skeleton, model, tokenizer, transform, seed, torch, device):
    """One batch, made exactly as st_rig.py makes it, from the read file instead of Blender."""
    from src.data.datapath import LazyAsset
    from src.data.dataset import DatasetConfig, RigDatasetModule
    from src.rig_package.info.asset import Asset

    fields = asset_io.asset_fields(arrays, with_skeleton=with_skeleton)

    class ReadFileAsset(LazyAsset):
        def load(self):
            asset = Asset(**{key: (value.copy() if isinstance(value, (np.ndarray, list)) else value)
                             for key, value in fields.items()})
            asset.cls = self.cls
            asset.path = self.path
            return asset

    seeding.seed_all(seed, torch)
    datapath = {"data_name": None, "loader": None, "filepaths": {CLASS: [str(read_path)]}}
    config = DatasetConfig.parse(shuffle=False, batch_size=1, num_workers=0, pin_memory=True,
                                 persistent_workers=False, datapath=datapath).split_by_cls()
    for part in config.values():
        part.datapath.loader = ReadFileAsset
    module = RigDatasetModule(predict_dataset_config=config, predict_transform=transform,
                              tokenizer=tokenizer, process_fn=model._process_fn)
    batches = iter(module.predict_dataloader()[CLASS])
    if _on_cuda(device) or device.kind != "mps":
        return seeding.fetch_like_a_worker(batches, torch)
    # As CUDA's generator is around the fetch (fork_rng), the GPU's is kept as it was before it.
    state = torch.mps.get_rng_state()
    try:
        return seeding.fetch_like_a_worker(batches, torch)
    finally:
        torch.mps.set_rng_state(state)


def rig(root, weights_folder, read_path, arrays, seed, keep_skeleton, standard_names, torch, device, events):
    modelpath.use("skintokens")
    os.chdir(root)  # SkinTokens names Qwen3's config and its skeleton configs relative to its folder
    from torch import Tensor

    if not _on_cuda(device):
        _adapt(torch, device, events)
    model, tokenizer, transform = _load_model(torch, weights_folder, events, device)
    has_skeleton = bool(arrays["has_skeleton"][()])
    keeping = keep_skeleton and has_skeleton
    info = {"skeleton": "made", "keep_skeleton": None}
    if keep_skeleton and not has_skeleton:
        info["keep_skeleton"] = "the file has no skeleton, so one was made"

    if _on_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)
    events.progress("preparing")
    try:
        batch = _fetch(read_path, arrays, True, model, tokenizer, transform, seed, torch, device)
    except Exception as error:
        if keeping:
            raise EngineError("skeleton_not_kept", f"SkinTokens could not take the file's skeleton: "
                              f"{type(error).__name__}: {error}"[:2000])
        if not has_skeleton:
            raise model_failure(error, "SkinTokens could not prepare the model file")
        # The file's own skeleton, which isn't wanted, could not be tokenized. Start the job again
        # without it, seeded afresh, so the rig doesn't depend on the failed attempt.
        log(f"the file's skeleton was not usable ({type(error).__name__}: {error}); rigging without it")
        info["input_skeleton_ignored"] = True
        try:
            batch = _fetch(read_path, arrays, False, model, tokenizer, transform, seed, torch, device)
        except Exception as again:
            raise model_failure(again, "SkinTokens could not prepare the model file")
    target = "cuda" if _on_cuda(device) else device.kind
    batch = {k: v.to(target) if isinstance(v, Tensor) else v for k, v in batch.items()}
    batch["generate_kwargs"] = dict(GENERATE)
    fingerprints = {"points": fingerprint.fp(batch["vertices"])}

    events.progress("generating")
    file_names = None
    if keeping:
        if "skeleton_tokens" not in batch:
            raise EngineError("skeleton_not_kept", "SkinTokens made no tokens from the file's skeleton")
        mask = batch["skeleton_mask"][0] == 1
        skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()
        source = batch["model_input"][0].asset
        file_names = list(source.joint_names)
        file_joints = np.asarray(source.joints, dtype=np.float64)
        file_parents = [None if p is None or p < 0 else int(p) for p in source.parents]
        try:
            result = model.predict_step(batch, skeleton_tokens=[skeleton_tokens], make_asset=True)["results"][0]
        except Exception as error:
            raise model_failure(error, "SkinTokens failed to skin the file's skeleton")
        out = result.detokenize_output
        out_joints = np.asarray(out.joints, dtype=np.float64)
        out_parents = [None if p is None or p < 0 else int(p) for p in out.parents]
        # keep the file's names only if the bones came back in the file's order (st_rig.py's check)
        low, high = tokenizer.continuous_range
        step = (high - low) / tokenizer.num_discrete
        same_count = len(out_joints) == len(file_names)
        worst = float(np.linalg.norm(out_joints - file_joints, axis=-1).max()) if same_count else None
        kept = same_count and worst <= 1.5 * step and out_parents == file_parents
        info["keep_check"] = dict(file_bones=len(file_names), bones=len(out_joints), worst_bone_distance=worst,
                                  quantization_step=step, parents_equal=out_parents == file_parents)
        if not kept:
            raise EngineError("skeleton_not_kept", "the bones SkinTokens gave back are not the file's bones",
                              keep_check=info["keep_check"])
        info["skeleton"] = "kept"
    else:
        batch.pop("skeleton_tokens", None)
        batch.pop("skeleton_mask", None)
        try:
            result = model.predict_step(batch, make_asset=True)["results"][0]
        except Exception as error:
            raise model_failure(error, "SkinTokens failed to rig the file")
        out = result.detokenize_output
    _synchronize(torch, device)
    if result.skin_pred is None or result.asset is None or result.asset.skin is None:
        raise EngineError("rig_failed", "SkinTokens made a skeleton but no skin for it")

    asset = result.asset
    parents = np.asarray(asset.parents).astype(np.int32)
    bones = int(len(parents))
    if file_names is not None:
        final_names, names_source = file_names, "kept"
        info["standard_names"] = {"asked": bool(standard_names), "matched": False, "layout": None,
                                  "placement_warnings": [], "note": "the file's names were kept"}
    else:
        standard, report = names.decide(parents.tolist(), asset.joints, standard_names)
        info["standard_names"] = report
        if standard is not None:
            final_names, names_source = standard, "standard"
        else:
            final_names, names_source = [f"bone_{i}" for i in range(bones)], "numbered"
    fingerprints.update(tokens=fingerprint.tokens_fp(out.tokens), joints=fingerprint.joints_fp(out.joints),
                        skin=fingerprint.fp(result.skin_pred))
    info.update(stage="final", bones=bones, names=names_source, fingerprints=fingerprints)

    rig_arrays = {
        "stage": plainnpz.text("final"),
        "skeleton": plainnpz.text(info["skeleton"]),
        "names_source": plainnpz.text(names_source),
        "names": plainnpz.texts(final_names),
        "parents": parents,
        "joints": np.asarray(asset.joints),
        # what transfer_rigging reads from the source asset, exactly as predict_step made it
        "vertices": np.asarray(asset.vertices),
        "matrix_local": np.asarray(asset.matrix_local),
        "skin": np.asarray(asset.skin),
    }
    return rig_arrays, info
