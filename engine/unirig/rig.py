"""UniRig's rig step, one pass per process (--stage skeleton or skin). PyTorch only.

This is UniRig's run.py, predict path, line for line in the same order (so the random number
generators are used exactly as there), with these differences only:
  - the input is the read file, copied into a fresh folder as raw_data.npz (run.py: get_files and
    the extract step's output folder);
  - the checkpoint is opened here with torch.load(weights_only=True), allowing python-box's Box only,
    and loaded into the system before predicting (run.py: Lightning loads it from ckpt_path);
  - Lightning gets logger=False and a scratch root folder, and no progress bar;
  - Exporter._export_fbx records its arguments instead of calling Blender (see capture.py).
"""
import json
import os
import shutil

import numpy as np

from .. import fingerprint, modelpath, names, plainnpz, workspace
from ..events import EngineError, log, model_failure, weights_failure
from . import capture

TASKS = {
    "skeleton": "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml",
    "skin": "configs/task/quick_inference_unirig_skin.yaml",
}
SKELETON_FIELDS = ("joints", "tails", "skin", "parents", "names", "matrix_local")


def _load_yaml(path):
    import yaml
    from box import Box

    if path.endswith(".yaml"):
        path = path.removesuffix(".yaml")
    path += ".yaml"
    with open(path, "r", encoding="utf-8") as handle:
        return Box(yaml.safe_load(handle))


def _make_stash():
    """A Lightning callback that keeps what the writer is given, for fingerprints."""
    from lightning.pytorch.callbacks import Callback

    class Stash(Callback):
        def __init__(self):
            self.batches = []

        def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self.batches.append((outputs, {"vertices": batch["vertices"].detach().float().cpu()}))

    return Stash()


def _prepare_input(read_path, arrays, stage, scratch):
    folder = scratch / "input"
    folder.mkdir()
    target = folder / "raw_data.npz"
    if stage == "skeleton" and bool(arrays["has_skeleton"][()]):
        # The skeleton pass never uses a skeleton the file has; leave it out so a strange one cannot
        # stop the pass (the mesh, and so the sampled points, are the same either way).
        plainnpz.save(target, {k: v for k, v in arrays.items() if k not in SKELETON_FIELDS})
    else:
        shutil.copyfile(read_path, target)
    return folder


def rig(root, weights_folder, read_path, arrays, seed, stage, keep_skeleton, standard_names, torch, device, work,
        events):
    modelpath.use("unirig")
    os.chdir(root)  # UniRig reads its configs and OPT's config relative to its folder
    has_skeleton = bool(arrays["has_skeleton"][()])
    if stage == "skin" and not has_skeleton:
        raise EngineError("bad_input", "UniRig's skin pass needs a read file with a skeleton")

    import lightning as L
    from box import Box

    from src.data.datapath import Datapath
    from src.data.dataset import DatasetConfig, UniRigDatasetModule
    from src.data.exporter import Exporter
    from src.data.transform import TransformConfig
    from src.inference.download import download
    from src.model.parse import get_model
    from src.system.parse import get_system, get_writer
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizerConfig

    scratch = workspace.scratch_folder(work, f"unirig-{stage}")
    try:
        # run.py, from the top
        torch.set_float32_matmul_precision("high")
        L.seed_everything(seed, workers=True)
        task = _load_yaml(TASKS[stage])
        input_folder = _prepare_input(read_path, arrays, stage, scratch)
        datapath = Datapath(files=[str(input_folder)], cls=None)

        data_config = _load_yaml(os.path.join("configs/data", task.components.data))
        transform_config = _load_yaml(os.path.join("configs/transform", task.components.transform))
        tokenizer_config = task.components.get("tokenizer", None)
        if tokenizer_config is not None:
            tokenizer_config = _load_yaml(os.path.join("configs/tokenizer", task.components.tokenizer))
            tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)
        data_name = "raw_data.npz"  # the kit's --data_name for the skin pass; the skeleton task's own value
        predict_dataset_config = data_config.get("predict_dataset_config", None)
        if predict_dataset_config is not None:
            predict_dataset_config = DatasetConfig.parse(config=predict_dataset_config).split_by_cls()
        predict_transform_config = transform_config.get("predict_transform_config", None)
        if predict_transform_config is not None:
            predict_transform_config = TransformConfig.parse(config=predict_transform_config)

        events.progress("loading_model")
        model_config = _load_yaml(os.path.join("configs/model", task.components.model))
        tokenizer = get_tokenizer(config=tokenizer_config) if tokenizer_config is not None else None
        model = get_model(tokenizer=tokenizer, **model_config)

        data = UniRigDatasetModule(
            process_fn=model._process_fn,
            train_dataset_config=None,
            predict_dataset_config=predict_dataset_config,
            predict_transform_config=predict_transform_config,
            validate_dataset_config=None,
            train_transform_config=None,
            validate_transform_config=None,
            tokenizer_config=tokenizer_config,
            debug=False,
            data_name=data_name,
            datapath=datapath,
            cls=None,
        )

        stash = _make_stash()
        callbacks = [stash]
        writer_config = task.get("writer", None)
        writer_config["npz_dir"] = str(scratch)
        writer_config["output_dir"] = None
        writer_config["output_name"] = str(scratch / f"{stage}.fbx")  # never written: the export is recorded
        writer_config["user_mode"] = True
        callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))

        trainer_config = dict(task.get("trainer", {}))
        trainer_config["devices"] = [device]  # the chosen card; upstream's devices: 1 is card 0
        system_config = _load_yaml(os.path.join("configs/system", task.components.system))
        system = get_system(**system_config, model=model, optimizer_config=None, loss_config=None,
                            scheduler_config=None, steps_per_epoch=1)

        # the checkpoint's path relative to UniRig's folder upstream, taken from the weights folder here
        checkpoint = weights_folder / download(task.get("resume_from_checkpoint", None))
        try:
            with torch.serialization.safe_globals([Box]):
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            system.on_load_checkpoint(state)
            system.load_state_dict(state["state_dict"], strict=True)
        except Exception as error:
            raise weights_failure(error, f"UniRig's {stage} weights would not load")
        del state

        trainer = L.Trainer(callbacks=callbacks, logger=False, default_root_dir=str(scratch),
                            enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False,
                            **trainer_config)

        recorder = capture.Recorder()
        Exporter._export_fbx = lambda self, *args, **kwargs: recorder(self, *args, **kwargs)
        # ARSystem.predict_step prints any error and returns nothing; keep the error to report it
        failures = []
        if hasattr(system, "_predict_step"):
            inner = system._predict_step

            def keeping_errors(*args, **kwargs):
                try:
                    return inner(*args, **kwargs)
                except Exception as error:
                    failures.append(error)
                    raise

            system._predict_step = keeping_errors
        torch.cuda.reset_peak_memory_stats(device)
        events.progress("generating" if stage == "skeleton" else "skinning")
        try:
            trainer.predict(system, datamodule=data, ckpt_path=None, return_predictions=False)
            torch.cuda.synchronize()
        except Exception as error:
            raise model_failure(error, f"UniRig's {stage} pass failed")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if failures:
        raise model_failure(failures[0], f"UniRig's {stage} pass failed")
    if len(recorder.calls) != 1:
        raise EngineError("rig_failed", f"UniRig's {stage} pass made {len(recorder.calls)} results instead of 1; "
                          "the log has its message")
    call = recorder.calls[0]
    rig_arrays = capture.to_arrays(call)
    bones = len(call["names"])
    info = {"stage": "skeleton" if stage == "skeleton" else "final", "bones": bones}
    fingerprints = {}
    if stash.batches:
        outputs, batch = stash.batches[0]
        fingerprints["points"] = fingerprint.fp(batch["vertices"])
        if stage == "skeleton" and outputs:
            fingerprints["tokens"] = fingerprint.tokens_fp(outputs[0].tokens)
            fingerprints["joints"] = fingerprint.joints_fp(outputs[0].joints)
        elif stage == "skin" and outputs and outputs.get("skin_pred"):
            fingerprints["skin"] = fingerprint.fp(outputs["skin_pred"][0])
    if stage == "skin":
        fingerprints["joints"] = fingerprint.joints_fp(call["joints"])
        fingerprints["skin_written"] = fingerprint.fp(np.asarray(call["skin"], dtype=np.float64))
    info["fingerprints"] = fingerprints

    if stage == "skeleton":
        names_source, skeleton = "model", "made"
        info["names"] = names_source
    elif keep_skeleton:
        file_names = plainnpz.as_texts(arrays["names"])
        file_bones = int(arrays["armature_bone_count"][()])
        kept = call["names"] == file_names and bones == file_bones
        info["keep_check"] = dict(file_bones=file_bones, bones=bones, names_equal=call["names"] == file_names)
        if not kept:
            raise EngineError("skeleton_not_kept", "UniRig did not give back every bone of the file's skeleton",
                              keep_check=info["keep_check"])
        names_source, skeleton = "kept", "kept"
        info["standard_names"] = {"asked": bool(standard_names), "matched": False, "layout": None,
                                  "placement_warnings": [], "note": "the file's names were kept"}
    else:
        skeleton = "made"
        standard, report = names.decide(rig_arrays["parents"].tolist(), call["joints"], standard_names)
        info["standard_names"] = report
        if standard is not None:
            rig_arrays["names"] = plainnpz.texts(standard)
            names_source = "standard"
        else:
            names_source = "numbered"
    info.update(skeleton=skeleton, names=names_source)
    log("export arguments: " + json.dumps(plainnpz.as_json(rig_arrays["export_options"])))
    rig_arrays.update(
        stage=plainnpz.text(info["stage"]),
        skeleton=plainnpz.text(skeleton),
        names_source=plainnpz.text(names_source),
    )
    return rig_arrays, info
