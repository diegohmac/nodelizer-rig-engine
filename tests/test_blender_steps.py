"""The read and write steps for real, with Blender (bpy 4.2.0, emulated x86_64 Linux here).

Without a graphics card the rig step cannot run, so rigs come from files the models already made on
the Windows computer: each is read back, turned into exactly the rig file the rig step would have
written, written onto the matching input character by our write step, and read back again. Joints,
parents, names and weights must come out as they went in.
"""
import shutil

import numpy as np
import pytest

import rigfiles
from conftest import run_python, run_step
from glbrig import GlbRig

ST_RODIN = "output/st_fresh/mine_rodin_a/rigged.glb"         # SkinTokens, Windows, 52 bones
UR_RODIN_SKIN = "output/ur_full/mine_rodin_a/skin.fbx"       # UniRig's own skin export, Windows
UR_RODIN_RIGGED = "output/ur_full/mine_rodin_a/rigged.glb"   # UniRig's merge of it onto mine_rodin.glb
UR_RODIN_SKELETON = "output/ur_full/mine_rodin_a/skeleton.fbx"
EDITED = "edited/example_tira_tail.fbx"                      # 58 bones: UniRig's 52 plus a hand made tail


def read(engine, work, model, source, name):
    out = work / f"{name}.{model}.read.npz"
    run = run_step(engine, "read", "--model", model, "--input", source, "--output", out, "--work", work)
    return run.result, rigfiles.load(out)


def write(engine, work, rig_arrays, source, name, suffix=".glb"):
    rig_path = rigfiles.save(work / f"{name}.rig.npz", rig_arrays)
    out = work / f"{name}{suffix}"
    args = ["write", "--rig", rig_path, "--output", out, "--work", work]
    if source is not None:
        args += ["--input", source]
    run = run_step(engine, *args)
    return run, out


def test_read_skintokens_plain_mesh(engine, inputs, work):
    result, arrays = read(engine, work, "skintokens", inputs / "mine_rodin.glb", "rodin")
    assert result["has_skeleton"] is False and result["bone_count"] == 0
    assert arrays["vertices"].dtype == np.float64 and arrays["faces"].dtype == np.int32
    assert "joint_names" not in arrays and "skin" not in arrays
    assert result["forbidden_modules_loaded"] == []
    leftovers = sorted(p.name for p in work.iterdir())
    assert leftovers == ["cache", "rodin.skintokens.read.npz", "tmp"], leftovers
    assert not any((work / "tmp").iterdir()), "Blender's temp folder was left behind"


def test_read_matches_skintokens_own_loader(engine, inputs, work):
    """The read file equals what BpyParser.load returns in a process of its own, array for array."""
    _, arrays = read(engine, work, "skintokens", inputs / EDITED, "tira_tail")
    direct = work / "direct.npz"
    run_python(
        "import sys, numpy as np\n"
        "sys.path.insert(0, sys.argv[1] + '/models/skintokens'); sys.path.insert(0, sys.argv[1])\n"
        "from src.rig_package.parser.bpy import BpyParser\n"
        "from engine.skintokens import asset_io\n"
        "asset = BpyParser.load(filepath=sys.argv[2])\n"
        "np.savez(sys.argv[3], **asset_io.to_arrays(asset))\n",
        engine, inputs / EDITED, direct)
    with np.load(direct, allow_pickle=False) as data:
        assert sorted(data.files) == sorted(k for k in arrays if k in data.files)
        for key in data.files:
            assert data[key].dtype == arrays[key].dtype, key
            np.testing.assert_array_equal(data[key], arrays[key], err_msg=key)
    assert int(arrays["bone_count"]) == 58
    direct_count = run_python(
        "import sys, json, bpy\n"
        "bpy.ops.import_scene.fbx(filepath=sys.argv[1], ignore_leaf_bones=False)\n"
        "arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']\n"
        "print(json.dumps({'armatures': len(arms), 'bones': sum(len(a.data.bones) for a in arms)}))\n",
        inputs / EDITED)
    assert direct_count == {"armatures": 1, "bones": 58}
    names = rigfiles.plainnpz.as_texts(arrays["joint_names"])
    assert names[:2] == ["bone_0", "bone_1"] and names[-1] == "tail_5"


def test_read_matches_unirig_extract(engine, inputs, downloads, work, scratch):
    """The read file equals what UniRig's own, unpatched extract saves for the same file."""
    _, arrays = read(engine, work, "unirig", inputs / EDITED, "tira_tail")
    pristine = scratch / "pristine-unirig"
    if not pristine.exists():
        run_python(
            "import sys, tarfile\n"
            "from pathlib import Path\n"
            "with tarfile.open(sys.argv[1]) as tar:\n"
            "    members = []\n"
            "    for m in tar.getmembers():\n"
            "        parts = Path(m.name).parts[1:]\n"
            "        if parts:\n"
            "            m.name = '/'.join(parts); members.append(m)\n"
            "    tar.extractall(sys.argv[2], members=members, filter='data')\n",
            downloads / "unirig.tar.gz", pristine)
    direct = work / "direct.npz"
    run_python(
        "import sys, numpy as np\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from src.data import log, raw_data\n"
        "from src.data.extract import clean_bpy, load, get_arranged_bones, process_mesh, process_armature, save_raw_data\n"
        "kept = {}\n"
        "raw_data.RawData.save = lambda self, path: kept.update(self.__dict__)\n"
        "log.new_log(sys.argv[3], 'x'); log.new_entry(sys.argv[2]); clean_bpy()\n"
        "arm = load(sys.argv[2]); bones = get_arranged_bones(arm)\n"
        "v, f, s = process_mesh(bones)\n"
        "j, t, p, n, m = process_armature(arm, bones)\n"
        "save_raw_data(path='unused', vertices=v, faces=f-1, skin=s, joints=j, tails=t, parents=p, names=n,\n"
        "              matrix_local=m, target_count=50000)\n"
        "kept['parents'] = np.array([-1 if x is None else x for x in kept['parents']], dtype=np.int64)\n"
        "kept['names'] = np.array(kept['names'])\n"
        "np.savez(sys.argv[4], **{k: v for k, v in kept.items() if v is not None})\n",
        pristine, inputs / EDITED, work, direct)
    with np.load(direct, allow_pickle=False) as data:
        for key in data.files:
            assert data[key].dtype == arrays[key].dtype, key
            np.testing.assert_array_equal(data[key], arrays[key], err_msg=key)
    assert int(arrays["armature_bone_count"]) == 58 and len(arrays["names"]) == 58
    assert len(arrays["faces"]) <= 50000


@pytest.mark.parametrize("character,bones", [("mine_rodin", 52), ("mine_trellis", 49)])
def test_write_skintokens_round_trip(engine, inputs, native, work, character, bones):
    """A SkinTokens rig made on Windows, written onto its character again, comes out unchanged, with
    the character's own materials and textures."""
    from engine import glbinfo

    reference = native / f"output/st_fresh/{character}_a/rigged.glb"
    _, rigged = read(engine, work, "skintokens", reference, "reference")
    _, target = read(engine, work, "skintokens", inputs / f"{character}.glb", "target")
    rig = rigfiles.skintokens_rig_onto(target, rigged)
    run, out = write(engine, work, rig, inputs / f"{character}.glb", character)
    result = run.result
    assert result["bones"] == bones and result["forbidden_modules_loaded"] == []
    source = glbinfo.summary(inputs / f"{character}.glb")
    assert (result["materials"], result["images"]) == (source["materials"], source["images"])
    got, want = GlbRig(out), GlbRig(reference)
    assert got.names == want.names and got.parents == want.parents
    np.testing.assert_allclose(got.joints, want.joints, atol=1e-5)
    assert got.weights.shape == want.weights.shape
    assert np.abs(got.weights - want.weights).max() < 1e-4


def test_write_is_repeatable(engine, inputs, native, work):
    _, rigged = read(engine, work, "skintokens", native / ST_RODIN, "reference")
    rig = rigfiles.skintokens_rig_from_read(rigged)
    _, first = write(engine, work, rig, inputs / "mine_rodin.glb", "first")
    _, second = write(engine, work, rig, inputs / "mine_rodin.glb", "second")
    a, b = GlbRig(first), GlbRig(second)
    np.testing.assert_array_equal(a.joints, b.joints)
    np.testing.assert_array_equal(a.weights, b.weights)


def test_write_standard_names(engine, inputs, final, work):
    """A skeleton that matched the Mixamo layout (Docker round, work-B: mine_rodin seed 12345) is written
    with the Mixamo names."""
    from engine import names as standard

    reference = final / "work-B/runs/mine_rodin/articulation_seed_12345/rigged.glb"
    _, rigged = read(engine, work, "skintokens", reference, "reference")
    matched = standard.match(rigged["parents"].tolist())
    assert matched is not None and matched[0] == "full52-dfs"
    rig = rigfiles.skintokens_rig_from_read(rigged, names=matched[1], names_source="standard")
    run, out = write(engine, work, rig, inputs / "mine_rodin.glb", "rodin")
    assert run.result["bones"] == 52
    got = GlbRig(out)
    assert got.names[0] == "mixamorig:Hips" and "mixamorig:RightHandPinky3" in got.names
    assert got.parents == GlbRig(reference).parents


def test_write_unirig_skin_round_trip(engine, inputs, native, work):
    """UniRig's skin pass rig, from its own skin .fbx on Windows, exported and merged by our write step,
    gives the bones and weights of UniRig's own merge."""
    _, skin = read(engine, work, "unirig", native / UR_RODIN_SKIN, "skin")
    rig = rigfiles.unirig_rig_from_read(skin, "final")
    run, out = write(engine, work, rig, inputs / "mine_rodin.glb", "rodin")
    assert run.result["bones"] == 45
    got, want = GlbRig(out), GlbRig(native / UR_RODIN_RIGGED)
    assert got.names == want.names and got.parents == want.parents
    np.testing.assert_allclose(got.joints, want.joints, atol=1e-4)
    top_got, top_want = got.weights.argmax(1), want.weights.argmax(1)
    assert (top_got == top_want).mean() > 0.995
    assert np.abs(got.weights - want.weights).mean() < 2e-3


def test_write_unirig_skeleton_pass(engine, native, work):
    """UniRig's skeleton pass: the recorded export written as a skeleton .fbx, which the next round
    reads back with the same bones."""
    _, skeleton = read(engine, work, "unirig", native / UR_RODIN_SKELETON, "skeleton")
    rig = rigfiles.unirig_rig_from_read(skeleton, "skeleton", names_source="model")
    run, out = write(engine, work, rig, None, "skeleton", suffix=".fbx")
    assert run.code == 0, run
    _, again = read(engine, work, "unirig", out, "again")
    assert rigfiles.plainnpz.as_texts(again["names"]) == rigfiles.plainnpz.as_texts(skeleton["names"])
    np.testing.assert_array_equal(again["parents"], skeleton["parents"])
    np.testing.assert_allclose(again["joints"], skeleton["joints"], atol=1e-5)
    np.testing.assert_allclose(again["vertices"], skeleton["vertices"], atol=1e-5)


def test_write_unirig_names_with_colons(engine, inputs, native, work):
    """Mixamo names hold a colon, which FBX treats specially; they must survive UniRig's .fbx round trip."""
    _, skin = read(engine, work, "unirig", native / UR_RODIN_SKIN, "skin")
    names = [f"mixamorig:Bone{i}" for i in range(len(skin["names"]))]
    rig = rigfiles.unirig_rig_from_read(skin, "final", names=names, names_source="standard")
    run, out = write(engine, work, rig, inputs / "mine_rodin.glb", "rodin")
    assert run.code == 0, run
    assert sorted(GlbRig(out).names) == sorted(names)


@pytest.mark.parametrize("model", ["skintokens", "unirig"])
def test_keep_skeleton_names_survive(engine, inputs, native, work, model):
    """The hand fixed skeleton (it has bones but no weights), skinned as each model skinned it on
    Windows, written onto the file itself: all 58 names are kept (the keep skeleton case)."""
    _, edited = read(engine, work, model, inputs / EDITED, "tira_tail")
    if model == "skintokens":
        _, skinned = read(engine, work, model, native / "output/st_edited/tira_tail/rigged.glb", "skinned")
        rig = rigfiles.skintokens_rig_onto(edited, skinned, skeleton="kept", names_source="kept")
        file_names = rigfiles.plainnpz.as_texts(edited["joint_names"])
    else:
        _, skinned = read(engine, work, model, native / "output/ur_edited/tira_tail/skin.fbx", "skinned")
        rig = rigfiles.unirig_rig_from_read(skinned, "final", skeleton="kept", names_source="kept")
        file_names = rigfiles.plainnpz.as_texts(edited["names"])
    run, out = write(engine, work, rig, inputs / EDITED, "tira_tail")
    assert run.result["bones"] == 58
    assert sorted(GlbRig(out).names) == sorted(file_names)


def test_write_reports_unweighted_vertices(engine, inputs, work):
    """A skin that leaves vertices without any weight makes Blender's glTF export add a bone of its
    own, "neutral_bone"; the result says so."""
    _, edited = read(engine, work, "skintokens", inputs / EDITED, "tira_tail")
    rig = rigfiles.skintokens_rig_from_read(edited, skeleton="kept", names_source="kept")
    run, out = write(engine, work, rig, inputs / EDITED, "tira_tail")
    assert run.result["bones"] == 58 and run.result["extra_bones"] == ["neutral_bone"]
    assert out.exists()


def test_odd_paths(engine, inputs, scratch):
    """Spaces and accents in every path, as Windows user folders often have."""
    folder = scratch / "Zé da Silva" / "rig kit"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    source = folder / "Zé modelo 1 ção.glb"
    shutil.copyfile(inputs / "mine_rodin.glb", source)
    result, arrays = read(engine, folder, "skintokens", source, "modelo ção")
    assert result["vertices"] == len(arrays["vertices"])
    assert result["armatures"] == 0 and result["armature_bones"] == 0
    result, _ = read(engine, folder, "unirig", source, "modelo ção")
    assert result["faces"] > 0


def _obj_bundle(character_glb, folder):
    """An unpacked OBJ bundle as the app hands it: the .obj with its .mtl and textures beside it, made by
    Blender from a character that has a texture."""
    folder.mkdir(parents=True)
    run_python(
        "import sys, bpy\n"
        "for thing in list(bpy.data.objects):\n"
        "    bpy.data.objects.remove(thing)\n"
        "bpy.ops.import_scene.gltf(filepath=sys.argv[1])\n"
        "for image in bpy.data.images:\n"
        "    if image.type != 'IMAGE':\n"
        "        continue\n"
        "    image.filepath_raw = sys.argv[2] + '/' + image.name.replace(' ', '_') + '.png'\n"
        "    image.file_format = 'PNG'\n"
        "    image.save()\n"
        "bpy.ops.wm.obj_export(filepath=sys.argv[2] + '/model.obj', export_materials=True, path_mode='RELATIVE')\n"
        "import os; sys.stdout.flush(); os._exit(0)  # skip Blender's shutdown, which crashes under emulation\n",
        character_glb, folder)
    files = sorted(p.suffix for p in folder.iterdir())
    assert ".mtl" in files and ".png" in files, files
    # point the material at the texture beside it, as a bundle has it
    mtl = folder / "model.mtl"
    lines = mtl.read_text(encoding="utf-8").splitlines()
    texture = next(p.name for p in folder.iterdir() if p.suffix == ".png")
    mtl.write_text("\n".join(f"map_Kd {texture}" if line.startswith("map_Kd") else line for line in lines) + "\n",
                   encoding="utf-8")
    return folder / "model.obj"


@pytest.mark.parametrize("model", ["skintokens", "unirig"])
def test_obj_bundle(engine, inputs, native, work, model):
    """read and write on an OBJ with its .mtl and texture beside it: the rigged .glb carries the
    material and the texture."""
    obj = _obj_bundle(inputs / "mine_trellis.glb", work / "bundle Zé")
    result, target = read(engine, work, model, obj, "bundle")
    assert result["faces"] > 0 and result["has_skeleton"] is False
    if model == "skintokens":
        _, rigged = read(engine, work, model, native / "output/st_fresh/mine_trellis_a/rigged.glb", "reference")
        rig = rigfiles.skintokens_rig_onto(target, rigged)
    else:  # any UniRig skin pass rig: merge fits it onto the target
        _, skin = read(engine, work, model, native / UR_RODIN_SKIN, "skin")
        rig = rigfiles.unirig_rig_from_read(skin, "final")
    run, out = write(engine, work, rig, obj, "bundle")
    result = run.result
    assert result["materials"] >= 1 and result["images"] >= 1, result
    assert result["bones"] == len(rig["names"])
    from glbrig import _load

    doc, blob = _load(out)
    for image in doc["images"]:  # the texture's pixels are inside the file, not just its name
        view = doc["bufferViews"][image["bufferView"]]
        assert view["byteLength"] > 1000 and image["mimeType"] in ("image/png", "image/jpeg")


def test_long_paths(engine, inputs, scratch):
    """A work folder about 230 characters deep, with accents, and a long file name."""
    import shutil as sh

    base = scratch / "long"
    if base.exists():
        sh.rmtree(base)
    folder = base
    while len(str(folder)) < 200:
        folder = folder / "pasta de trabalho çã"
    folder.mkdir(parents=True)
    source = folder / ("modelo " * 3 + "ção.glb")
    sh.copyfile(inputs / "mine_rodin.glb", source)
    assert len(str(source)) > 220
    result, arrays = read(engine, folder, "skintokens", source, "r")
    two_bones = np.tile(np.eye(4), (2, 1, 1))
    two_bones[1, :3, 3] = arrays["vertices"].max(0)  # a second joint away from the first
    rig = rigfiles.skintokens_rig_onto(arrays, arrays | {"skin": np.full((len(arrays["vertices"]), 2), 0.5),
                                                          "matrix_local": two_bones,
                                                          "parents": np.array([-1, 0], np.int32),
                                                          "joint_names": rigfiles.plainnpz.texts(["a", "b"])})
    run, out = write(engine, folder, rig, source, "saída rigged")
    assert run.result["bones"] == 2 and len(str(out)) > 220


def test_unirig_upper_case_endings(engine, inputs, work):
    """UniRig's loaders match endings by case: a .GLB is read from a copy, other upper case endings
    are refused clearly instead of failing inside Blender."""
    upper = work / "MODEL.GLB"
    shutil.copyfile(inputs / "mine_rodin.glb", upper)
    result, _ = read(engine, work, "unirig", upper, "upper")
    assert result["faces"] > 0
    (work / "MODEL.OBJ").write_text("v 0 0 0\n", encoding="utf-8")
    run = run_step(engine, "read", "--model", "unirig", "--input", work / "MODEL.OBJ", "--output",
                   work / "o.npz", "--work", work)
    assert run.code == 1 and run.last["code"] == "bad_input", run.last


def test_read_refuses_a_file_without_mesh(engine, work):
    empty = work / "empty.obj"
    empty.write_text("# nothing\n", encoding="utf-8")
    run = run_step(engine, "read", "--model", "skintokens", "--input", empty, "--output", work / "e.npz", "--work", work)
    assert run.code == 1 and run.last["code"] in ("no_mesh", "read_failed"), run.last
