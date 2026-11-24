# The pinned inputs of an engine folder. Every file is locked to one revision and checked by SHA-256
# before it is used. The list is the Windows kit's kit/fetch.py list, with the two PyG wheels left out
# (they are packages of the Python environment, not part of the engine folder).
#
# Plain data only, no imports beyond the standard library, so the build and the tests can read it.

ST = "https://huggingface.co/VAST-AI/SkinTokens/resolve/79736cad0fd84de384d5eede659b4ebd24effe33/"
UR = "https://huggingface.co/VAST-AI/UniRig/resolve/36842e2b5947e9e60f89275b83208c8e74071c63/"

SKINTOKENS_COMMIT = "273b691d35989d71cd17ff2895fdc735097b92d1"
UNIRIG_COMMIT = "6793c6640ff01c8fb389f3993434124bb43d2933"

# name in the downloads folder: (url, sha256, kind, where it goes in the engine folder or None)
# kind "code" is always needed; kind "weights" only for a folder that rigs.
SOURCES = {
    "skintokens.tar.gz": (
        f"https://github.com/VAST-AI-Research/SkinTokens/archive/{SKINTOKENS_COMMIT}.tar.gz",
        "f886ce830f8f6ed5a3eabebb9399244812ac17a44d5b51fe8853c381a214e334",
        "code", None),
    "unirig.tar.gz": (
        f"https://github.com/VAST-AI-Research/UniRig/archive/{UNIRIG_COMMIT}.tar.gz",
        "ebdda5907b520b6ccfa0524f5050c7141475d203289052e9c39024fae039c1a1",
        "code", None),
    "qwen3-config.json": (
        "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/config.json",
        "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        "code", "models/skintokens/models/Qwen3-0.6B/config.json"),
    "opt-350m-config.json": (
        "https://huggingface.co/facebook/opt-350m/resolve/08ab08cc4b72ff5593870b5d527cf4230323703c/config.json",
        "3ec75df0e166476e342cf95226ba775d8155449a7b1e430728b3ca26b425e4ab",
        "code", "models/unirig/hf/opt-350m/config.json"),
    "skintokens-grpo_1400.ckpt": (
        ST + "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
        "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692",
        "weights", "models/skintokens/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"),
    "skintokens-skin_vae-last.ckpt": (
        ST + "experiments/skin_vae_2_10_32768/last.ckpt",
        "4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4",
        "weights", "models/skintokens/experiments/skin_vae_2_10_32768/last.ckpt"),
    "unirig-skeleton-model.ckpt": (
        UR + "skeleton/articulation-xl_quantization_256/model.ckpt",
        "d8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4",
        "weights", "models/unirig/experiments/skeleton/articulation-xl_quantization_256/model.ckpt"),
    "unirig-skin-model.ckpt": (
        UR + "skin/articulation-xl/model.ckpt",
        "9d40cf42fb9d4c10d8b373d9f4f557c6fbbc5ebacae7ae6b3dce5a0d8a18bc33",
        "weights", "models/unirig/experiments/skin/articulation-xl/model.ckpt"),
}

# Where each code archive is unpacked in the engine folder.
ARCHIVES = {
    "skintokens.tar.gz": "models/skintokens",
    "unirig.tar.gz": "models/unirig",
}
