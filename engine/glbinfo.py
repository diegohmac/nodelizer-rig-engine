"""Reads a .glb's JSON part, standard library only: the write step's own check of what it wrote."""
import json
import struct


def summary(path):
    with open(path, "rb") as handle:
        header = handle.read(20)
        magic, _version, _length, chunk_length, chunk_type = struct.unpack("<4sIIII", header)
        if magic != b"glTF" or chunk_type != 0x4E4F534A:
            raise ValueError("not a binary glTF file")
        doc = json.loads(handle.read(chunk_length))
    nodes = doc.get("nodes", [])
    skins = doc.get("skins", [])
    joints = [nodes[j].get("name", "") for j in skins[0]["joints"]] if skins else []
    return {
        "meshes": len(doc.get("meshes", [])),
        "materials": len(doc.get("materials", [])),
        "images": len(doc.get("images", [])),
        "skins": len(skins),
        "joints": joints,
    }
