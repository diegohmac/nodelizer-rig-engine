"""A small reader for the rig in a .glb (first skin): bone names, parents, world joint positions and
per vertex weights. NumPy only, for comparing results in the tests. Adapted from the Docker rounds'
rigload.py."""
import json
import struct

import numpy as np

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _load(path):
    with open(path, "rb") as handle:
        data = handle.read()
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + length]), memoryview(data)[20 + length + 8:]


def _accessor(doc, blob, index):
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT[accessor["componentType"]])
    width = WIDTH[accessor["type"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    count = accessor["count"]
    if stride == dtype.itemsize * width:
        array = np.frombuffer(blob, dtype, count * width, offset).reshape(count, width)
    else:
        raw = np.frombuffer(blob, np.uint8, stride * (count - 1) + dtype.itemsize * width, offset)
        index = (np.arange(count)[:, None] * stride + np.arange(dtype.itemsize * width)[None]).ravel()
        array = raw[index].view(dtype).reshape(count, width)
    if accessor.get("normalized"):
        array = array.astype(np.float32) / np.iinfo(dtype).max
    return array


def _local(node):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    t = np.array(node.get("translation", [0, 0, 0]), float)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), float)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * s
    m[:3, 3] = t
    return m


class GlbRig:
    def __init__(self, path):
        doc, blob = _load(path)
        nodes = doc["nodes"]
        parent = {}
        for i, node in enumerate(nodes):
            for child in node.get("children", []):
                parent[child] = i
        world = {}

        def go(i):
            if i not in world:
                world[i] = (go(parent[i]) if i in parent else np.eye(4)) @ _local(nodes[i])
            return world[i]

        skin = doc["skins"][0]
        self.joint_nodes = skin["joints"]
        index = {node: k for k, node in enumerate(self.joint_nodes)}
        self.names = [nodes[j].get("name", "") for j in self.joint_nodes]
        self.parents = [index.get(parent.get(j), -1) for j in self.joint_nodes]
        self.joints = np.array([go(j)[:3, 3] for j in self.joint_nodes])
        vertices, joints, weights = [], [], []
        for i, node in enumerate(nodes):
            if "mesh" not in node or "skin" not in node:
                continue
            for primitive in doc["meshes"][node["mesh"]]["primitives"]:
                attributes = primitive["attributes"]
                vertices.append(_accessor(doc, blob, attributes["POSITION"]).astype(float))
                joints.append(_accessor(doc, blob, attributes["JOINTS_0"]).astype(np.int64))
                weights.append(_accessor(doc, blob, attributes["WEIGHTS_0"]).astype(float))
        self.vertices = np.vstack(vertices)
        dense = np.zeros((len(self.vertices), len(self.names)))
        rows = np.repeat(np.arange(len(self.vertices)), 4)
        np.add.at(dense, (rows, np.vstack(joints).ravel()), np.vstack(weights).ravel())
        self.weights = dense / np.maximum(dense.sum(1, keepdims=True), 1e-9)
