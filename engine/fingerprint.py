"""Short hashes of arrays, the same as the Windows kit's st_rig.py, so results can be compared."""
import hashlib


def fp(value):
    import numpy as np

    if hasattr(value, "detach"):  # a torch tensor, without importing torch here
        value = value.detach().float().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:12]


def joints_fp(joints):
    import numpy as np

    return fp(np.round(np.asarray(joints, dtype=np.float64), 4))


def tokens_fp(tokens):
    import numpy as np

    return fp(np.asarray(tokens, dtype=np.int64))
