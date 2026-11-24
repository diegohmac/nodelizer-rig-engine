"""Seeds. Same seed, same computer, same settings: the same rig.

The app gives every run a seed (0 to 2**32 - 1, the range NumPy and Lightning accept). SkinTokens is
seeded right before its one job, after the model has loaded, so loading never changes the rig.

SkinTokens' own runs loaded the model file in a data loader worker. The engine has no workers (the
sandbox allows one process), so the one fetch is seeded the way PyTorch seeds a worker
(torch/utils/data/_utils/worker.py) and this process's random state is put back after it. That gives
the same points, and so the same rig, as a worker would (Windows round, work-1).
"""
import random

SEED_MAX = 2**32 - 1


def check(seed):
    if not 0 <= seed <= SEED_MAX:
        raise ValueError(f"the seed must be between 0 and {SEED_MAX}")
    return seed


def seed_all(seed, torch=None):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def fetch_like_a_worker(batches, torch):
    """next(batches) for a DataLoader iterator with no workers, seeded as worker 0 would be."""
    import numpy as np
    from torch.utils.data._utils.worker import _generate_state

    base = batches._base_seed
    py_state, np_state, threads = random.getstate(), np.random.get_state(), torch.get_num_threads()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            torch.set_num_threads(1)
            random.seed(base)
            torch.manual_seed(base)
            np.random.seed(_generate_state(base, 0))
            return next(batches)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_num_threads(threads)
