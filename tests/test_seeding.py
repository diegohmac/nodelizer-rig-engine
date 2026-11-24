"""Seed handling: the fetch with no data loader workers gives what a worker would, and leaves this
process's random state as it was."""
import pytest

from conftest import REPO, run_python

CHILD = """
import json, random, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from engine import seeding

class Draws(Dataset):
    def __len__(self):
        return 1
    def __getitem__(self, index):
        return torch.tensor([random.random(), float(np.random.rand()), float(torch.rand(1)), float(torch.get_num_threads())])

def states():
    return [random.getstate()[1][:4], np.random.get_state()[1][:4].tolist(), torch.get_rng_state()[:16].tolist()]

seed = int(sys.argv[2])
seeding.seed_all(seed, torch)
worker = next(iter(DataLoader(Draws(), batch_size=1, num_workers=1))).tolist()
seeding.seed_all(seed, torch)
batches = iter(DataLoader(Draws(), batch_size=1, num_workers=0))
before = states()
alike = seeding.fetch_like_a_worker(batches, torch).tolist()
after = states()
seeding.seed_all((seed + 1) % 2**32, torch)
other = seeding.fetch_like_a_worker(iter(DataLoader(Draws(), batch_size=1, num_workers=0)), torch).tolist()
print(json.dumps({"worker": worker, "alike": alike, "restored": before == after, "other": other}))
"""


@pytest.mark.parametrize("seed", [12345, 0, 2**32 - 1])
def test_fetch_like_a_worker(seed):
    result = run_python(CHILD, REPO, seed)
    assert result["alike"] == result["worker"]
    assert result["restored"]
    assert result["other"] != result["alike"]


def test_seed_range():
    import sys

    sys.path.insert(0, str(REPO))
    from engine import seeding

    assert seeding.check(0) == 0 and seeding.check(2**32 - 1) == 2**32 - 1
    for bad in (-1, 2**32):
        with pytest.raises(ValueError):
            seeding.check(bad)
