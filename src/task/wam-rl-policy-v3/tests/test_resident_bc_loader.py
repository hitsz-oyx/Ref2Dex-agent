import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/run'))
from train_current_bc import load_resident_array


def test_chunked_resident_loader_preserves_every_label_and_partial_final_chunk(tmp_path):
    expected = np.random.default_rng(7).normal(size=(103, 565)).astype(np.float32)
    path = tmp_path/'x.npy'
    np.save(path, expected)
    loaded = load_resident_array(path, 'cpu', rows_per_chunk=17)
    assert loaded.dtype == torch.float32
    np.testing.assert_array_equal(loaded.numpy(), expected)
