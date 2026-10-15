"""Batch reset suppression must preserve native per-motion reference clamping."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location('batch_collector',
    Path(__file__).resolve().parents[1]/'tools/run/collect_rl660_batch.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def test_variable_length_reference_queries_equal_original_native_clamp():
    original = np.arange(2*9*4).reshape(2,9,4).copy()
    lengths = np.array([4,9])
    task = SimpleNamespace(hoi_data=original.copy(),hoi_refs=original[:,None].copy())
    collector.hold_terminal_references(task,lengths)
    for seq, length in enumerate(lengths):
        np.testing.assert_array_equal(task.hoi_data[seq,:length],original[seq,:length])
        for tick in range(length):
            for delta in [0,1,16]:
                expected = original[seq,min(tick+delta,length-1)]
                np.testing.assert_array_equal(task.hoi_data[seq,min(tick+delta,8)],expected)
                np.testing.assert_array_equal(task.hoi_refs[seq,0,min(tick+delta,8)],expected)
