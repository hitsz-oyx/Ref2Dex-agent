"""Exact lookup differential tests, including signed int64 boundary cases."""
import numpy as np
import pytest

from pose_mesh_data import v4_interaction as v4


def reference_counts(cells, counts, queries):
    merged, inverse = np.unique(np.concatenate((cells, queries.reshape(-1, 3))),
                                axis=0, return_inverse=True)
    lookup = np.zeros((len(merged), 6), np.float32)
    lookup[inverse[:len(cells)]] = counts
    return lookup[inverse[len(cells):]].reshape(*queries.shape[:-1], 6).sum(1)


@pytest.mark.parametrize('seed', range(12))
def test_integer_lookup_is_exact_for_negative_shifted_and_empty_cells(seed):
    rng = np.random.default_rng(seed)
    for shift in (0, -10**12, 10**12):
        cells = np.unique(rng.integers(-12, 12, (100, 3)), axis=0) + shift
        counts = rng.integers(0, 50, (len(cells), 6)).astype(np.float32)
        queries = rng.integers(-15, 15, (128, 27, 3)) + shift
        queries[:len(cells), 0] = cells
        for n in (0, len(cells)):
            np.testing.assert_array_equal(v4._query_voxel_counts(cells[:n], counts[:n], queries),
                                          reference_counts(cells[:n], counts[:n], queries))


def test_integer_lookup_rejects_out_of_bounds_before_encoding_and_falls_back_on_overflow():
    limit = np.iinfo(np.int64)
    for cells in (np.array([[0, 0, 0], [2, 2, 2]], np.int64),
                  np.array([[limit.min, 0, 0], [limit.max, 0, 0]], np.int64),
                  np.array([[0, 0, 0], [2**32, 2**32, 1]], np.int64)):
        counts = np.arange(len(cells)*6, dtype=np.float32).reshape(-1, 6)
        queries = np.array([[[limit.min, 0, 0], [limit.max, 0, 0], [0, 0, 0],
                             cells[0], cells[-1], [0, 0, 3]]], np.int64)
        np.testing.assert_array_equal(v4._query_voxel_counts(cells, counts, queries),
                                      reference_counts(cells, counts, queries))


@pytest.mark.parametrize('dilation', [0, 1, 2])
def test_complete_labels_and_missing_hand_masks_match_reference(monkeypatch, dilation):
    rng = np.random.default_rng(192)
    hand = rng.normal(0, .025, (24, 2, 11, 3)).astype(np.float32)
    valid = np.ones((24, 2, 11), bool)
    valid[2, 0, 3] = False
    valid[:, 1] = False
    queries = rng.normal(0, .035, (24, 128, 3)).astype(np.float32)
    policy = v4.V4SamplingPolicy(dilation=dilation)
    origin = v4.interaction_origin(queries[0])
    actual = v4.interaction_field(hand, valid, queries, policy, origin=origin,
                                  active_sides=np.array([True, False]))
    monkeypatch.setattr(v4, '_query_voxel_counts', reference_counts)
    expected = v4.interaction_field(hand, valid, queries, policy, origin=origin,
                                    active_sides=np.array([True, False]))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
