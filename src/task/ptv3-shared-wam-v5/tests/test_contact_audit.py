from pathlib import Path
import sys
import numpy as np
import pytest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v5.contact_audit import (
    contact_counts, future_exposure, normal_candidate, opposite_surface_proxy, add_counts, empty_counts)
from ptv3_shared_wam_v5.preparation import surface_contact


def test_thin_slab_back_face_is_suppressed_without_changing_top_contact_or_part():
    # 3mm slab, hand 2mm above its top: opposite face is only5mm away.
    hand = np.array([[0., 0., .002]])
    queries = np.array([[0., 0., 0.], [0., 0., -.003]])
    normals = np.array([[0., 0., 1.], [0., 0., -1.]])
    result = normal_candidate(hand, [6], queries, normals)
    np.testing.assert_array_equal(result['baseline'], surface_contact(hand, np.array([6]), queries)[0])
    assert result['candidate'][0] == 1
    assert result['baseline'][1] > .8 and result['candidate'][1] < .2
    np.testing.assert_array_equal(result['nearest_part'], [6, 6])
    np.testing.assert_array_equal(opposite_surface_proxy(queries, normals), [True, True])
    # Joint rigid transform must preserve the direction test and labels.
    rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    shifted = normal_candidate(hand@rotation.T+2, [6], queries@rotation.T+2, normals@rotation.T)
    np.testing.assert_array_equal(result['candidate'], shifted['candidate'])


def test_normal_candidate_handles_touching_penetration_tolerance_and_unknown_normals():
    vertices = np.array([[0., 0., 0.]])
    queries = np.array([[0., 0., 0.], [0., 0., .0005], [0., 0., .005], [0., 0., .005]])
    normals = np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 0.], [np.nan, 0., 1.]])
    r = normal_candidate(vertices, [0], queries, normals)
    np.testing.assert_array_equal(r['normal_valid'], [True, True, False, False])
    np.testing.assert_array_equal(r['candidate'], r['baseline'])
    assert np.isfinite(r['candidate']).all()


def test_prevalence_excludes_invalid_geometry_and_weights_repeated_k24_targets():
    values = np.array([[[0., .3, .7]], [[np.nan, -1., 5.]], [[1., 0., 0.]]])
    valid = np.array([[True], [False], [True]])
    parts = np.array([[[0, 2, 3]], [[-9, -9, -9]], [[2, 0, 0]]])
    counts = contact_counts(values, valid, parts)
    assert counts['queries'] == 6 and counts['positive'] == [3, 3, 2]
    assert counts['positive_part_counts'][2] == 2
    weighted = contact_counts(values, valid, exposure=np.array([[2], [100], [0]]))
    assert weighted['queries'] == 6 and weighted['positive'] == [4, 4, 2]
    assert weighted['soft_mass'] == pytest.approx(2.)
    total = empty_counts(); add_counts(total, counts); add_counts(total, counts)
    assert total['queries'] == 12
    with pytest.raises(ValueError, match='finite'):
        contact_counts(values, np.ones_like(valid))


def test_future_exposure_keeps_all24_steps_and_separate_object_slots():
    rows = np.array([[0, 3], [0, 4], [1, 3]])
    result = future_exposure(rows, 29, 2)
    brute = np.zeros((29, 2), int)
    for slot, tick in rows: brute[tick+1:tick+25, slot] += 1
    np.testing.assert_array_equal(result, brute)
    assert result.sum() == 3*24
    with pytest.raises(ValueError, match='outside'):
        future_exposure(np.array([[0, 5]]), 29, 2)
