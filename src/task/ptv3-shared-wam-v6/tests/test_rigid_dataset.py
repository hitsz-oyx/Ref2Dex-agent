"""Admission must accept real reader contracts and reject semantic substitutions."""
import copy

import numpy as np

from ptv3_shared_wam_v5.contracts import JOINT_NAMES, LABEL_SEMANTICS, PART_NAMES
from ptv3_shared_wam_v5.data import sample_window
from ptv3_shared_wam_v6.rigid_dataset import admission_reasons, geometry_class, duplicate_of


def bundle():
    n = 32
    seq = dict(joints=np.zeros((n, 2, 21, 3), np.float32),
               joint_valid=np.ones((n, 2, 21), bool),
               surface_offsets=np.zeros((n, 2, 778, 3), np.float16),
               surface_valid=np.ones((n, 2), bool),
               surface_parts=np.zeros((2, 778), np.uint8),
               object_poses=np.tile(np.eye(4), (n, 1, 1, 1)),
               object_pose_valid=np.ones((n, 1), bool))
    geo = dict(surface=np.zeros((2048, 3)), normals=np.tile([0., 0., 1.], (2048, 1)),
               query=np.zeros((32, 3)))
    contact = dict(contact=np.zeros((n, 1, 2048), np.float16),
                   part=np.zeros((n, 1, 2048), np.uint8), valid=np.ones((n, 1), bool))
    event = dict(state=np.zeros((n, 1), np.int8), event_type=np.zeros((n, 1), np.uint8),
                 event_delta_steps=np.zeros((n, 1), np.int32), event_valid=np.ones((n, 1), bool))
    evidence = dict(joint_semantics=list(JOINT_NAMES), part_semantics=list(PART_NAMES),
                    label_semantics=LABEL_SEMANTICS, binding_verified=True,
                    event_binding_verified=True, geometry_binding_verified=True)
    return seq, geo, contact, event, evidence


def test_admitted_contract_reaches_actual_reader():
    seq, geo, contact, event, evidence = bundle()
    assert admission_reasons(seq, geo, contact, event, evidence, 3, 0) == []
    assert sample_window(seq, geo, contact, 3, 0, 1, [0, 0, 3])['trajectory_target'].shape == (24, 2, 21, 3)


def test_semantic_permutation_and_missing_binding_rejected():
    args = bundle()
    args[-1]['joint_semantics'] = list(reversed(JOINT_NAMES))
    args[-1]['geometry_binding_verified'] = False
    reasons = admission_reasons(*args, 3, 0)
    assert 'hand21_semantics_unbound' in reasons
    assert 'surface_registry_unbound' in reasons


def test_all_four_blockers_reported_without_keyerror():
    reasons = admission_reasons({}, {}, {}, {}, {}, 3, 0)
    assert {'hand21_missing', 'joint_valid_missing', 'object_validity_missing',
            'contact_part_valid_missing', 'event_missing', 'surface_registry_missing'} <= set(reasons)


def test_validity_cannot_hide_bad_observed_values_or_tail():
    args = bundle()
    args[0]['joints'][4, 0, 0, 0] = np.nan
    assert 'hand21_nonfinite_valid' in admission_reasons(*args, 3, 0)
    assert 'h4_k24_out_of_bounds' in admission_reasons(*bundle(), 8, 0)


def test_contact_proxy_or_part_corruption_never_admitted():
    args = bundle()
    args[2]['part'][4, 0, 0] = 16
    args[-1]['label_semantics'] = 'distance_2cm'
    reasons = admission_reasons(*args, 3, 0)
    assert 'contact_semantics_unbound' in reasons
    assert 'part_out_of_range' in reasons


def test_explicit_unknown_masks_do_not_fabricate_negatives():
    args = bundle()
    args[2]['valid'][:] = False
    args[3]['event_valid'][:] = False
    args[3]['state'][:] = -1
    assert admission_reasons(*args, 3, 0) == []


def test_duplicate_requires_geometry_motion_contact_and_masks():
    a = dict(shape=np.array([.1, .05, .04]), phase='hold',
             trajectory=np.zeros(10), mask=np.ones(10, bool), contact=np.zeros(16))
    b = copy.deepcopy(a)
    assert duplicate_of(a, b)
    b['trajectory'][0] = .02
    assert not duplicate_of(a, b)
    b = copy.deepcopy(a); b['contact'][0] = .1
    assert not duplicate_of(a, b)
    b = copy.deepcopy(a); b['mask'][0] = False
    assert not duplicate_of(a, b)
    b = copy.deepcopy(a); b['shape'] *= 2
    assert not duplicate_of(a, b)


def test_shape_categories_are_numeric():
    assert geometry_class(np.array([.2, .1, .01])) == 'flat'
    assert geometry_class(np.array([.2, .03, .02])) == 'elongated'
    assert geometry_class(np.array([.1, .08, .07])) == 'compact'
