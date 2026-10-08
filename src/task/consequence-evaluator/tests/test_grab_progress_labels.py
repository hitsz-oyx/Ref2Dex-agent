"""Labels respect raw timing, support geometry and a shared train-derived scale."""
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]/'src'))

from consequence_evaluator.grab_progress_labels import (
    dense_labels, training_stage_weights, support_planes, plane_in_object_frame,
    lowest_clearance, propose)
from consequence_evaluator.progress_contracts import check_execution_mask


def test_training_weights_mean_sequence_proportions_and_hold_saturates():
    boundaries = [[0,10,20,30],[0,10,20,120]]
    alpha = training_stage_weights(boundaries)
    assert np.allclose(alpha, ([1/3]*3+np.array([1/12,1/12,10/12]))/2)
    stage, progress, valid = dense_labels(160, boundaries[1], 150, alpha)
    assert np.all(progress[120:150] == 1) and np.all(stage[120:150] == 3)
    assert not valid[150:].any() and valid[:150].all()
    with pytest.raises(ValueError):
        dense_labels(160,[0,10,10,120],150,alpha)


def test_support_plane_uses_nonzero_height_and_object_frame():
    vertices = np.array([[x,y,z] for x in [-.5,.5] for y in [-.003,.003] for z in [-.5,.5]])
    rotation = np.array([[1,0,0],[0,0,-1],[0,1,0.]])
    plane = support_planes(vertices, rotation[None], np.array([[0,0,.8]]))
    obj = np.eye(4)[None]; obj[0,:3,3]=[0,0,.9]
    local = plane_in_object_frame(plane,obj)
    assert np.allclose(local,[[0,0,1,.097]])
    assert np.allclose(lowest_clearance(np.array([[0,0,-.02],[0,0,.02]]),local),.077)


def test_boundaries_exclude_other_support_and_returning_tail():
    clearance = np.r_[np.zeros(10),np.linspace(0,.08,20),np.full(25,.08),np.linspace(.08,0,15)]
    contact = np.arange(70)>=10
    wrist = np.column_stack([np.linspace(0,.1,70),np.zeros((70,2))])
    b, reason = propose(clearance,contact,np.zeros(70,bool),np.zeros(70,bool),wrist)
    assert reason is None and b['end'] < 70
    left=np.zeros(70,bool);left[15:20]=True
    assert propose(clearance,contact,left,np.zeros(70,bool),wrist)[1]=='other_hand_or_body_support'


def test_replanning_does_not_supervise_unexecuted_future():
    assert check_execution_mask(8).sum()==8
    assert not check_execution_mask(8)[8:].any()
