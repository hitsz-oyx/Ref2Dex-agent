from types import SimpleNamespace

import pytest
import torch

from goal_point_wam.hand_anchored import gather_hand_tokens
from goal_point_wam.model import SceneHistoryEncoder


def observed_scene():
    features=torch.zeros(1,534,18);features[0,512:,14]=1
    features[0,512:,15]=torch.arange(22)//11
    features[0,512:,16]=(torch.arange(22)%11)/10
    packed=torch.arange(534*128,dtype=torch.float32).reshape(1,534,128).requires_grad_()
    batch=dict(features=features,scene_object=torch.cat((torch.zeros(1,512),-torch.ones(1,22)),1),
               point_valid=torch.ones(1,534,dtype=torch.bool),start_hand_valid=torch.ones(1,2,11,dtype=torch.bool))
    return packed,batch


def test_sampling_can_drop_hand_but_full_semantic_readout_survives_permutation():
    packed,batch=observed_scene()
    tokens,_=SceneHistoryEncoder._sample_tokens(SimpleNamespace(token_count=2),packed,batch['point_valid'])
    assert len(tokens[0])==2
    hand,mask=gather_hand_tokens(dict(point_features=packed,scene_tokens=tokens),batch)
    torch.testing.assert_close(hand.reshape(1,22,128),packed[:,512:])
    permutation=torch.randperm(534)
    shuffled={k:(v[:,permutation] if k!='start_hand_valid' else v) for k,v in batch.items()}
    same,_=gather_hand_tokens(dict(point_features=packed[:,permutation]),shuffled)
    torch.testing.assert_close(same,hand)
    hand.sum().backward()
    assert packed.grad[:,:512].count_nonzero()==0 and packed.grad[:,512:].count_nonzero()==22*128
    assert mask.all()


def test_missing_points_are_zero_and_duplicate_semantics_are_rejected():
    packed,batch=observed_scene()
    batch['point_valid'][0,-1]=False;batch['start_hand_valid'][0,1,10]=False
    hand,mask=gather_hand_tokens(dict(point_features=packed),batch)
    assert not mask[0,1,10] and hand[0,1,10].count_nonzero()==0
    batch['features'][0,-2,16]=.8
    with pytest.raises(ValueError,match='duplicate'):gather_hand_tokens(dict(point_features=packed),batch)
