import pytest
import torch

from ptv3_shared_wam.model import SharedWAM
from ptv3_shared_wam.expansion import expand_forecaster
from test_shared_contract import DenseScene, stats, batch


@pytest.mark.parametrize('width', [128,256,768])
def test_expansion_preserves_predictions_and_new_layers_learn(width):
    config=dict(interaction_layers=2)
    source=SharedWAM(stats(),config,DenseScene()).eval()
    target=SharedWAM(stats(),dict(interaction_width=width,interaction_layers=4),DenseScene()).eval()
    expand_forecaster(target,source.state_dict(),config)
    b=batch()
    before,after=source(b),target(b)
    for key in before:
        torch.testing.assert_close(before[key],after[key],rtol=3e-5,atol=3e-7)
    loss=(after['trajectory']-b['trajectory_target']).square().mean()
    loss.backward()
    assert torch.isfinite(target.interaction.layers[2].self_attn.out_proj.weight.grad).all()
    assert target.interaction.layers[2].self_attn.out_proj.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in target.parameters() if p.grad is not None)


def test_expansion_rejects_missing_weights_and_shrinking():
    config=dict(interaction_layers=2)
    source=SharedWAM(stats(),config,DenseScene())
    target=SharedWAM(stats(),dict(interaction_width=256,interaction_layers=4),DenseScene())
    state=dict(source.state_dict());state.pop('goal.target')
    with pytest.raises(ValueError,match='missing'):
        expand_forecaster(target,state,config)
    with pytest.raises(ValueError,match='non-decreasing'):
        expand_forecaster(source,source.state_dict(),dict(interaction_width=128,interaction_layers=4))


def test_interaction_checkpointing_preserves_training_predictions_and_gradients():
    import copy
    source=SharedWAM(stats(),dict(interaction_width=256,interaction_layers=2),DenseScene()).train()
    target=copy.deepcopy(source);target.checkpoint_interaction=True
    b=batch();before,after=source(b),target(b)
    for key in before:
        torch.testing.assert_close(before[key],after[key])
    before['trajectory'].square().sum().backward()
    after['trajectory'].square().sum().backward()
    for left,right in zip(source.parameters(),target.parameters()):
        if left.grad is not None:torch.testing.assert_close(left.grad,right.grad)


def test_expanded_parameter_groups_cover_every_parameter_once_with_distinct_rates():
    target=SharedWAM(stats(),dict(interaction_width=256,interaction_layers=4),DenseScene())
    groups=target.parameter_groups(1e-5,5e-5,interaction_lr=2.5e-5,new_layer_lr=1e-5,expansion_source_layers=2)
    params=[p for g in groups for p in g['params']]
    assert len(params)==len({id(p) for p in params})==len(list(target.parameters()))
    assert {g['name']:g['lr'] for g in groups}==dict(ptv3_scene=1e-5,shared_and_heads=5e-5,
        transferred_interaction=2.5e-5,new_interaction=1e-5)


@pytest.mark.parametrize('points,tokens',[(1,1),(11,4),(19,8),(512,256)])
def test_vectorized_token_sampling_matches_legacy_outputs_and_gradients(points,tokens):
    from ptv3_shared_wam.scene import SceneEncoder,SceneHistoryEncoder
    net=SceneEncoder.__new__(SceneEncoder)
    torch.nn.Module.__init__(net);net.token_count=tokens
    torch.manual_seed(124)
    value=torch.randn(4,points,7,requires_grad=True)
    valid=torch.rand(4,points)>.3;valid[:,0]=True;valid[0]=True;valid[1,1:]=False
    old,om=SceneHistoryEncoder._sample_tokens(net,value,valid)
    new,nm=net._sample_tokens(value,valid)
    torch.testing.assert_close(new,old,rtol=0,atol=0)
    assert torch.equal(om,nm)
    torch.testing.assert_close(torch.autograd.grad(new.sum(),value)[0],torch.autograd.grad(old.sum(),value)[0])
