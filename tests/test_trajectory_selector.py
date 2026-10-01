import tempfile
from pathlib import Path
import torch
from src.task.CmResidual.trajectory_selector import FrozenTrajectorySelectors,recommendation_probability
from src.task.CmResidual.contact_trajectory import TrajectoryNetwork,all_trajectories,decode_trajectories,retained_choice
from src.task.CmResidual.contact_ranker import physical_history


def test_native_ignored_channels_do_not_create_distinct_propensity():
    a=torch.zeros(2,6,18);a[:,0,0]=.1;a[:,1,0]=.1;a[:,1,7]=.9
    pool=torch.tensor([[0,1,4,4,0],[0,1,4,4,0]])
    p=recommendation_probability(a,pool,torch.tensor([1,4]))
    assert torch.equal(p,torch.tensor([.6,.4]))
    a[0,1,8]=.2
    assert recommendation_probability(a,pool,torch.tensor([1,4]))[0]==.2


def test_frozen_runtime_matches_calibrated_physical_formula_and_preserves_rng():
    models={v:[TrajectoryNetwork(state_only=v=='state_only').state_dict()] for v in ['cm','state_only','action_shuffled']}
    saved=dict(schema='ref2dex.contact_trajectory.v1',setup_gate={'passed':True},score_key='supported_change_mm',models=models,
               history_mean=torch.zeros(69),history_scale=torch.ones(69),action_mean=torch.zeros(18),action_scale=torch.ones(18),
               context_mean=torch.zeros(22),context_scale=torch.ones(22),height_mean=torch.zeros(10),height_scale=torch.ones(10),
               probability_calibration={v:{k:dict(scale=.5,bias=-.2) for k in ['joint_contact','release']} for v in models},
               calibration={v:dict(margin_mm=.5) for v in models},release_supported=True,best_fixed=1)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'model.pt';torch.save(saved,path);rng=torch.get_rng_state().clone()
        runtime=FrozenTrajectorySelectors(path,'cpu')
        assert torch.equal(rng,torch.get_rng_state())
        assert all(not p.requires_grad for m in runtime.models for p in m.parameters())
        h=torch.randn(3,10,69);a=torch.randn(3,6,18);rest=torch.ones(3);motion=torch.arange(3);start=torch.zeros(3);trigger=torch.ones(3)*10
        actual=runtime.predict(h,a,rest,motion,start,trigger)
        context=torch.cat((a[:,4],torch.eye(3),((start+trigger)/600)[:,None]),-1)
        for index,v in enumerate(models):
            raw=all_trajectories(runtime.pool[v][0],physical_history(h,rest),a,context)[None]
            decoded=decode_trajectories(raw,saved['height_mean'],saved['height_scale'],((h[:,-1,38]-rest)*1000)[None,:,None],saved['probability_calibration'][v])
            expected=retained_choice(decoded,.5,True,score_key='supported_change_mm')[0]
            assert torch.equal(actual['policy_proposals'][:,index],expected)
        assert (actual['policy_proposals'][:,3]==4).all()
        assert (actual['policy_proposals'][:,4]==1).all()
