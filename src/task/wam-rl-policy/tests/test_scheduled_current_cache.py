import numpy as np
import torch
from wam_rl_policy.scheduled_current_cache import RecordInputs
from wam_rl_policy.wam_v4_runtime import v4_online_samples,collate_v4


def test_worker_preparation_matches_main_process_inputs(tmp_path):
    rng=np.random.RandomState(7)
    poses=np.repeat(np.eye(4)[None],12,0)
    hand=np.zeros((12,11,3));hand[:,:,2]=.03
    obs=np.zeros((12,300));actions=np.zeros((11,24))
    goal=np.tile([0,0,.1],(12,1))
    np.savez(tmp_path/'sequence.npz',poses=poses,hand=hand,time=np.arange(12)/60,
        observation=obs,action=actions,goal=goal)
    surface=rng.randn(512,3)*.01;normals=surface/np.linalg.norm(surface,axis=1,keepdims=True)
    geometry=(surface,normals,surface[:32],surface[:128],normals[:128])
    records=[dict(source=dict(path='sequence.npz',ticks=list(range(9))),lo=0,hi=9,geometry=0)]
    old=RecordInputs(tmp_path,records,[geometry])[0]
    new=RecordInputs(tmp_path,records,[geometry],prepare_batch=2)[0]
    assert len(new['batches'])==2
    for i,lo in enumerate(range(0,3,2)):
        hi=min(lo+2,3)
        expected=collate_v4(v4_online_samples(old['objects'][lo:hi],old['hands'][lo:hi],
            [geometry]*(hi-lo),goal_delta=old['goals'][lo:hi]))
        for key,value in expected.items():torch.testing.assert_close(new['batches'][i][key],value,atol=0,rtol=0)


def test_parallel_completion_only_accepts_owned_records():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'tools/run/parallel_scheduled_cache.py'
    spec=importlib.util.spec_from_file_location('parallel_cache',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=np.zeros(11,bool);base[0]=True
    shards=[np.zeros(11,bool) for _ in range(3)]
    shards[0][3]=True;shards[1][4]=True;shards[2][5]=True
    shards[0][7]=True  # Wrong-owner marker must not publish this row.
    merged=module.merge_completion(base,shards)
    assert np.flatnonzero(merged).tolist()==[0,3,4,5]
    for rank,flags in enumerate(shards):flags[rank::3]=True
    assert module.merge_completion(base,shards).all()
