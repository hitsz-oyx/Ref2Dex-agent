import json
import numpy as np
import pytest

from ptv3_shared_wam.rl_robot import RLRobotWindows, CORPUS_SCHEMA
from ptv3_shared_wam.data import shared_collate
from ptv3_shared_wam.robot import ACTIVE
from ptv3_shared_wam.loading import StepBatchSampler


def corpus(tmp_path):
    records=[]
    for name,split,shift in [('train','train',0.),('dev','val',100.)]:
        folder=tmp_path/name
        folder.mkdir()
        length=64
        tick=np.arange(length,dtype=np.float32)
        obj=np.tile(np.eye(4,dtype=np.float32),(length,1,1))
        obj[:,0,3]=tick*.001
        hand=np.zeros((length,11,3),np.float32)
        hand[:,:,0]=tick[:,None]*.002
        action=np.repeat((tick[:-1]*.001)[:,None],18,axis=1)
        np.savez(folder/'trajectory.npz',q=np.full((length,18),shift,np.float32),
            dq=np.zeros((length,18),np.float32),hand=hand,obj=obj,reference_obj=obj,
            velocity=np.zeros((length,6),np.float32),action=action,applied=action,
            pd_targets=action)
        np.savez(folder/'canonical.npz',points=np.zeros((512,3),np.float32),
            normals=np.zeros((512,3),np.float32),center=np.zeros(3,np.float32),radius=np.float32(.1))
        records.append(dict(sequence=name,split=split,path=name,frames=length))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema=CORPUS_SCHEMA,status='COMPLETED',
        training_allowed=True,fps=30,num_sequences=2,records=records,
        native_controller=dict(offset=[0.]*18,scale=[1.]*18))))
    return tmp_path


def test_same_tick_commands_future_hand_and_whole_sequence_split(tmp_path):
    root=corpus(tmp_path)
    train,dev=[RLRobotWindows(root,s) for s in ['train','val']]
    sample=train[3]
    np.testing.assert_allclose(sample['native_action'][:,0],np.arange(3,11)*.001,atol=1e-7)
    np.testing.assert_allclose(sample['trajectory_target'][:,0,:,0],
        np.repeat((np.arange(1,25)*.002)[:,None],11,axis=1),atol=1e-7)
    assert sample['trajectory_valid'][:,0].all() and not sample['trajectory_valid'][:,1].any()
    assert sample['native_action'].shape==(8,len(ACTIVE))
    assert len(train)==len(dev)==40
    state,_,_=train.normalization()
    assert np.all(state[:,:18]==0.)
    with pytest.raises(ValueError,match='TRAIN'):
        dev.normalization()
    batch=shared_collate([sample,train[0]])
    assert batch['native_state'].shape==(2,87)


def test_sampler_resume_matches_complete_run(tmp_path):
    dataset=RLRobotWindows(corpus(tmp_path),'train')
    complete=list(StepBatchSampler(dataset,16,124,1,1,5))
    resumed=list(StepBatchSampler(dataset,16,124,1,3,5))
    assert complete[2:]==resumed
    for indices in complete:
        ticks=[dataset.keys[i][0] for i in indices]
        assert all(t<16 for t in ticks[:8]) and all(t>=16 for t in ticks[8:])


def test_actual_command_mismatch_is_rejected(tmp_path):
    root=corpus(tmp_path)
    path=root/'train/trajectory.npz'
    with np.load(path) as f:data={k:f[k].copy() for k in f.files}
    data['applied'][0,0]=.9
    np.savez(path,**data)
    with pytest.raises(ValueError,match='same-execution'):
        RLRobotWindows(root,'train')[0]


def test_tracking_filter_checks_whole_future_and_normalization(tmp_path):
    root=corpus(tmp_path)
    path=root/'train/trajectory.npz'
    with np.load(path) as f:data={k:f[k].copy() for k in f.files}
    # Dropped/failed tail must invalidate windows before the drop as well.
    data['obj'][50:,0,3]+=1.
    data['q'][50:]=999.
    np.savez(path,**data)
    dataset=RLRobotWindows(root,'train')
    assert [tick for tick,_ in dataset.keys]==list(range(26))
    state,_,_=dataset.normalization()
    assert len(state)==26 and (state[:,:18]==0.).all()
