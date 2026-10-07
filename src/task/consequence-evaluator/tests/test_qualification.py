"""The first-episode gate must detect drops and exclude later resets."""
from pathlib import Path
import sys

import numpy as np
import pytest

TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.qualification import qualify_transitions


def native_trace(count=64,steps=80):
    before=np.zeros((steps,count,13),dtype='float32');before[:,:,6]=1
    before[1:,:,2]=.04
    after=before.copy();after[:,:,2]=.04
    contacts=np.ones((steps,count,1),dtype=bool)
    done=np.zeros_like(contacts);done[59]=True;done[-1]=True
    # Later episodes may drop, but the first qualified episode already finished.
    after[60:,:,2]=0;before[61:,:,2]=0
    progress=np.broadcast_to(np.arange(steps)[:,None,None],(steps,count,1)).copy()
    progress[60:]-=60
    result=dict(object_state=before,next_object_state=after,hand_contact=contacts,
                object_contact=contacts.copy(),done=done,progress=progress,
                data_id=np.zeros_like(progress),action=np.zeros((steps,count,18),dtype='float32'))
    flat={k:v.reshape(steps*count,*v.shape[2:]) for k,v in result.items()}
    episodes=[dict(env_id=i,start_frame=0,motion_id=0,steps=60) for i in range(count)]
    return flat,episodes


def test_gate_recomputes45_frame_holds_and_excludes_later_episodes():
    arrays,episodes=native_trace();r=qualify_transitions(arrays,episodes)
    assert r['data_readiness_pass'] and r['qualified_episodes']==64
    assert all(x['completion_tick']==45 and x['maximum_held_frames']==60 for x in r['per_episode'])
    assert r['training_allowed'] is False


def test_drop_after_qualification_cannot_count_as_success():
    arrays,episodes=native_trace()
    # Contact lost for six frames after45-frame hold, even while elevated.
    contact=arrays['object_contact'].reshape(80,64,1);contact[49:59,:60]=False
    r=qualify_transitions(arrays,episodes)
    assert r['qualified_episodes']==4 and not r['data_readiness_pass']
    assert all(x['dropped_after_hold'] for x in r['per_episode'][:60])


@pytest.mark.parametrize('kind',['clock','pose','missing_done','start','teleport'])
def test_gate_rejects_broken_native_first_episode_contract(kind):
    arrays,episodes=native_trace()
    if kind=='clock':arrays['progress'][64,0]=5
    if kind=='pose':arrays['object_state'][64,0]+=1
    if kind=='missing_done':arrays['done'][::64]=False
    if kind=='start':episodes[0]['start_frame']=1
    if kind=='teleport':arrays['object_state'][0,0]+=1.5
    with pytest.raises(ValueError):qualify_transitions(arrays,episodes)
