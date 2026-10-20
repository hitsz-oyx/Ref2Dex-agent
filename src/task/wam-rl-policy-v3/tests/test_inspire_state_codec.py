import json
import numpy as np
import pytest
from wam_rl_policy_v3.inspire_state_codec import pack_state,unpack_state,FIELDS


def fixture():
    rng=np.random.default_rng(23)
    state={k:rng.normal(size=(8,n)).astype(np.float32) for k,n in zip(FIELDS[:-1],(18,18,248,13,33))}
    state['hand']=state['hand'].reshape(8,11,3);state['tick']=np.arange(8,dtype=np.int64)
    state.update(episodes=list(range(8)),sequences=['motion']*8)
    return state


def test_json_wire_roundtrip_preserves_all_numeric_bits_and_metadata():
    state=fixture();result=unpack_state(json.loads(json.dumps(pack_state(state),allow_nan=False)))
    for name in FIELDS:
        assert result[name].dtype==state[name].dtype
        assert result[name].tobytes()==state[name].tobytes()
    assert result['episodes']==state['episodes'] and result['sequences']==state['sequences']
    legacy={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in state.items()}
    assert unpack_state(legacy) is legacy


@pytest.mark.parametrize('fault',['dtype','shape','bytes','limit','nan','base64'])
def test_invalid_numeric_packet_is_rejected(fault):
    packet=pack_state(fixture())
    if fault=='dtype':packet['q']['dtype']='object'
    elif fault=='shape':packet['q']['shape']=[-1,18]
    elif fault=='bytes':packet['q']['shape']=[8,19]
    elif fault=='limit':packet['q']['shape']=[2**30,18]
    elif fault=='base64':packet['q']['data']='!invalid!'
    else:
        state=fixture();state['q'][0,0]=np.nan
        with pytest.raises(ValueError):pack_state(state)
        return
    with pytest.raises(ValueError):unpack_state(packet)


def test_real_adapter_observation_reward_and_done_match_json_lists(tmp_path):
    import torch
    from wam_rl_policy_v3.inspire_ppo_bridge import RemoteInspireEnv
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    ref=np.broadcast_to(np.eye(4),(8,4,4)).copy();ref[:,2,3]=np.arange(8)*.001
    np.savez(tmp_path/'trajectory.npz',obj=ref,reference_obj=np.full_like(ref,np.nan))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(sequences=[dict(split='train',sequence='motion',object='fixture',source=str(tmp_path))])))
    rng=np.random.default_rng(19);surface=rng.normal(size=(512,3)).astype(np.float32)*.02
    normals=np.zeros_like(surface);normals[:,2]=1
    class Runtime:
        device=torch.device('cpu');config=dict(voxel_origin_m=[-2]*3,voxel_m=.01)
        geometry=[(surface,normals,surface[:32],surface[:128],normals[:128])]*2
        def predict(self,batch):return dict(trajectory=torch.full((len(batch['xyz']),24,2,11,3),.001))
    initial={k:rng.normal(size=(2,n)).astype(np.float32)*.01 for k,n in zip(FIELDS[:-1],(18,18,248,13,33))}
    initial['hand']=initial['hand'].reshape(2,11,3)
    initial['objects'][:]=0;initial['objects'][:,6]=1
    initial.update(tick=np.zeros(2,np.int64),episodes=[0,0],sequences=['motion']*2)
    def wire(state,packed):
        state=pack_state(state) if packed else {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in state.items()}
        return json.loads(json.dumps(state))
    class FixtureEnv(RemoteInspireEnv):
        def rpc(self,op,**values):
            assert op=='step'
            self.control+=1;state={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in initial.items()}
            state['tick'][:]=self.control;state['objects'][:,2]=self.control*.001
            return dict(state=unpack_state(wire(state,self.packed)))
    envs=[]
    for packed in (False,True):
        env=FixtureEnv(None,dict(state=wire(initial,packed),audit={}),tmp_path,Runtime())
        env.control=0;env.packed=packed;envs.append(env)
    for _ in range(5):
        left=envs[0].step(torch.zeros(2,12));right=envs[1].step(torch.zeros(2,12))
        for a,b in zip(left[:3],right[:3]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert left[3]['execution_episode']==right[3]['execution_episode']
