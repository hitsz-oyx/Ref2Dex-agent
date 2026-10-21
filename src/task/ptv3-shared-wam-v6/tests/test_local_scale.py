import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.local_subset import LOCAL_SCOPE,EXPANDED_SCOPE,index_identity
from ptv3_shared_wam_v6.local_scale import transition_local,data_identity
from ptv3_shared_wam_v5.cache_contract import metadata_identity
from ptv3_shared_wam_v5.training_contract import execution_contract,validate_resume
from test_v6_training import packet


def write(path,value):path.write_text(json.dumps(value))


def scale_packet(packet):
    root,recipe,_,config=packet
    parent=json.loads((root/'parent.json').read_text())
    stats=json.loads((root/'stats.json').read_text())
    events=json.loads((root/'events.json').read_text())
    stats.update(method='checkpoint-normalizer-buffers; inherited TRAIN scales; no moments refit',
                 initialization_checkpoint='/original')
    write(root/'stats.json',stats)
    recipes=[]
    for name,scope,train in [('old',LOCAL_SCOPE,[[0,0,3]]),('new',EXPANDED_SCOPE,[[0,0,3],[2,0,3]])]:
        folder=root/name;folder.mkdir();p=copy.deepcopy(parent);r=copy.deepcopy(recipe);e=copy.deepcopy(events)
        p.update(local_subset_scope=scope,engineering_subset=True,original_frozen_windows_recovered=False,pending_sequences=38349)
        for split,rows in [('train',train),('dev',[[1,0,3]])]:
            np.save(folder/(split+'.npy'),np.array(rows,np.int32));np.save(folder/(split+'-strata.npy'),np.zeros(len(rows),np.int8))
            p['indices'][split]=str(folder/(split+'.npy'));p['strata'][split]=str(folder/(split+'-strata.npy'))
        write(folder/'parent.json',p);e['parent_manifest_identity']=metadata_identity(folder/'parent.json');write(folder/'events.json',e)
        r.update(local_subset_scope=scope,initialization_checkpoint='/original',
            parent_manifest=str(folder/'parent.json'),event_manifest=str(folder/'events.json'),stats=str(root/'stats.json'),
            windows=dict(train=len(train),dev=1),
            index_identities={s:index_identity(p['indices'][s]) for s in ('train','dev')},
            strata_identities={s:index_identity(p['strata'][s]) for s in ('train','dev')})
        for key in ('parent_manifest','event_manifest','stats'):r[key+'_identity']=metadata_identity(r[key])
        write(folder/'manifest.json',r);recipes.append(folder/'manifest.json')
    c=dict(config,local_subset_scope=LOCAL_SCOPE,batch_size=64,updates=2000,max_seconds=14400,artifact_budget_bytes=8*2**30)
    saved=dict(schema=config['schema'],world_size=1,config=c,execution_contract=execution_contract(c,1),
        data_identity=data_identity(recipes[0],root/'stats.json'),step=1269,elapsed_s=1100.,selection={'score':1.},
        model={'normalizer.hand_std':torch.ones(24,3)},optimizer={'state':{'p':{'step':torch.tensor(1269)}}},
        random_state=[{'rank':0}])
    target=dict(c,local_subset_scope=EXPANDED_SCOPE,batch_size=128,updates=10000)
    rng=[dict(python=i,numpy=i,torch=torch.zeros(8,dtype=torch.uint8),cuda=torch.zeros(8,dtype=torch.uint8)) for i in (1,2)]
    return saved,target,recipes,rng


def test_local_scale_retains_learned_state_and_requires_explicit_transition(packet):
    saved,target,(old,new),rng=scale_packet(packet)
    with pytest.raises(ValueError):validate_resume(saved,target,data_identity(new,json.loads(new.read_text())['stats']),3)
    changed=transition_local(saved,target,old,new,3,rng)
    assert changed['model'] is saved['model'] and changed['optimizer'] is saved['optimizer']
    assert changed['step']==1269 and changed['elapsed_s']==1100 and changed['selection']==saved['selection']
    assert changed['execution_contract']['effective_global_batch']==384
    assert changed['random_state'][0] is saved['random_state'][0] and len(changed['random_state'])==3
    assert saved['world_size']==1
    validate_resume(changed,target,changed['data_identity'],3)
    with pytest.raises(ValueError,match='unapproved'):
        transition_local(saved,dict(target,geometry_lr=2e-5),old,new,3,rng)
    with pytest.raises(ValueError,match='RNG'):
        transition_local(saved,target,old,new,3,[])


def test_local_scale_rejects_dev_change_and_train_window_removal(packet):
    saved,target,(old,new),rng=scale_packet(packet)
    r=json.loads(new.read_text());parent=json.loads(Path(r['parent_manifest']).read_text())
    np.save(parent['indices']['train'],np.array([[2,0,3]],np.int32))
    np.save(parent['strata']['train'],np.zeros(1,np.int8))
    r['windows']['train']=1
    r['index_identities']['train']=index_identity(parent['indices']['train']);r['strata_identities']['train']=index_identity(parent['strata']['train']);write(new,r)
    with pytest.raises(ValueError,match='retain every prior'):
        transition_local(saved,target,old,new,3,rng)
    np.save(parent['indices']['dev'],np.array([[1,0,4]],np.int32));r['index_identities']['dev']=index_identity(parent['indices']['dev']);write(new,r)
    with pytest.raises(ValueError,match='DEV panel'):
        transition_local(saved,target,old,new,3,rng)
