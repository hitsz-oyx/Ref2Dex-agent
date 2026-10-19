import copy
import hashlib
import importlib.util
import json
import pickle
import random
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from test_v5_contract import Geometry,config,stats,fixture
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.contracts import SCHEMA,DATA_SCHEMA
from ptv3_shared_wam_v6.events import EVENT_ENCODING
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.data import V6Windows,collate
from ptv3_shared_wam_v6.recipe import validate_recipe
import ptv3_shared_wam_v6.migration as migration
from ptv3_shared_wam_v5.model import SharedWAMV5
from ptv3_shared_wam_v5.data import V5Windows
from ptv3_shared_wam_v5.contracts import DATA_SCHEMA as V5_DATA,INPUT_SEMANTICS,LABEL_SEMANTICS,SAMPLING_SEMANTICS
from ptv3_shared_wam_v5.event_labels import EVENT_LABEL_SCHEMA
from ptv3_shared_wam_v5.cache_contract import metadata_identity
from ptv3_shared_wam_v5.training_contract import execution_contract


def write(path,value):path.write_text(json.dumps(value))


@pytest.fixture
def packet(tmp_path,monkeypatch):
    seq,geo,contact=fixture();source_ids=[0,1,2,4,5,6,8,9,10]
    for name,values in [('hand',seq),('contact',contact)]:
        folder=tmp_path/name;folder.mkdir()
        for k,v in values.items():np.save(folder/(k+'.npy'),v)
        write(folder/'meta.json',{})
    np.savez(tmp_path/'object.npz',**geo)
    write(tmp_path/'geometry.json',dict(objects=[dict(key=k,path='object.npz') for k in ['s0:obj','s1:obj','s2:obj','obj']]))
    for key in ('hands','contacts'):write(tmp_path/(key+'.json'),dict(identity=key))
    rows=[dict(key='s%d-test'%sid,source_id=sid,path=str(tmp_path/'hand'),contact_path=str(tmp_path/'contact'),
               objects=['obj'],hand_meta_identity=metadata_identity(tmp_path/'hand/meta.json'),
               contact_meta_identity=metadata_identity(tmp_path/'contact/meta.json')) for sid in source_ids]
    np.save(tmp_path/'index.npy',np.array([[i,0,3] for i in range(9)],np.int32));np.save(tmp_path/'strata.npy',np.zeros(9,np.int8))
    parent=dict(schema=V5_DATA,status='FROZEN',training_allowed=True,engineering_subset=False,pending_sequences=0,
        input_semantics=INPUT_SEMANTICS,label_semantics=LABEL_SEMANTICS,sequences=rows,
        indices={s:str(tmp_path/'index.npy') for s in ('train','dev')},geometry_manifest=str(tmp_path/'geometry.json'),
        source_ids=source_ids,source_probabilities={str(s):1./9 for s in source_ids},sampling_semantics=SAMPLING_SEMANTICS,
        strata={s:str(tmp_path/'strata.npy') for s in ('train','dev')},
        cache_bindings=dict(hands=str(tmp_path/'hands.json'),hand_identity='hands',contacts=str(tmp_path/'contacts.json'),
                            contact_identity='contacts',geometry_identity=metadata_identity(tmp_path/'geometry.json')))
    write(tmp_path/'parent.json',parent);write(tmp_path/'stats.json',stats())
    event_rows=[]
    for i,row in enumerate(rows):
        folder=tmp_path/row['key'];folder.mkdir()
        for k,v in dict(event_type=np.full((28,1),i%3,np.uint8),
                        event_delta_steps=np.full((28,1),7 if i%3 else 0,np.int32),event_valid=np.ones((28,1),bool)).items():
            np.save(folder/(k+'.npy'),v)
        write(folder/'meta.json',dict(contact_meta_identity=row['contact_meta_identity'],objects=['obj'],shape=[28,1]))
        event_rows.append(dict(key=row['key'],path=str(folder),event_meta_identity=metadata_identity(folder/'meta.json')))
    events=dict(schema=EVENT_LABEL_SCHEMA,status='COMPLETED',parent_manifest_identity=metadata_identity(tmp_path/'parent.json'),sequences=event_rows)
    write(tmp_path/'events.json',events)
    recipe=dict(schema=DATA_SCHEMA,status='FROZEN',training_allowed=True,
        parent_manifest=str(tmp_path/'parent.json'),event_manifest=str(tmp_path/'events.json'),stats=str(tmp_path/'stats.json'),
        parent_manifest_identity=metadata_identity(tmp_path/'parent.json'),event_manifest_identity=metadata_identity(tmp_path/'events.json'),
        stats_identity=metadata_identity(tmp_path/'stats.json'),event_encoding=EVENT_ENCODING,event_dropout=.15,
        source_ids=source_ids,windows=dict(train=9,dev=9))
    write(tmp_path/'manifest.json',recipe)
    c=dict(config(),seed=170,geometry_lr=.001,shared_lr=.001,weight_decay=.01,clip_grad=1.,
        updates=4,batch_size=1,max_seconds=120,artifact_budget_bytes=64*2**20,
        validation_samples=1,validation_interval=2,validation_initial_step=20,validation_microbatch_size=1,
        checkpoint_interval=2,checkpoint_keep_recent=2,checkpoint_milestone_interval=0,save_first_step=False,
        log_interval=1,warmup_updates=0,amp=False)
    v6=dict(c,schema=SCHEMA,event_conditioning=True,event_encoding=EVENT_ENCODING,event_dropout=.15)
    write(tmp_path/'config.json',v6)
    monkeypatch.setattr(migration,'SharedWAMV5',lambda s,c:SharedWAMV5(s,c,Geometry(8)))
    return tmp_path,recipe,c,v6


def original_checkpoint(c,root):
    base=SharedWAMV5(stats(),c,Geometry(8))
    opt=torch.optim.AdamW(base.parameter_groups(c['geometry_lr'],c['shared_lr']),weight_decay=c['weight_decay'])
    for p in base.parameters():p.grad=torch.ones_like(p)*.03
    opt.step()
    saved=dict(schema=c['schema'],config=c,step=1,world_size=1,execution_contract=execution_contract(c,1),
        data_identity=hashlib.sha256((root/'parent.json').read_bytes()+(root/'stats.json').read_bytes()).hexdigest(),
        model=base.state_dict(),optimizer=opt.state_dict(),elapsed_s=1.,selection={},
        random_state=[dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=None)])
    return base,saved


def test_v6_recipe_reader_dropout_and_spawn_preserve_parent_inputs(packet):
    root,_,_,_=packet;data=V6Windows(root/'manifest.json','train');old=V5Windows(root/'parent.json','train')
    assert [i for i,_ in data.draw(32,170,3,0)]==old.draw(32,170,3,0)
    for i in range(9):
        a,b=old[i],data[i]
        assert all(np.array_equal(v,b[k]) for k,v in a.items())
    restored=pickle.loads(pickle.dumps(data));assert not restored.event_cache
    loader=DataLoader(restored,batch_sampler=[data.draw(2,170,3,0)],num_workers=1,multiprocessing_context='spawn',timeout=30,collate_fn=collate)
    batch=next(iter(loader));assert batch['event_valid'].shape==(2,)
    assert batch['trajectory_target'].shape==(2,24,2,21,3)


def test_expansion_preserves_complete_state_and_rejects_hidden_config_changes(packet):
    root,recipe,c,new=packet;base,saved=original_checkpoint(c,root)
    target=SharedWAMV6(stats(),new,Geometry(8))
    migrated=migration.migrate_checkpoint(saved,target,root/'manifest.json',root/'stats.json',recipe,new,1)
    assert migrated['schema']==SCHEMA and migrated['step']==1
    assert migrated['migration']['optimizer']['preserved_states']==len(saved['optimizer']['state'])
    assert migrated['random_state'] is saved['random_state'] and migrated['elapsed_s']==1.
    base.eval();target.eval();data=V6Windows(root/'manifest.json','dev');b=collate([data[1]])
    with torch.no_grad():before=base(b);after=target(b)
    for k in ('trajectory','scene','interaction'):torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
    for key,value in [('batch_size',2),('geometry_lr',.01),('horizon',12),('event_dropout',.2)]:
        with pytest.raises(ValueError,match='only schema'):
            migration.validate_expansion_config(c,dict(new,**{key:value}))
    with pytest.raises(ValueError,match='v6 requires'):SharedWAMV6(stats(),dict(new,event_conditioning=False),Geometry(8))


def test_v6_rejects_upstream_and_stats_drift(packet):
    root,_,_,_=packet
    altered=stats();altered['hand']['std'][0][0]=2.
    write(root/'other-stats.json',altered)
    with pytest.raises(ValueError,match='CLI stats'):validate_recipe(root/'manifest.json',root/'other-stats.json')
    write(root/'events.json',{})
    with pytest.raises(ValueError,match='upstream changed'):V6Windows(root/'manifest.json','train')


def test_actual_v6_entrypoint_resume_matches_uninterrupted_and_rejects_v5(packet,monkeypatch):
    root,recipe,c,new=packet
    task=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('v6_training',task/'tools/run/train_pretrain.py')
    training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
    monkeypatch.setattr(training,'SharedWAMV6',lambda s,c:SharedWAMV6(s,c,Geometry(8)))
    for key in ('WORLD_SIZE','RANK','LOCAL_RANK'):monkeypatch.delenv(key,raising=False)
    _,saved=original_checkpoint(c,root);target=SharedWAMV6(stats(),new,Geometry(8))
    migrated=migration.migrate_checkpoint(saved,target,root/'manifest.json',root/'stats.json',recipe,new,1)
    initial=root/'initial.pt';torch.save(migrated,initial)
    def run(name,resume,extra=()):
        output=root/name
        monkeypatch.setattr(sys,'argv',['train','--config',str(root/'config.json'),'--manifest',str(root/'manifest.json'),
            '--stats',str(root/'stats.json'),'--output',str(output),'--device','cpu','--workers','0',
            '--git-commit','test','--resume',str(resume)]+list(extra))
        training.main();return output
    full=run('full',initial);first=run('first',initial,['--recovery-probe-updates','1'])
    resumed=run('resumed',first/'step000002.pt')
    a=torch.load(full/'step000004.pt',weights_only=False);b=torch.load(resumed/'step000004.pt',weights_only=False)
    for key,value in a['model'].items():torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)
    assert b['step']==4 and len(b['optimizer']['state'])==len(a['optimizer']['state'])
    assert b['migration_binding']['original_weights_exact']
    assert b['model']['event_goal.encoder.2.weight'].abs().sum()>0
    dev=json.loads((full/'dev-step000004.json').read_text());assert 'unknown_event_control' in dev
    torch.save(saved,root/'v5.pt')
    with pytest.raises(ValueError,match='independent v6'):run('wrong-v5',root/'v5.pt')
    with pytest.raises(ValueError,match='mismatch'):run('wrong-batch',initial,['--batch-size','2'])
