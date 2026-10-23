"""An actual full manifest must keep held-out representatives and honest provenance."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from test_rigid_dataset import bundle

import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/audit'))
summary=__import__('summarize_full_rigid_data')


@pytest.mark.parametrize('workers',[1,2])
@pytest.mark.parametrize('nontransitive',[False,True])
def test_complete_manifest_cross_split_dedup_and_unknown_certification(tmp_path,monkeypatch,nontransitive,workers):
    prep=summary.prep
    run=tmp_path/'run';run.mkdir();(run/'receipts').mkdir()
    registry=tmp_path/'registry';registry.mkdir();(registry/'geometry').mkdir()
    seq,geo,contact,event,evidence=bundle()
    geo['surface']=np.random.default_rng(170).uniform(-.1,.1,(2048,3))
    np.savez(registry/'geometry/g.npz',**geo)
    prep.write(registry/'manifest.json',dict(objects=[dict(key='s1:example',path='geometry/g.npz')]))
    prep._GEOS.clear();monkeypatch.setattr(prep,'REGISTRY',registry)
    # Real filesystem bytes, not a mirrored implementation of dedup.
    for sid,split in [(0,'train'),(1,'dev')]:
        folder=run/('hand'+str(sid));folder.mkdir();cf=run/('contact'+str(sid));cf.mkdir()
        for k,v in seq.items():np.save(folder/(k+'.npy'),v)
        for k,v in contact.items():np.save(cf/(k+'.npy'),v)
        tuples=np.array([[sid,0,t] for t in range(3,8)],np.int32)
        shard=run/('windows'+str(sid)+'.npz')
        np.savez(shard,tuples=tuples,phase=np.full(5,3,np.uint8),geometry_class=np.full(5,2,np.uint8))
        prep.write(run/'receipts'/('s1-'+str(sid)+'.json'),dict(sequence_id=sid,key='s1-'+str(sid),source_id=1,
            source_sequence='different_mother_'+str(sid),split=split,status='ADMITTED',admitted_windows=5,
            reasons=[],conversion_started=False,path=str(folder),contact_path=str(cf),objects=['example'],
            window_shard=str(shard),label_object_pose_sha256=prep.digest(folder/'object_poses.npy'),label_filesystem_mtime_ns=123))
    if nontransitive:
        train=np.load(run/'hand0/joints.npy');train[:]=.0009;train[28]=.0025;np.save(run/'hand0/joints.npy',train)
        dev=np.load(run/'hand1/joints.npy');dev[:]=-.0009;np.save(run/'hand1/joints.npy',dev)
    for i in [0,1]:np.savez(run/('clock'+str(i)+'.npz'),framerate=np.float64(120),n_frames=np.int64(128))
    fixture_plan=tmp_path/'plan.json'
    prep.write(fixture_plan,dict(jobs=[dict(key='s1-'+str(i),source_id=1,source_sequence='different_mother_'+str(i),sequence_folder=str(run/('hand'+str(i))),raw_path=str(run/('clock'+str(i)+'.npz')),windows={('train' if i==0 else 'dev'):'fixture.npy'}) for i in [0,1]]))
    monkeypatch.setattr(prep,'PLAN',fixture_plan)
    prep.write(run/'progress.json',dict(status='CONVERSION_AND_ADMISSION_COMPLETED'))
    prep.write(run/'inventory.json',dict(total_sequences=2,by_status=dict(REUSE=2),seeds=dict(full_manifest=170)))
    pilot=tmp_path/'pilot.json'
    prep.write(pilot,dict(selection=dict(train_window_rows=[dict(sequence_key='s1-0',slot=0,tick=t,object='example',source_id=1,phase='hold') for t in range(3,8)]),pending_sequence_batch=dict(selected_keys=[]),audit_rows=[dict(sequence_key='s1-1',slot=0,tick=t,object='example',source_id=1,phase='hold',split='dev',admitted=True) for t in range(3,8)]))
    monkeypatch.setattr(prep,'PILOT',pilot)
    monkeypatch.setattr(summary,'legacy_census',lambda *_:dict(unique_mothers=0))
    out=run/'summary';summary.summarize(run,out,workers)
    result=prep.read(out/'manifest.json')
    assert result['windows_before']==dict(train=5,dev=5)
    assert result['windows_after']==dict(train=1 if nontransitive else 0,dev=1)
    assert result['dedup']['cross_split_train_removed']==(1 if nontransitive else 5)
    assert result['versus_6000']['dev_baseline_windows']==5
    assert result['versus_6000']['dev_baseline_rows_present_before_dedup']==5
    assert result['versus_6000']['dev_baseline_rows_present_after_dedup']==1
    assert result['versus_6000']['dev_added_before_dedup']==0
    assert result['versus_6000']['dev_added_after_dedup']==0
    assert result['provenance']['uncertifiable']==10
    assert result['provenance']['certified_legal']==0
    rows=[json.loads(x) for x in (out/'train-windows.jsonl').read_text().splitlines()]
    assert all(r['source_role']=='UNKNOWN' and r['available_before_reset'] is None for r in rows)
    assert (out/'manifest.sha256').read_text().split()[0]==prep.digest(out/'manifest.json')
    assert np.load(out/'train-deduplicated.npy').shape==(1 if nontransitive else 0,3)
    summary.sequence.cache_clear();prep._GEOS.clear()


@pytest.mark.parametrize('recorded',['train',['train']])
def test_historical_split_representation_keeps_frozen_membership(recorded):
    assert summary.bound_split(recorded,dict(windows={'train':'fixed.npy'}))=='train'


@pytest.mark.parametrize('recorded',[['train','dev'],['dev'],'dev',[],None])
def test_split_normalization_rejects_membership_drift(recorded):
    with pytest.raises(ValueError,match='split disagreement'):
        summary.bound_split(recorded,dict(windows={'train':'fixed.npy'}))


def test_timing_gate_rejects_a_boundary_inside_the_full_horizon(tmp_path):
    clock=tmp_path/'clock.npz'
    t=np.arange(32,dtype=np.float64)/30
    segment=np.zeros(32,np.int32);segment[20:]=1
    np.savez(clock,time=t,segment=segment)
    record=dict(key='example',clock_kind='packed_grid30',clock_uri=str(clock))
    with pytest.raises(ValueError,match='clock/segment boundary'):
        summary.timing_gate(record,np.array([[0,0,3]],np.int32),32)
