from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ptv3_shared_wam_v3.multimodality import reconstruct_history,neighborhood_metrics,choose_neighbors


def test_history_reconstruction_uses_only_observed_velocity_acceleration_displacement():
    h=np.random.default_rng(17).normal(size=(4,2,11,3))
    f=np.concatenate(((h[3]-h[2])*30,(h[3]-2*h[2]+h[1])*900,h[3]-h[0]),-1)
    np.testing.assert_allclose(reconstruct_history(h[3],f),h,atol=1e-12)


def test_between_mode_mean_vs_real_future_and_unimodal_null():
    rng=np.random.default_rng(17);path=np.linspace(.003,.08,24)[:,None,None]*np.array([1,0,0])[None,None]
    f=np.repeat(path[None],32,axis=0)*np.r_[np.ones(16),-np.ones(16)][:,None,None,None]
    f=np.repeat(f,11,axis=2)+rng.normal(0,.0002,(32,24,11,3))
    m=neighborhood_metrics(f,f.mean(0),f[0]);assert m['multimodal'] and m['between_modes'] and m['off_empirical_support']
    real=neighborhood_metrics(f,f[1],f[0]);assert not real['between_modes']
    null=rng.normal(0,.005,(64,24,11,3));assert not neighborhood_metrics(null,null.mean(0),null[0])['multimodal']


def test_goal_gate_and_capture_deduplication_ignore_future_labels():
    m=dict(source_id=0,side=0,hand_pattern=2047,geometry_key='obj',scene_signature='obj',capture='query')
    mb=[dict(m,capture='same-train') for _ in range(2)]+[dict(m,capture='different')]
    q=dict(history=np.zeros((4,2,11,3)),goal=np.zeros((32,3)),scene_history=np.zeros((8,4,8,3)))
    b={k:np.stack([v,v,v]) for k,v in q.items()};b['goal'][2]=1
    out=choose_neighbors(q,b,m,mb,np.array([1,2,0]),.02)
    assert [x[0] for x in out]==[0]


def test_physical_recording_identity_merges_h2o_gaps_without_merging_unidex_batches():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'tools/audit/analyze_multimodality.py'
    spec=importlib.util.spec_from_file_location('multimodality_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[dict(source_id=sid,source_sequence='same-relative-name',objects=['object'],target_slot=0,group_id='group',path=str(i))
          for i,sid in enumerate((9,9,4,4))]
    meta=[dict(source_id=row['source_id'],row=i,capture='unique-file-'+str(i),geometry_key='object') for i,row in enumerate(rows)]
    module.bind_recordings(meta,dict(sequences=rows))
    assert meta[0]['capture']==meta[1]['capture']
    assert meta[2]['capture']!=meta[3]['capture']
    a=[dict(source_id=9,capture='a',x=0),dict(source_id=9,capture='b',x=1)]
    assert module.capture_bootstrap(a,'x')==module.capture_bootstrap(a+[a[0]]*100,'x')
