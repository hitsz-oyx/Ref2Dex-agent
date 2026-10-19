"""A policy cannot act on the state that already terminated its episode."""
import json
import numpy as np


def test_task_cache_excludes_actions_from_the_timeout_state_onward(tmp_path, monkeypatch):
    from wam_rl_policy_v2.inspire_task_cache import build_task_fields
    from wam_rl_policy_v2 import wam_v4_runtime
    n=301
    poses=np.repeat(np.eye(4)[None],n,axis=0)
    reference=poses.copy();reference[:,0,3]=np.linspace(0,.1,n)
    source=tmp_path/'source';source.mkdir()
    np.savez(source/'trajectory.npz',obj=poses,hand=np.zeros((n,11,3)),reference_obj=reference)
    cache=tmp_path/'cache';cache.mkdir()
    rng=np.random.default_rng(246);surface=rng.normal(size=(512,3))*.02
    normals=np.zeros_like(surface);normals[:,2]=1
    monkeypatch.setattr(wam_v4_runtime,'load_geometry',lambda *a:[(surface,normals,surface[:32],surface[:128],normals[:128])])
    meta=dict(records=[],sequences=[dict(source=str(source),sequence='stationary',geometry=0,cache_split='train',lo=0,hi=n-1)],
        splits=dict(train=dict(commands=n-1),dev=dict(commands=0)))
    (cache/'manifest.json').write_text(json.dumps(meta))
    for split,size in [('train',n-1),('dev',0)]:
        np.save(cache/f'{split}-scheduled.npy',np.zeros((size,577),np.float32))
        np.save(cache/f'{split}-include.npy',np.ones(size,bool))
    build_task_fields(cache)
    assert (cache/'train-task-active.npy').exists(), 'BC silently admits commands after the online scheduler has terminated'
    active=np.load(cache/'train-task-active.npy')
    assert active.dtype==np.bool_ and active.shape==(300,)
    assert active[:240].all() and not active[240:].any()
    from wam_rl_policy_v2.inspire_shadow import bc_admission
    np.testing.assert_array_equal(bc_admission(cache,'train',False),active)
    np.testing.assert_array_equal(bc_admission(cache,'train',True),active)
    admitted=json.loads((cache/'manifest.json').read_text())['task_admission']
    assert admitted['splits']['train']==dict(source_commands=300,active_commands=240,post_timeout_commands=60)
    workspace=np.ones(300,bool);workspace[100]=False
    np.save(cache/'train-include.npy',workspace)
    assert bc_admission(cache,'train',False)[100]
    assert not bc_admission(cache,'train',True)[100]


def test_old_cache_without_a_task_terminal_mask_is_rejected(tmp_path):
    import pytest
    from wam_rl_policy_v2.inspire_shadow import bc_admission
    (tmp_path/'manifest.json').write_text(json.dumps(dict(task_status='COMPLETED')))
    with pytest.raises(ValueError,match='rebuild task fields in a fresh cache'):
        bc_admission(tmp_path,'train',False)
