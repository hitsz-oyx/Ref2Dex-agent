import json
from pathlib import Path

import numpy as np
import pytest

from shadow_finetune import paths
from shadow_finetune.data import ShadowWindows
from shadow_finetune.scale_controls import SCHEMA, load_controls, stamp, verify_nested, window_key


def caches(root):
    geometry=root/'geometry.npz'
    np.savez(geometry,surface=np.zeros((32,3)),normals=np.tile([0,0,1],(32,1)))
    payloads={}
    rows=[]
    for index,split in enumerate(('train','dev','train')):
        poses=np.tile(np.eye(4),(80,1,1));obs=np.full((80,300),index*.1,np.float32)
        name='sequence%d.npz'%index
        payloads[name]=dict(hand=np.zeros((80,11,3)),poses=poses,goal=np.zeros((80,3)),
                            observation=obs,action=np.full((79,24),index*.1,np.float32),time=np.arange(80)/60)
        rows.append(dict(path=name,geometry=str(geometry),raw_episode=str(root/'raw.npz'),env=index,
                         split=split,object_code='object%d'%index,scale=.08,ticks=[0,1,2]))
    for name,selected in [('small',rows[:2]),('full',[rows[2],rows[1],rows[0]])]:
        out=root/name;out.mkdir()
        for row in selected:np.savez(out/row['path'],**payloads[row['path']])
        (out/'manifest.json').write_text(json.dumps(dict(schema=paths.SCHEMA,status='COMPLETED',records=selected)))
    return root/'small',root/'full'


def test_nested_data_uses_physical_identity_and_rejects_changed_labels(tmp_path):
    small,full=caches(tmp_path)
    a=ShadowWindows(small,'train');b=ShadowWindows(full,'train')
    am=dict(a.meta,_root=str(small));bm=dict(b.meta,_root=str(full))
    verify_nested(am,bm)  # Full record order differs; raw identity must be used.
    path=full/'sequence0.npz'
    with np.load(path) as z:d={k:z[k] for k in z.files}
    d['action'][0,0]=.7;np.savez(path,**d)
    with pytest.raises(ValueError,match='state or label differs'):verify_nested(am,bm)


def test_controls_share_dev_and_small_train_statistics_and_refuse_drift(tmp_path):
    small,full=caches(tmp_path)
    small_train=ShadowWindows(small,'train');dev=ShadowWindows(full,'dev')
    normalizer=small_train.normalization()
    assert not normalizer['state_mean'].any()  # Nonzero DEV/full-only observations excluded.
    parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
    c=dict(schema=SCHEMA,status='PREPARED',training_caches=dict(small=str(small),full=str(full)),
           settings=dict(parent=str(parent),seed=214,updates=1000,batch_size=64),
           validation_cache=str(full),validation_ids=[1],validation_keys=[window_key(dev,1)],
           normalization={k:v.tolist() for k,v in normalizer.items()},
           input_stamps=[dict(path=str(full/'manifest.json'),stamp=stamp(full/'manifest.json'))])
    file=tmp_path/'controls.json';file.write_text(json.dumps(c))
    loaded=[load_controls(file,cache,parent,214,1000,64) for cache in (small,full)]
    for result in loaded:
        assert result[1].split=='dev' and result[1].root==full
        assert window_key(result[1],result[2][0])==c['validation_keys'][0]
        for key in normalizer:np.testing.assert_array_equal(result[3][key],normalizer[key])
    with pytest.raises(ValueError,match='settings changed'):load_controls(file,full,parent,214,1000,128)
    c['validation_keys'][0][-1]=0;file.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='physical identities'):load_controls(file,full,parent,214,1000,64)
    c['validation_keys']=[window_key(dev,1)];file.write_text(json.dumps(c))
    (full/'manifest.json').write_text((full/'manifest.json').read_text()+'\n')
    with pytest.raises(ValueError,match='input changed'):load_controls(file,full,parent,214,1000,64)
