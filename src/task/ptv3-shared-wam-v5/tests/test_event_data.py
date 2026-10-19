import json
import pickle
import numpy as np
import pytest
from test_v5_contract import fixture
from ptv3_shared_wam_v5.contracts import DATA_SCHEMA, INPUT_SEMANTICS, LABEL_SEMANTICS
from ptv3_shared_wam_v5.cache_contract import metadata_identity
from ptv3_shared_wam_v5.data import V5Windows
from ptv3_shared_wam_v5.event_data import EventV5Windows
from ptv3_shared_wam_v5.event_labels import EVENT_LABEL_SCHEMA


def dataset(root):
    root.mkdir(); seq, geo, contact = fixture()
    for name, values in [('hand', seq), ('contact', contact)]:
        folder = root/name; folder.mkdir()
        for k, v in values.items(): np.save(folder/(k+'.npy'), v)
        (folder/'meta.json').write_text('{}')
    np.savez(root/'object.npz', **geo)
    (root/'geometry.json').write_text(json.dumps(dict(objects=[dict(key='s1:obj', path='object.npz')])))
    np.save(root/'index.npy', np.array([[0, 0, 3]], np.int32))
    row=dict(key='s1-test', source_id=1, path=str(root/'hand'), contact_path=str(root/'contact'),
             objects=['obj'], contact_meta_identity=metadata_identity(root/'contact/meta.json'))
    parent=dict(schema=DATA_SCHEMA, status='FROZEN', input_semantics=INPUT_SEMANTICS,
                label_semantics=LABEL_SEMANTICS, engineering_subset=True, sequences=[row],
                indices=dict(train=str(root/'index.npy'), dev=str(root/'index.npy')),
                geometry_manifest=str(root/'geometry.json'), source_ids=[1], source_probabilities={'1':1.})
    (root/'manifest.json').write_text(json.dumps(parent))
    side=root/'event'; side.mkdir()
    for k, v in dict(event_type=np.ones((28,1),np.uint8), event_delta_steps=np.ones((28,1),np.int32)*7,
                     event_valid=np.ones((28,1),bool)).items(): np.save(side/(k+'.npy'),v)
    (side/'meta.json').write_text(json.dumps(dict(contact_meta_identity=row['contact_meta_identity'], objects=['obj'],shape=[28,1])))
    events=dict(schema=EVENT_LABEL_SCHEMA, status='COMPLETED', parent_manifest_identity=metadata_identity(root/'manifest.json'),
                sequences=[dict(key=row['key'],path=str(side),event_meta_identity=metadata_identity(side/'meta.json'))])
    (root/'events.json').write_text(json.dumps(events))
    return EventV5Windows(root/'manifest.json','train',root/'events.json',event_dropout=.5)


def test_event_draw_does_not_change_base_sampling_and_dropout_is_reproducible(tmp_path):
    data=dataset(tmp_path/'data')
    base=V5Windows(data.path,'train')
    draw=data.draw(32,177,100,2)
    assert [i for i,_ in draw] == base.draw(32,177,100,2)
    assert draw == data.draw(32,177,100,2)
    assert any(drop for _,drop in draw) and not all(drop for _,drop in draw)
    sample=data[0]; dropped=data[(0,True)]
    assert sample['event_valid'] and sample['event_type']==1 and sample['event_delta_steps']==7
    assert not dropped['event_valid'] and dropped['event_type']==0
    np.testing.assert_array_equal(sample['trajectory_target'],dropped['trajectory_target'])
    restored=pickle.loads(pickle.dumps(data))
    assert not restored.event_cache
    np.testing.assert_array_equal(restored[0]['trajectory_target'],sample['trajectory_target'])
    assert restored[0]['event_delta_steps']==7


def test_event_reader_rejects_changed_annotation_metadata(tmp_path):
    data=dataset(tmp_path/'data')
    (tmp_path/'data/event/meta.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'): data[0]
