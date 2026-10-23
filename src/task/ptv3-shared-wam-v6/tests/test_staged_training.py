"""Source/mother balancing and spawn reopen preserve the frozen stage identity."""
import json
from collections import Counter

import numpy as np
import pytest

from ptv3_shared_wam_v6.staged_training import StagedWindows
from ptv3_shared_wam_v6.prepared_reader import sha256


@pytest.fixture
def stage(tmp_path):
    rows=[];indices={s:[] for s in ('train','dev')};phases={s:[] for s in indices}
    for source in (0,1,2,4,5,6,8,9,10):
        for split,n in (('train',1),('train',101),('dev',5)):
            sid=100+len(rows)*7;offset=len(indices[split])
            rows.append(dict(sequence_id=sid,source_id=source,split=split,status='ADMITTED',
                normalized_offset=offset,admitted_windows=n))
            indices[split].extend([[sid,0,i+3] for i in range(n)])
            phases[split].extend([4 if source==8 else (0 if i%2==0 else 3) for i in range(n)])
    for split in indices:
        np.save(tmp_path/(split+'-all.npy'),np.asarray(indices[split],np.int32))
        np.save(tmp_path/(split+'-clock-valid.npy'),np.ones(len(indices[split]),bool))
        np.save(tmp_path/(split+'-phase.npy'),np.asarray(phases[split],np.uint8))
    (tmp_path/'sequences.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    files={f.name:dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in tmp_path.iterdir()}
    meta=dict(schema='ref2dex.nine-source-readiness.v1',expected_mothers=len(rows),files=files,
        sources=[{s:dict(admitted_windows=len(indices[s])) for s in indices}])
    (tmp_path/'manifest.json').write_text(json.dumps(meta));(tmp_path/'geometry.json').write_text('{"objects":[]}')
    return StagedWindows(tmp_path/'manifest.json','train',tmp_path/'geometry.json',sha256(tmp_path/'manifest.json'))


def test_sources_and_mothers_are_balanced_instead_of_long_sequences_dominating(stage):
    mother_counts=Counter();source_counts=Counter()
    for step in range(500):
        selected=stage.draw(16,170,step,0)
        assert selected==stage.draw(16,170,step,0)
        mothers=stage.index[selected,0]
        assert max(Counter(mothers).values())<=8
        assert all(stage.rows[int(s)]['split']=='train' for s in mothers)
        mother_counts.update(map(int,mothers))
        source_counts.update(stage.rows[int(s)]['source_id'] for s in mothers)
    for source,mothers in stage.source_mothers.items():
        assert .4<mother_counts[mothers[0]]/sum(mother_counts[s] for s in mothers)<.6
        assert .08<source_counts[source]/8000<.14


def test_spawn_reopen_preserves_draw_and_rejects_phase_identity_drift(stage):
    state=stage.__getstate__();reopened=StagedWindows(**state)
    assert reopened.draw(16,170,123,2)==stage.draw(16,170,123,2)
    assert reopened.draw(16,170,123,2)!=stage.draw(16,170,123,3)
    np.save(stage.path.parent/'train-phase.npy',np.zeros(len(stage),np.uint8))
    with pytest.raises(ValueError,match='phase index changed'):
        StagedWindows(**state)
