import copy

import numpy as np
import pytest

from trajectory_policy.wam_multitakeover import balanced_draw, validate_aggregation


def fixture():
    sid, ticks, valid = [], [], []
    admissions = []
    for source, takeover in enumerate((8, 16, 32)):
        ts = list(range(takeover, 105))+list(range(128))
        sid += [source]*len(ts)
        ticks += ts
        valid += [True]*(105-takeover)+[False]*128
        admissions.append(dict(takeover=takeover, pure_future_training_allowed=False,
                               actual_expert_future_training_allowed=True))
    valid = np.asarray(valid)
    data = dict(source_id=np.asarray(sid), tick=np.asarray(ticks), trajectory_valid=valid,
                role=np.where(valid, 3, 2), tau24x33=np.zeros((643, 24, 33), np.float32))
    manifest = dict(status='COMPLETED', training_allowed=True, samples=643,
                    true_expert_future_samples=259, pure_current_PD_samples=384,
                    aggregation_sha256='data', model_identities=[['head', 'R']]*3, admissions=admissions)
    return manifest, dict(pass_=True), data


def admitted():
    m, _, d = fixture()
    return m, {'pass': True, 'aggregation_sha256': 'data'}, d


def test_mixed_dataset_admitted_but_pure_future_denied():
    m, a, d = admitted()
    assert validate_aggregation(m, a, d, ('head', 'R'))['pure_future_training_allowed'] is False
    d['tau24x33'][~d['trajectory_valid'], 0, 0] = 1
    with pytest.raises(ValueError):
        validate_aggregation(m, a, d, ('head', 'R'))


@pytest.mark.parametrize('tamper', ['pair', 'takeover', 'role', 'tick', 'audit'])
def test_cannot_admit_wrong_source_or_scope(tamper):
    m, a, d = admitted()
    if tamper == 'pair':
        m['model_identities'][2] = ['other', 'R']
    elif tamper == 'takeover':
        m['admissions'][1]['takeover'] = 8
    elif tamper == 'role':
        d['role'][0] = 2
    elif tamper == 'tick':
        d['tick'][0] = 0
    else:
        a['pass'] = False
    with pytest.raises(ValueError):
        validate_aggregation(m, a, d, ('head', 'R'))


def test_every_batch_contains_all_actual_takeover_sources():
    pools = [np.arange(i*100, i*100+73) for i in range(3)]
    rng = np.random.default_rng(296)
    extra = set()
    for _ in range(100):
        ids = balanced_draw(rng, pools)
        counts = np.bincount(ids//100, minlength=3)
        assert len(ids) == 16 and sorted(counts.tolist()) == [5, 5, 6]
        extra.add(int(counts.argmax()))
    assert extra == {0, 1, 2}
