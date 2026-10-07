"""Physical-event and joint-supervision regressions; synthetic CPU checks."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/run')]
from consequence_evaluator.data import sha, Windows
from consequence_evaluator.model import matched_loss
from consequence_evaluator.supervision import (
    CONTACT_SEMANTICS, expert_anchor, local_event, local_preferences, physical_trace,
)
from label_continuous import label
from prepare_windows import prepare
from train_matched import require_supervision
from consequence_evaluator.contracts import EPISODE_SCHEMA,ACTION_SEMANTICS


def packet(drop=None, steps=100):
    pose = np.broadcast_to(np.eye(4), (steps+1,4,4)).copy()
    pose[1:,2,3] = .04
    contact = np.r_[False,np.ones(steps,dtype=bool)]
    if drop is not None:
        pose[drop:,2,3] = 0
        contact[drop:] = False
    arrays = dict(history=np.zeros((steps+1,6),dtype='float32'),
                  residual_plan=np.zeros((steps,24,18),dtype='float32'),plan_known=np.ones(steps,dtype=bool),
                  hand_keypoints=np.zeros((steps+1,11,3),dtype='float32'),
                  action=np.zeros((steps,18),dtype='float32'), object_pose=pose,
                  timestamps=np.arange(steps+1)/30, phase=np.asarray(['hold']*steps),
                  progress=np.full(steps+1,np.nan,dtype='float32'),
                  progress_mask=np.zeros(steps+1,dtype=bool))
    diagnostics = dict(contact=contact,contact_valid=np.r_[False,np.ones(steps,dtype=bool)],
                       surface_gap=np.full(steps+1,.001),
                       initial_height=0.)
    return arrays,diagnostics


def test_progress_requires_clean_verified_full_episode_hold_and_no_later_drop():
    arrays, diagnostics = packet()
    trace = physical_trace(arrays,diagnostics)
    record = dict(assigned_phase='clean',perturbation_tick=-1)
    quality,progress,mask,completion = expert_anchor(trace,record)
    assert quality == 'expert_success' and completion == 45 and mask.all()
    assert progress[8] == pytest.approx(8/45) and progress[-1] == 1
    assert expert_anchor(trace,dict(record,assigned_phase='hold',perturbation_tick=8))[2].sum() == 0
    arrays,diagnostics = packet(drop=70)
    quality,progress,mask,_ = expert_anchor(physical_trace(arrays,diagnostics),record)
    assert quality == 'suboptimal' and not mask.any() and np.isnan(progress).all()


def test_local_event_ignores_episode_outcome_and_abstains_from_recovery():
    clean,clean_diag = packet()
    later_drop,drop_diag = packet(drop=70)
    # Both windows are genuinely held, even though the second later fails.
    assert local_event(physical_trace(clean,clean_diag),'hold',8) == 'maintained_hold'
    assert local_event(physical_trace(later_drop,drop_diag),'hold',8) == 'maintained_hold'
    drop,drop_diag = packet(drop=20)
    assert local_event(physical_trace(drop,drop_diag),'hold',8) == 'unrecovered_drop'
    drop['object_pose'][28:,2,3] = .04
    drop_diag['contact'][28:] = True
    assert local_event(physical_trace(drop,drop_diag),'hold',8) is None


def test_reset_contact_or_invalid_post_action_contact_cannot_label_windows():
    arrays,diagnostics = packet()
    assert local_event(physical_trace(arrays,diagnostics),'hold',0) is None
    diagnostics['contact_valid'][10] = False
    with pytest.raises(ValueError,match='diagnostics'):
        physical_trace(arrays,diagnostics)


def test_local_preferences_only_compare_other_episodes_same_split_task_phase_height():
    good = dict(episode='good',tick=8,split='train',task='airplane',phase='hold',
                expert='fixture',motion='fixture',object_pose=np.eye(4),hand_keypoints=np.zeros((11,3)),
                initial_relative_height=.04,event='maintained_hold')
    bad = dict(good,episode='bad',event='unrecovered_drop')
    invalid = [dict(bad,episode='val',split='val'),dict(bad,episode='duck',task='duck'),
               dict(bad,episode='lift',phase='lift'),dict(bad,episode='far',initial_relative_height=.051),
               dict(bad,episode='good')]
    pairs = local_preferences([good,bad,*invalid])
    assert len(pairs) == 1 and pairs[0]['rejected']['episode'] == 'bad'
    many = [dict(good,tick=i) for i in range(20)] + [dict(bad,tick=i) for i in range(20)]
    assert len(local_preferences(many)) == 2


def test_expert_progress_is_trained_even_when_both_preference_endpoints_are_masked():
    def prediction():
        return dict(score=torch.zeros(1,requires_grad=True),
                    progress_logits=torch.zeros(1,24,10,requires_grad=True))
    left,right,expert = prediction(),prediction(),prediction()
    masked = dict(progress=torch.full((1,24),float('nan')),
                  progress_mask=torch.zeros(1,24,dtype=torch.bool))
    reliable = dict(progress=torch.linspace(0,1,24)[None],
                    progress_mask=torch.ones(1,24,dtype=torch.bool))
    loss,terms = matched_loss(left,right,masked,masked,expert,reliable)
    loss.backward()
    assert terms['progress'] > 0 and expert['progress_logits'].grad.abs().sum() > 0
    assert left['progress_logits'].grad.abs().sum() == 0
    assert right['progress_logits'].grad.abs().sum() == 0
    with pytest.raises(ValueError,match='provided together'):
        matched_loss(left,right,masked,masked,expert,None)


def collection_fixture(root,route,anchors=True):
    """Synthetic traces with explicit test provenance, never physical evidence."""
    route.write_text(json.dumps(dict(experts={str(i):dict(sha256=str(i)*64) for i in range(6)})))
    records=[]
    for split in ('train','val','test'):
        for name,drop in [('clean',None),('drop',20)]:
            arrays,diagnostics=packet(drop)
            if name == 'clean' and not anchors:
                assigned,perturbation='hold',8
            else:
                assigned,perturbation=('clean',-1) if name == 'clean' else ('hold',8)
            ep=split+'_'+name
            path=root/(ep+'.npz');sidecar=root/(ep+'-diag.npz')
            np.savez_compressed(path,**arrays);np.savez_compressed(sidecar,**diagnostics)
            records.append(dict(episode=ep,split=split,split_group='seed:'+split,
                quality='unlabeled',task='airplane',expert='0',motion='fixture',assigned_phase=assigned,
                perturbation_tick=perturbation,path=path.name,sha256=sha(path),
                diagnostics=sidecar.name,diagnostics_sha256=sha(sidecar)))
    (root/'manifest.json').write_text(json.dumps(dict(schema=EPISODE_SCHEMA,action_semantics=ACTION_SEMANTICS,
        status='COMPLETED',rollout_kind='continuous',training_allowed=True,fps=30,units='m',
        history_contract='synthetic engineering fixture',contact_semantics=CONTACT_SEMANTICS,
        sources={str(i):str(i)*64 for i in range(6)},episodes=records)))


def test_label_prepare_and_train_selection_preserve_raw_inputs_and_absolute_scale(tmp_path):
    source=tmp_path/'source';source.mkdir();route=tmp_path/'route.json'
    collection_fixture(source,route)
    out=tmp_path/'labels';report=label([source],out,route)
    assert report['status'] == 'READY' and all(report['pairs'].values())
    m=json.loads((out/'manifest.json').read_text())
    for r in m['episodes']:
        with np.load(source/(r['episode']+'.npz')) as raw,np.load(out/r['path']) as labeled:
            assert all(np.array_equal(raw[k],labeled[k]) for k in ('history','action','object_pose','timestamps'))
    data_root=tmp_path/'windows';prepare(out,out/'preferences.json',data_root)
    data=Windows(data_root);ids=require_supervision(data)
    assert (data.arrays['split'][ids] == 'train').all()
    # No selected dense progress anchors depend on whether they appear in a pair.
    assert set(data.arrays['episode'][ids]) == {'train_clean'}
    masks=data.arrays['progress_mask'];targets=data.arrays['progress']
    assert np.all(targets[masks]>=0) and np.all(targets[masks]<=1)


def test_missing_clean_experts_prevent_training_instead_of_inventing_progress(tmp_path):
    source=tmp_path/'source';source.mkdir();route=tmp_path/'route.json'
    collection_fixture(source,route,anchors=False)
    out=tmp_path/'labels';report=label([source],out,route)
    assert report['status'] == 'INSUFFICIENT_EXPERT_PROGRESS'
    assert json.loads((out/'manifest.json').read_text())['training_allowed'] is False
