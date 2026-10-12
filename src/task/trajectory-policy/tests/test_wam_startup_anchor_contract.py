"""CPU label admission, fixed sampling, and dynamic startup screen contracts."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[4]
SPEC=importlib.util.spec_from_file_location('startup_anchor_fitter',ROOT/'src/task/trajectory-policy/tools/run/fit_wam_startup_anchored_recovery.py')
FITTER=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FITTER)


@pytest.fixture(scope='module')
def authenticated_native():
    path=ROOT/'outputs/trajectory-policy/wam-ref2-execution-geometry-agg-20261012-r1/native'
    records=[json.loads((path/k).read_text()) for k in ('manifest.json','root_recovery_audit.json','result.json')]
    with np.load(path/'trajectory.npz',allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    group=json.loads((path.parent/'group_status.json').read_text())
    return records,data,FITTER.sha(path/'trajectory.npz'),group


def admission(bundle):
    records,data,digest,group=bundle
    return FITTER.own_expert_admission(*records,data,digest,group)


def test_real_complete_native_admission_preserves_failed_group(authenticated_native):
    result=admission(authenticated_native)
    assert result['windows']==161
    assert result['first_tick']==8 and result['last_tick']==168
    assert result['future_causing_control_last']==191 and result['measured_future_state_last']==192
    assert result['source_actor_future_hand_training_allowed'] is False
    assert result['enclosing_collection_status']=='FAILED'
    assert result['actor_expert_future_training_allowed'] is True


@pytest.mark.parametrize('mutation',['group_pass','sha','handoff','terminal_clip','actual_action','current_q','own_history'])
def test_rejects_scope_and_label_corruption(authenticated_native,mutation):
    records,data,digest,group=copy.deepcopy(authenticated_native)
    if mutation=='group_pass':group['status']='COMPLETED'
    if mutation=='sha':digest='0'*64
    if mutation=='handoff':data['expert_applied'][7,1]=True
    if mutation=='terminal_clip':data['expert_clipped'][191,1]=True
    if mutation=='actual_action':data['applied'][191,3,0]+=.01
    if mutation=='current_q':data['q'][8,3,0]+=.01
    if mutation=='own_history':data['learner_hand_history'][8,1,0,0,0]+=.01
    with pytest.raises(ValueError):admission((records,data,digest,group))


def test_fixed32_32_64_draws_reproducible_no_diagnostic_rows():
    ticks=np.repeat(np.arange(519),16);recovery=np.arange(len(ticks)+64,len(ticks)+64+161)
    a=FITTER.anchored_draws(np.random.default_rng(296),ticks,recovery)
    b=FITTER.anchored_draws(np.random.default_rng(296),ticks,recovery)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    nominal,expert=a
    assert len(nominal)==64 and len(expert)==64
    assert (ticks[nominal[:32]]<16).all() and (ticks[nominal[32:]]>=16).all()
    assert np.isin(expert,recovery).all()
    with pytest.raises(ValueError):FITTER.anchored_draws(np.random.default_rng(296),ticks,recovery[:-1])


def test_startup_gate_uses_initial_baseline_not_absolute_bound():
    initial={'nominal_report':dict(first8_epe_mm=20.,xyz_p95_mm=10.,clipping_rate=0.),
             'startup_t0':dict(xyz_p95_mm=1.),'repeat_H4':dict(xyz_p95_mm=2.,wrist_h1_epe_mm=4.)}
    final=copy.deepcopy(initial)
    assert all(FITTER.startup_screen(initial,final).values())
    final['startup_t0']['xyz_p95_mm']=1.501
    final['repeat_H4'].update(xyz_p95_mm=3.001,wrist_h1_epe_mm=5.001)
    checks=FITTER.startup_screen(initial,final)
    assert checks['nominal_PD_retained']
    assert not checks['t0_PD_retained']
    assert not checks['repeat_H4_PD_retained']
    assert not checks['repeat_H4_wrist_retained']
