"""CPU-only R override identity and normalization contracts."""
import copy
import json

import pytest
import torch

from trajectory_policy import wam_retargeter_admission as admission


def packet():
    values={k:torch.ones(2) for k in admission.NORMALIZATION}
    values['net.0.weight']=torch.ones(2,2)
    return dict(schema='ref2dex.competent-pd-retargeter.v1',width=256,use_tau=True,
        model=values,base_R_sha256=admission.BASE_R_SHA)


def test_packet_retains_six_buffers_but_allows_net_update():
    base=packet();new=copy.deepcopy(base)
    new['model']['net.0.weight'].add_(.1)
    admission.validate_r_packet(new,base)


@pytest.mark.parametrize('defect',['norm','signed_zero','key','shape','dtype','nonfinite','schema','width','use_tau','base'])
def test_packet_rejects_architecture_and_normalization_drift(defect):
    base=packet();new=copy.deepcopy(base)
    if defect=='norm':new['model']['tau_scale'][0]+=1e-7
    elif defect=='signed_zero':
        base['model']['state_mean'].zero_();new['model']['state_mean'].fill_(-0.)
    elif defect=='key':new['model']['extra']=torch.ones(2)
    elif defect=='shape':new['model']['net.0.weight']=torch.ones(1)
    elif defect=='dtype':new['model']['net.0.weight']=torch.ones(2,2).double()
    elif defect=='nonfinite':new['model']['net.0.weight'][0,0]=float('nan')
    elif defect=='schema':new['schema']='other'
    elif defect=='width':new['width']=64
    elif defect=='use_tau':new['use_tau']=False
    else:new['base_R_sha256']='other'
    with pytest.raises(ValueError):admission.validate_r_packet(new,base)


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(admission,'ROOT',tmp_path)
    owner=tmp_path/'outputs/trajectory-policy';owner.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    native=owner/'native';native.mkdir()
    source=native/'trajectory.npz';source.write_bytes(b'bounded measured source')
    source_sha=admission.sha(source)
    write(native/'manifest.json',dict(status='COMPLETED',controls=192,takeover_tick=32,
        actor_future_hand_training_allowed=False,mixed_policy_future_hand_training_allowed=False,
        trajectory_sha256=source_sha))
    write(native/'root_recovery_audit.json',{'pass':True,'native_data_sha256':source_sha,
        'controls':192,'takeover_tick':32,'checks_max':0,'check_count':3468})
    write(native/'result.json',dict(local_grasp_calibration=True,local_grasp_recovery=True))
    nominal=owner/'nominal';nominal.mkdir()
    (nominal/'trajectory.npz').write_bytes(b'qualified nominal measured source')
    write(nominal/'audit.json',{'pass':True})
    base_dir=owner/'competent-pd-fit-20261011-r1';base_dir.mkdir()
    base=packet();base['input_sha256']={str(p):admission.sha(p) for p in
        (nominal/'audit.json',nominal/'trajectory.npz')}
    torch.save(base,base_dir/'tau.pt')
    base_sha=admission.sha(base_dir/'tau.pt');monkeypatch.setattr(admission,'BASE_R_SHA',base_sha)
    physical=owner/'competent-pd-execution-20261011-r1';physical.mkdir()
    write(physical/'manifest.json',dict(input_sha256={str(base_dir/'tau.pt'):base_sha}))
    write(physical/'result.json',dict(teacher_readiness=True,summary={'gt_tau_R':dict(episodes=8,success=8,hold45=8)}))
    directory=owner/'adapted';directory.mkdir()
    scoped=dict(schema='ref2dex.unmixed-recovery-R-admission.v1',admitted=True,R_training_allowed=True,
        actor_future_hand_training_allowed=False,mixed_policy_future_hand_training_allowed=False,
        role=3,first_tick=32,last_tick=168,horizon=24,windows=137,source_trajectory_sha256=source_sha,
        nominal_audit_bound_by_base_R_sha256=base_sha)
    paths=dict(native_source=source,native_manifest=native/'manifest.json',
        root_audit=native/'root_recovery_audit.json',native_result=native/'result.json',
        nominal_audit=nominal/'audit.json',nominal_trajectory=nominal/'trajectory.npz')
    for kind,path in paths.items():scoped.update({kind+'_path':str(path),kind+'_sha256':admission.sha(path)})
    scoped_path=directory/'scoped_admission.json';write(scoped_path,scoped)
    scoped_sha=admission.sha(scoped_path)
    inputs={str(path):admission.sha(path) for path in paths.values()}
    inputs[str(base_dir/'tau.pt')]=base_sha
    new=copy.deepcopy(base);new.update(base_R_sha256=base_sha,input_sha256=inputs,
        scoped_admission_sha256=scoped_sha)
    new['model']['net.0.weight'].add_(.1);torch.save(new,directory/'tau.pt')
    checkpoint_sha=admission.sha(directory/'tau.pt')
    manifest=dict(schema='ref2dex.wam-recovery-R-fit.v1',status='COMPLETED',
        checkpoint_sha256=checkpoint_sha,base_R_sha256=base_sha,input_sha256=inputs,
        admission_path=str(scoped_path),admission_sha256=scoped_sha)
    result=dict(status='COMPLETED',offline_screen_pass=True,checkpoint_sha256=checkpoint_sha,
        base_R_sha256=base_sha,admission_path=str(scoped_path),admission_sha256=scoped_sha)
    write(directory/'result.json',result)
    manifest['result_sha256']=admission.sha(directory/'result.json')
    write(directory/'manifest.json',manifest)
    return directory,dict(input_sha256={str(base_dir/'tau.pt'):base_sha})


def test_default_selection_keeps_base_and_no_override(evidence):
    _,head=evidence
    path,identity,_=admission.retargeter_admission(None,head)
    assert path.parent.name=='competent-pd-fit-20261011-r1'
    assert identity['R_override'] is False and not identity['R_identity_changed_since_head']


def test_override_authenticates_full_chain_and_records_head_mismatch(evidence):
    directory,head=evidence
    path,identity,hashes=admission.retargeter_admission(directory,head)
    assert path==directory/'tau.pt' and identity['R_override'] is True
    assert identity['R_identity_changed_since_head'] is True
    assert identity['head_training_R_sha256']==admission.BASE_R_SHA
    assert hashes[str(path)]==identity['execution_R_sha256']


def test_failed_offline_screen_refuses_execution(evidence):
    directory,head=evidence
    path=directory/'result.json';value=json.loads(path.read_text());value['offline_screen_pass']=False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):admission.retargeter_admission(directory,head)


def test_source_mutation_refuses_execution(evidence):
    directory,head=evidence
    manifest=json.loads((directory/'manifest.json').read_text())
    path=next(p for p in manifest['input_sha256'] if p.endswith('native/trajectory.npz'))
    from pathlib import Path
    Path(path).write_bytes(b'different recorded trajectory')
    with pytest.raises(ValueError):admission.retargeter_admission(directory,head)


def test_metric_result_mutation_refuses_execution(evidence):
    directory,head=evidence
    path=directory/'result.json';value=json.loads(path.read_text())
    value['final']={'fabricated_metric':0.0};path.write_text(json.dumps(value))
    with pytest.raises(ValueError):admission.retargeter_admission(directory,head)
