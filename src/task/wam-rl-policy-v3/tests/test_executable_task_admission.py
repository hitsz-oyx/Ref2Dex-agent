"""The actual PPO environment must expose, rather than relabel, UNKNOWN tasks."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from wam_rl_policy_v3.task_spec import TaskSpec
from wam_rl_policy_v3.task_scheduler import TaskScheduler
from wam_rl_policy_v3.measured_official_env import MeasuredOfficialEnv


def task(mode):
    poses=np.repeat(np.eye(4)[None],2,0);poses[1,2,3]=.08
    return TaskSpec.from_reference(poses,source_fps=30,terminal_mode=mode)


def corpus(tmp_path,modes=('HOLD','UNKNOWN','RELEASE')):
    rows=[]
    for i,mode in enumerate(modes):
        source=tmp_path/f'source-{i}';side=tmp_path/f'side-{i}';source.mkdir();side.mkdir()
        np.savez(source/'canonical.npz',points=np.zeros((32,3),np.float32))
        np.savez(side/'admission.npz',table_bounds=np.zeros((2,3)))
        (side/'task.json').write_text(json.dumps(task(mode).packet()))
        rows.append(dict(sequence=f'seq-{i}',source=str(source),sidecar=str(side),object='cube',split='train',engineering='PASS',training_prefix_commands=1))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(dict(schema='ref2dex.teacher-measured-rollout-admission.v2',status='COMPLETED',records=rows)))
    return path


def make_env(path,monkeypatch):
    monkeypatch.setattr(MeasuredOfficialEnv,'_refresh',lambda *args,**kwargs:None)
    model=SimpleNamespace(shape=lambda p:torch.zeros(len(p),64))
    return MeasuredOfficialEnv(None,dict(state=dict(q=np.zeros((1,18)))),path,SimpleNamespace(device=torch.device('cpu')),model)


def test_scheduler_refuses_unknown_before_any_transition():
    with pytest.raises(ValueError,match='UNKNOWN.*success'):
        TaskScheduler(task('UNKNOWN'),np.random.default_rng(17).normal(size=(32,3))*.02)


def test_actual_environment_excludes_and_accounts_unknown_without_relabel(tmp_path,monkeypatch):
    path=corpus(tmp_path);before=(tmp_path/'side-1/task.json').read_bytes()
    env=make_env(path,monkeypatch)
    assert set(env.records)=={'seq-0','seq-2'}
    assert env.task_admission['eligible_count']==3
    assert env.task_admission['admitted_count']==2
    assert env.task_admission['rejected_count']==1
    assert env.task_admission['rejected'][0]['sequence']=='seq-1'
    assert env.task_admission['rejected'][0]['terminal_mode']=='UNKNOWN'
    assert (tmp_path/'side-1/task.json').read_bytes()==before


def test_all_unknown_is_an_explicit_setup_failure(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='no executable.*UNKNOWN'):
        make_env(corpus(tmp_path,('UNKNOWN',)),monkeypatch)


def test_native_sampler_uses_same_admission_and_reports_every_rejection(tmp_path):
    from wam_rl_policy_v3.task_admission import admitted_native_records,load_executable_tasks
    path=corpus(tmp_path);rows=json.loads(path.read_text())['records']
    native,report=admitted_native_records(rows,path)
    entries,policy_report=load_executable_tasks(path)
    assert [r['sequence'] for r in native]==[r['sequence'] for r,_,_ in entries]
    assert report==policy_report and report['rejected_count']==1
    with pytest.raises(ValueError,match='geometry'):
        admitted_native_records(rows[:1],path)


def test_old_unknown_admission_checkpoint_cannot_resume_as_same_experiment():
    from wam_rl_policy_v3.resume_identity import validate_resume_identity
    from wam_rl_policy_v3.task_admission import POLICY
    old=dict(wam='v4',physics_backend='gpu')
    current=dict(old,task_admission_policy=POLICY)
    with pytest.raises(ValueError,match='task_admission_policy'):
        validate_resume_identity(old,current,migrate_physics=True)


def test_legacy_native_sampler_cannot_hide_unknown_until_a_later_reset(tmp_path):
    from wam_rl_policy_v3.task_admission import load_executable_tasks,require_native_membership
    _,report=load_executable_tasks(corpus(tmp_path))
    require_native_membership(['seq-0','seq-2'],report)
    with pytest.raises(ValueError,match='outside executable admission'):
        require_native_membership(['seq-0','seq-1','seq-2'],report)
    with pytest.raises(ValueError,match='declare'):
        require_native_membership(None,report)


def test_prefix_created_unknown_is_not_silently_dropped_before_migration(tmp_path):
    from wam_rl_policy_v3.task_admission import load_executable_tasks
    path=corpus(tmp_path)
    (tmp_path/'side-1/raw-task.json').write_text(json.dumps(task('RELEASE').packet()))
    with pytest.raises(ValueError,match='prefix-rewritten.*full declaration'):
        load_executable_tasks(path)
