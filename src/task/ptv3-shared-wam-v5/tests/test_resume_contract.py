"""Actual entrypoint recovery, batch provenance, loader failure and tau retention."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
import torch
from test_v5_contract import TASK,Geometry,config,stats,fixture as data_fixture
from ptv3_shared_wam_v5.data import sample_window
from ptv3_shared_wam_v5.model import SharedWAMV5
from ptv3_shared_wam_v5.selection import trajectory_score
from ptv3_shared_wam_v5.training_contract import execution_contract,validate_resume,loader_options


class TinyDataset:
    meta=dict(training_allowed=True,engineering_subset=False,pending_sequences=0,source_ids=[1])
    pools={'1':np.array([0,1])}
    fail_at=None
    def __len__(self):return 2
    def draw(self,count,seed,step,rank):return [step]*count
    def __getitem__(self,index):
        if index==self.fail_at:raise RuntimeError('injected data read failure')
        return sample_window(*data_fixture(),3,0,1,[0,0,3])


@pytest.fixture
def entrypoint(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('v5_contract_training',TASK/'tools/run/train_pretrain.py')
    training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
    dataset=TinyDataset()
    monkeypatch.setattr(training,'V5Windows',lambda *args:dataset)
    monkeypatch.setattr(training,'SharedWAMV5',lambda s,c:SharedWAMV5(s,c,Geometry(8)))
    c=dict(config(),seed=0,geometry_lr=.001,shared_lr=.001,weight_decay=.01,clip_grad=1.,
        updates=4,batch_size=1,max_seconds=120,artifact_budget_bytes=32*2**20,
        validation_samples=1,validation_interval=2,validation_initial_step=1,validation_microbatch_size=1,
        checkpoint_interval=2,checkpoint_keep_recent=1,checkpoint_milestone_interval=0,
        save_first_step=False,log_interval=1,warmup_updates=0,amp=False)
    for name in ('WORLD_SIZE','RANK','LOCAL_RANK'):monkeypatch.delenv(name,raising=False)
    def run(name,extra=()):
        for label,value in [('config',c),('stats',stats()),('manifest',{'identity':'same-data'})]:
            (tmp_path/(label+'.json')).write_text(json.dumps(value))
        out=tmp_path/name
        monkeypatch.setattr(sys,'argv',['train','--config',str(tmp_path/'config.json'),
            '--stats',str(tmp_path/'stats.json'),'--manifest',str(tmp_path/'manifest.json'),
            '--output',str(out),'--device','cpu','--workers','0','--seed','0',
            '--git-commit','test-fixture']+list(extra))
        training.main()
        return out
    return training,dataset,c,run


def test_cli_batch_override_cannot_silently_resume(entrypoint):
    _,_,_,run=entrypoint
    original=run('original',['--steps','2'])
    with pytest.raises(ValueError,match='mismatch'):
        run('changed-batch',['--batch-size','2','--resume',str(original/'step000002.pt')])
    saved=torch.load(original/'step000002.pt',weights_only=False)
    assert saved['execution_contract']['effective_global_batch']==1


def test_legacy_resume_requires_and_uses_actual_batch_provenance(entrypoint,tmp_path):
    _,_,_,run=entrypoint
    expected=run('full')
    original=run('original',['--steps','2'])
    saved=torch.load(original/'step000002.pt',weights_only=False)
    saved.pop('execution_contract');saved['selection'].pop('trajectory',None)
    legacy=tmp_path/'legacy';legacy.mkdir();torch.save(saved,legacy/'step000002.pt')
    with pytest.raises(ValueError,match='legacy resume requires'):
        run('no-provenance',['--resume',str(legacy/'step000002.pt')])
    shutil.copyfile(original/'run-manifest.json',legacy/'run-manifest.json')
    shutil.copyfile(original/'dev-step000002.json',legacy/'dev-step000002.json')
    actual=run('migrated',['--resume',str(legacy/'step000002.pt')])
    want=torch.load(expected/'step000004.pt',weights_only=False)
    got=torch.load(actual/'step000004.pt',weights_only=False)
    assert got['resume_binding']['kind']=='legacy-run-manifest-verified'
    for key,value in want['model'].items():torch.testing.assert_close(value,got['model'][key],rtol=0,atol=0)


def test_loader_failure_saves_last_completed_nonperiodic_update(entrypoint):
    _,dataset,_,run=entrypoint;dataset.fail_at=4
    out=run('reader-failure')
    result=json.loads((out/'result.json').read_text())
    assert result['status']=='FAILED_DATA_LOADER' and result['last_step']==3
    assert 'injected data read failure' in result['loader_failure']
    assert (out/'step000003.pt').is_file()


def evaluation(value,objective):
    keys=['hand/h%02d'%i for i in range(1,5)]+['hand/root_relative','hand/contact_phase']
    return dict(source_macro_objective=objective,sources={'1':dict(metrics={k:dict(mean=value,count=1) for k in keys})})


def test_tau_best_survives_when_total_objective_keeps_improving(entrypoint,monkeypatch):
    training,_,c,run=entrypoint;c.update(validation_interval=1,checkpoint_interval=1)
    calls=[]
    def evaluate(*args):
        step=len(calls)+1;calls.append(step)
        return evaluation(step,10-step)
    monkeypatch.setattr(training,'evaluate',evaluate)
    out=run('selection')
    assert (out/'trajectory-best.pt').resolve()==out/'step000001.pt'
    assert (out/'best.pt').resolve()==out/'step000004.pt'
    assert {p.name for p in out.glob('step*.pt')}=={'step000001.pt','step000004.pt'}


def test_tau_score_is_source_macro_and_independent_of_total_loss():
    e=evaluation(3,100);e['sources']['2']=evaluation(9,-100)['sources']['1']
    for x in e['sources']['2']['metrics'].values():x['count']=1000
    assert trajectory_score(e)['score']==6
    e['source_macro_objective']=-1000
    assert trajectory_score(e)['score']==6


def test_execution_contract_and_fork_rejection():
    c=dict(batch_size=128);contract=execution_contract(c,4)
    assert contract['effective_global_batch']==512
    saved=dict(config=c,data_identity='data',world_size=4,execution_contract=contract)
    validate_resume(saved,c,'data',4)
    saved['execution_contract']=dict(contract,effective_local_batch=64)
    with pytest.raises(ValueError,match='effective batch'):validate_resume(saved,c,'data',4)
    assert loader_options(8)['multiprocessing_context']=='spawn'
    assert loader_options(8)['timeout']==120
    with pytest.raises(ValueError):loader_options(8,context='fork')
