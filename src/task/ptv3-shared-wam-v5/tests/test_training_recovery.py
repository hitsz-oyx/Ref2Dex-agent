"""Actual optimizer/checkpoint/DEV loop on tiny geometry, without CUDA dependency."""
import importlib.util
import json
import sys
import numpy as np
import torch
from test_v5_contract import TASK,Geometry,config,stats,fixture
from ptv3_shared_wam_v5.data import sample_window
from ptv3_shared_wam_v5.model import SharedWAMV5


def test_controlled_stop_saves_last_update_and_resume_matches_uninterrupted(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('v5_training',TASK/'tools/run/train_pretrain.py')
    training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)

    class Dataset:
        meta=dict(training_allowed=True,engineering_subset=False,pending_sequences=0,source_ids=[1])
        pools={'1':np.array([0,1])}
        def __len__(self):return 2
        def draw(self,count,*args):return [0]*count
        def __getitem__(self,index):return sample_window(*fixture(),3,0,1,[0,0,3])

    monkeypatch.setattr(training,'V5Windows',lambda *args:Dataset())
    monkeypatch.setattr(training,'SharedWAMV5',lambda s,c:SharedWAMV5(s,c,Geometry(8)))
    c=dict(config(),seed=0,geometry_lr=.001,shared_lr=.001,weight_decay=.01,clip_grad=1.,
        updates=4,batch_size=1,max_seconds=120,artifact_budget_bytes=16*2**20,
        validation_samples=1,validation_interval=2,validation_initial_step=1,validation_microbatch_size=1,
        checkpoint_interval=2,checkpoint_keep_recent=1,checkpoint_milestone_interval=0,save_first_step=True,
        log_interval=1,warmup_updates=0,amp=False)
    for name,value in [('config',c),('stats',stats()),('manifest',{'identity':'same-data'})]:
        (tmp_path/(name+'.json')).write_text(json.dumps(value))
    for name in ('WORLD_SIZE','RANK','LOCAL_RANK'):monkeypatch.delenv(name,raising=False)
    original_loss=training.forecast_loss

    def run(name,resume=None,stop=False):
        output=tmp_path/name;calls=[0]
        def loss(*args,**kwargs):
            result=original_loss(*args,**kwargs);calls[0]+=1
            if stop and calls[0]==3:(output/'STOP').touch()
            return result
        monkeypatch.setattr(training,'forecast_loss',loss)
        argv=['train','--config',str(tmp_path/'config.json'),'--stats',str(tmp_path/'stats.json'),
            '--manifest',str(tmp_path/'manifest.json'),'--output',str(output),'--device','cpu',
            '--workers','0','--seed','0','--git-commit','test-fixture']
        if resume:argv+=['--resume',str(resume)]
        monkeypatch.setattr(sys,'argv',argv);training.main()
        return output

    full=run('full');stopped=run('stopped',stop=True)
    assert json.loads((stopped/'result.json').read_text())['last_step']==3
    assert json.loads((stopped/'result.json').read_text())['status']=='STOPPED'
    assert (stopped/'step000001.pt').exists() and (stopped/'step000003.pt').exists()
    resumed=run('resumed',stopped/'step000003.pt')
    expected=torch.load(full/'step000004.pt',weights_only=False)
    actual=torch.load(resumed/'step000004.pt',weights_only=False)
    for key,value in expected['model'].items():torch.testing.assert_close(value,actual['model'][key],rtol=0,atol=0)
    assert (resumed/'best.pt').is_file()
    assert actual['selection']['score']==expected['selection']['score']
