import copy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ptv3_shared_wam_v4.checkpoint import (save_checkpoint, load_checkpoint, data_identity,
    validate_resume_contract, checkpoint_alias, prune_recovery_checkpoints)
from ptv3_shared_wam_v4.objectives import OBJECTIVE_SEMANTICS, EVALUATION_SEMANTICS
from ptv3_shared_wam_v4.selection import consider_selection, inherit_checkpoint

TASK=Path(__file__).resolve().parents[1]


def test_resume_keeps_historical_best_even_when_first_validation_is_worse(tmp_path):
    out=tmp_path/'old';out.mkdir()
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters())
    stats=dict(recipe=dict(surface_geometry_sha256='bound-surface-digest'))
    config=dict(objective_semantics=OBJECTIVE_SEMANTICS,evaluation_semantics=EVALUATION_SEMANTICS)
    provenance=dict(world_size=1,run_id='old',git_commit='test')
    rng=[dict(torch=torch.get_rng_state(),cuda=torch.empty(0,dtype=torch.uint8))]
    best,_=consider_selection(None,.2,0,out/'step0.pt','old')
    save_checkpoint(out/'step0.pt',model,optimizer,config,stats,0,provenance,rng,best)
    save_checkpoint(out/'step10.pt',model,optimizer,config,stats,10,provenance,rng,best)
    parent=load_checkpoint(out/'step10.pt')
    validate_resume_contract(parent,config,stats,1)
    restored,improved=consider_selection(parent['selection'],.4,10,tmp_path/'new/step10.pt','new')
    assert not improved and restored['step']==0 and restored['score']==.2
    new=tmp_path/'new';new.mkdir()
    inherited=inherit_checkpoint(new,restored['checkpoint'])
    assert load_checkpoint(inherited)['step']==0
    later,improved=consider_selection(restored,.3,20,new/'step20.pt','new')
    assert not improved and later==restored
    later,improved=consider_selection(restored,.1,30,new/'step30.pt','new')
    assert improved and later['step']==30 and later['run_id']=='new'
    changed=copy.deepcopy(stats);changed['recipe']['surface_geometry_sha256']='changed'
    with pytest.raises(ValueError,match='bound data'):
        validate_resume_contract(parent,config,changed,1)
    del parent['objective_semantics']
    with pytest.raises(ValueError,match='legacy objective'):
        validate_resume_contract(parent,config,stats,1)


def test_retention_keeps_historical_best_referenced_by_retained_checkpoints(tmp_path):
    model=torch.nn.Linear(1,1);optimizer=torch.optim.AdamW(model.parameters())
    stats=dict(recipe={});provenance=dict(world_size=1)
    for step,best in ((10,10),(20,10),(30,30),(40,30)):
        selection,_=consider_selection(None,.1,best,tmp_path/('step%d.pt'%best),'run')
        save_checkpoint(tmp_path/('step%d.pt'%step),model,optimizer,{},stats,step,provenance,[],selection)
    checkpoint_alias(tmp_path,'best.pt','step30.pt');checkpoint_alias(tmp_path,'latest.pt','step40.pt')
    files=['step10.pt','step20.pt','step30.pt','step40.pt']
    retained=prune_recovery_checkpoints(tmp_path,files,
        dict(checkpoint_keep_recent=1,checkpoint_milestone_interval=20,updates=100))
    assert set(retained)==set(files)  # milestone20 still needs its best at10


@pytest.mark.parametrize('entry',['launch_pretrain.py','train_pretrain.py'])
def test_unsupported_subset_resume_is_rejected_before_loading_data(tmp_path,entry):
    path=TASK/'tools/run'/entry
    args=[sys.executable,str(path),'--config','missing.json','--recipe','missing',
          '--output',str(tmp_path/'out'),'--artifact-root',str(tmp_path),'--resume-reviewed-subset']
    result=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    assert result.returncode==2
    assert '--resume-reviewed-subset is not supported by WAM-v4' in result.stderr
