import copy
import json

import pytest
import torch

from trajectory_policy import wam_control_admission as admission


def test_adapter_admits_only_selected_complete_candidate_on_exact_C_and_frozen_R(tmp_path,monkeypatch):
    monkeypatch.setattr(admission,'ROOT',tmp_path)
    directory=tmp_path/'outputs/goal-point-wam/context'
    directory.mkdir(parents=True)
    path=directory/'stage1_step2000.pt'
    c=dict(applied=True,checkpoint='C-final',global_step=50500)
    packet=dict(schema=admission.SCHEMA,stage=1,step=2000,adapter={'alpha_p':torch.tensor(.02)},
        trajectory={'weight':torch.ones(2)},manifest=dict(C_identity=c,training_allowed=True),
        contracts=dict(frozen_max_difference=0.,R_max_difference=0.,decoder_difference=0.))
    (directory/'result.json').write_text(json.dumps(dict(status='COMPLETED',final_checkpoint=str(path))))
    torch.save(packet,path)
    identity,_=admission.control_admission(path,c)
    assert identity['applied'] and identity['stage']==1
    with pytest.raises(ValueError):admission.control_admission(path,dict(c,checkpoint='other-C'))
    for key,value in (('R_max_difference',.1),('frozen_max_difference',.1),('decoder_difference',.1)):
        bad=copy.deepcopy(packet);bad['contracts'][key]=value;torch.save(bad,path)
        with pytest.raises(ValueError):admission.control_admission(path,c)
    bad=copy.deepcopy(packet);bad['manifest']['training_allowed']=False;torch.save(bad,path)
    with pytest.raises(ValueError):admission.control_admission(path,c)
    # Decoder changes are permitted only in the explicitly declared stage2.
    packet['stage']=2;packet['contracts']['decoder_difference']=.1;torch.save(packet,path)
    assert admission.control_admission(path,c)[0]['stage']==2


def test_aligned_pair_requires_qualified_exact_training_R(tmp_path,monkeypatch):
    monkeypatch.setattr(admission,'ROOT',tmp_path)
    directory=tmp_path/'outputs/goal-point-wam/aligned';directory.mkdir(parents=True)
    path=directory/'selected.pt';c=dict(applied=True,checkpoint='C-final')
    packet=dict(schema=admission.SCHEMA,stage=2,step=500,adapter={'x':torch.ones(1)},
        trajectory={'x':torch.ones(1)},manifest=dict(C_identity=c,training_allowed=True,
        alignment=True,training_R_sha256='R-exact'),contracts=dict(frozen_max_difference=0.,R_max_difference=0.))
    result=dict(status='COMPLETED',final_checkpoint=str(path),offline_gate=True)
    torch.save(packet,path);(directory/'result.json').write_text(json.dumps(result))
    assert admission.control_admission(path,c,dict(execution_R_sha256='R-exact'))[0]['applied']
    for r in (None,dict(execution_R_sha256='other-R')):
        with pytest.raises(ValueError,match='exact frozen training R'):admission.control_admission(path,c,r)
    result['offline_gate']=False;(directory/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='offline admission'):admission.control_admission(path,c,dict(execution_R_sha256='R-exact'))
