"""Self-trained ancestry must survive transfer without accepting imported actors."""
import json
from pathlib import Path
import runpy
import sys

import pytest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK/'src'))
from consequence_evaluator.provenance import sha, self_trained_ancestry


def make_run(root, name, epoch, source=None):
    run = root/name
    weights = run/'train/nn'/('GRAB_%08d.pth' % epoch)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(name.encode())
    manifest = dict(run_status='COMPLETED', cm_enabled=False,
                    command=['--actual-epochs',str(epoch)],
                    checkpoint=str(weights),checkpoint_sha256=sha(weights),
                    initialization='pinned_scratch_resume' if source else 'random_scratch',
                    source_epoch=200 if source else 0,
                    source_checkpoint=str(source) if source else None,
                    source_checkpoint_sha256=sha(source) if source else None)
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    return run,weights


def test_transfer_follows_owned_random_scratch_ancestor(tmp_path):
    parent,weights=make_run(tmp_path,'parent',200)
    child,endpoint=make_run(tmp_path,'transfer',220,weights)
    frozen=self_trained_ancestry(child,tmp_path)
    assert set(frozen)=={str(parent/'run_manifest.json'),str(weights),
                         str(child/'run_manifest.json'),str(endpoint)}


@pytest.mark.parametrize('defect',['external','imported','tamper','epoch','unfinished','cm'])
def test_bad_ancestry_is_rejected(tmp_path,defect):
    owned=tmp_path/'owned';owned.mkdir()
    parent,weights=make_run(tmp_path if defect=='external' else owned,'parent',200)
    child,_=make_run(owned,'child',220,weights)
    path=parent/'run_manifest.json';m=json.loads(path.read_text())
    if defect=='imported':m['initialization']='official_pretrained'
    if defect=='tamper':weights.write_bytes(b'changed')
    if defect=='epoch':m['command']=['--actual-epochs','199']
    if defect=='unfinished':m['run_status']='STARTED'
    if defect=='cm':m['cm_enabled']=True
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):self_trained_ancestry(child,owned)


def test_completed_anneal_window_is_in_actual_dry_run_command(tmp_path,monkeypatch,capsys):
    root=TASK.parents[2]
    loaded=runpy.run_path(str(root/'src/task/CmResidual/tools/run_multitrajectory_baseline_probe.py'))
    loaded['main'].__globals__['gpu_used']=lambda _:0
    motion=tmp_path/'motion';motion.mkdir();(motion/'interaction_hand_inspire.pt').write_bytes(b'fixture')
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(motions=[str(motion)])))
    weights=tmp_path/'GRAB_00000200.pth';weights.write_bytes(b'fixture')
    argv=['train','--output',str(tmp_path/'new'),'--spec',str(spec),
          '--source-checkpoint',str(weights),'--source-sha256',sha(weights),
          '--source-epoch','200','--target-epoch','220','--anneal-start','40',
          '--anneal-end','80','--dry-run']
    monkeypatch.setattr(sys,'argv',argv)
    loaded['main']()
    command=json.loads(capsys.readouterr().out)['command']
    assert command[command.index('--curriculum-anneal-start')+1]=='40'
    assert command[command.index('--curriculum-anneal-end')+1]=='80'
    assert '--curriculum-backtrack-start' not in command
    assert not (tmp_path/'new').exists()
    crossing=argv.copy()
    crossing[crossing.index('--anneal-start')+1]='190'
    crossing[crossing.index('--anneal-end')+1]='210'
    monkeypatch.setattr(sys,'argv',crossing)
    with pytest.raises(SystemExit):loaded['main']()
    assert 'anneal window' in capsys.readouterr().err
