"""Reproduce legacy packets at the actual migration entry point."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from wam_rl_policy_v3.task_spec import TaskSpec


def test_legacy_packet_migration_preserves_bytes_and_is_fresh_output_only(tmp_path,monkeypatch):
    entry=Path(__file__).resolve().parents[1]/'tools/audit/repair_prefix_task_semantics.py'
    loader=importlib.util.spec_from_file_location('prefix_repair',entry)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    source=tmp_path/'source';side=tmp_path/'old/sequences/example';source.mkdir();side.mkdir(parents=True)
    poses=np.repeat(np.eye(4)[None],6,0);poses[:,0,3]=np.arange(6)*.01
    raw=TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD',trim_interaction=False).packet()
    raw.pop('annotation_invalid');raw.pop('annotation_invalid_reasons')
    original=json.dumps(raw,indent=4)+'\n'
    (side/'raw-task.json').write_text(original);(side/'task.json').write_text(json.dumps(raw))
    np.savez(source/'trajectory.npz',obj=poses)
    np.savez(side/'admission.npz',training_prefix=np.arange(6)<4,unsafe_frame=np.zeros(6,bool),coupled=np.ones(6,bool),supported=np.ones(6,bool))
    manifest=tmp_path/'old/manifest.json';manifest.write_text(json.dumps(dict(policy={},records=[dict(sequence='example',source=str(source),sidecar='sequences/example',frames=6,split='train',engineering='PASS',training_prefix_commands=4,object='cube')])))
    out=tmp_path/'fixed';monkeypatch.setattr(sys,'argv',[str(entry),'--audit-manifest',str(manifest),'--output',str(out)])
    module.main()
    assert (out/'sequences/example/task.json').read_text()==original
    assert (out/'sequences/example/raw-task.json').read_text()==original
    assert (side/'raw-task.json').read_text()==original
    assert (out/'sequences/example/admission.npz').read_bytes()==(side/'admission.npz').read_bytes()
    audit=json.loads((out/'repair-audit.json').read_text())
    assert audit['full_task_packet_preserved']==1
    with pytest.raises(ValueError,match='fresh versioned output'):module.main()
