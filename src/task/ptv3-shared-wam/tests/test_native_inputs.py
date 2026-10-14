import json
from pathlib import Path
import numpy as np
import pytest
from ptv3_shared_wam.native_inputs import resolve_native_inputs, load_native_arrays
from test_rl_robot import corpus


def packet(root):
    return dict(provenance=dict(source=str(root),canonical='per-sequence geometry',reference='per-sequence reference'),
                config=dict(native_controller=dict(offset=[0.]*18,scale=[1.]*18)))


def test_rl660_requires_explicit_sequence_and_uses_its_own_files(tmp_path):
    root=corpus(tmp_path)
    with pytest.raises(ValueError,match='explicit'):
        resolve_native_inputs(packet(root))
    selected=resolve_native_inputs(packet(root),'dev')
    arrays=load_native_arrays(selected)
    assert selected['sequence']=='dev' and selected['split']=='val'
    assert arrays['oracle'].shape==(64,11,3)
    np.testing.assert_array_equal(arrays['reference'],np.load(root/'dev/trajectory.npz')['reference_obj'])
    assert Path(selected['canonical']).resolve()==(root/'dev/canonical.npz').resolve()
    with pytest.raises(ValueError,match='unknown'):
        resolve_native_inputs(packet(root),'missing')


def test_rl660_rejects_record_escape_and_controller_mismatch(tmp_path):
    root=corpus(tmp_path);p=packet(root);p['config']['native_controller']['scale'][0]=2.
    with pytest.raises(ValueError,match='controller'):
        resolve_native_inputs(p,'train')
    manifest=json.loads((root/'manifest.json').read_text());manifest['records'][0]['path']='../escape'
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='escapes'):
        resolve_native_inputs(packet(root),'train')


def test_selected_motion_disables_only_membership_filter_and_restores_runtime_flag(tmp_path):
    from ptv3_shared_wam.native_inputs import install_selected_motion_contract
    motion=str((tmp_path/'seq').resolve())
    class Task:
        export_rl=False
        def _load_table(self,files):
            assert self.export_rl is True
            self.motion_file=files
    install_selected_motion_contract(Task,dict(kind='rl660',sequence='seq',motion=motion))
    task=Task();task._load_table([motion,motion])
    assert task.export_rl is False and task.motion_file==[motion]
    with pytest.raises(ValueError,match='another sequence'):
        task._load_table([str(tmp_path/'wrong')])


def test_legacy_arrays_keep_the_recorded_oracle_environment(tmp_path):
    root=tmp_path/'legacy';root.mkdir()
    np.savez(root/'trajectory.npz',hand=np.zeros((30,25,11,3),np.float32))
    np.savez(root/'reference.npz',obj=np.tile(np.eye(4),(30,1,1)))
    np.savez(root/'canonical.npz',points=np.zeros((512,3)))
    (root/'manifest.json').write_text(json.dumps(dict(status='COMPLETED')))
    p=dict(provenance=dict(source=str(root),reference=str(root/'reference.npz'),canonical=str(root/'canonical.npz')))
    result=load_native_arrays(resolve_native_inputs(p))
    assert result['oracle'].shape==(30,11,3)


def test_R_only_override_checks_source_units_normalizers_and_chunk_shape(tmp_path):
    import torch
    from ptv3_shared_wam.native_inputs import load_execution_retargeter
    from trajectory_policy.chunk_retargeter import ChunkRetargeter
    net=ChunkRetargeter(128);state=net.state_dict()
    pair=dict(model={'retargeter.'+k:v for k,v in state.items()},
              provenance=dict(source=str(tmp_path/'corpus')),config=dict(native_PD_scale=[1.]*12))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',source=str(tmp_path/'corpus'))))
    path=tmp_path/'R.pt'
    candidate=dict(schema='ref2dex.rl660-R-only.v1',width=128,model=state,native_PD_scale=torch.ones(12))
    torch.save(candidate,path)
    restored=load_execution_retargeter(pair,path)
    assert torch.equal(restored.action_mean,net.action_mean)
    changed={k:v.clone() for k,v in state.items()};changed['state_mean'][0]+=1
    torch.save(dict(candidate,model=changed),path)
    with pytest.raises(ValueError,match='normalizers'):
        load_execution_retargeter(pair,path)
    changed={k:v.clone() for k,v in state.items()};changed['action_queries']=changed['action_queries'][:1]
    torch.save(dict(candidate,model=changed),path)
    with pytest.raises(ValueError,match='8-query'):
        load_execution_retargeter(pair,path)
