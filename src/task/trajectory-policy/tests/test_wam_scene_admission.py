"""CPU-only inference admission checks; importing server does not run it."""
import importlib.util
from pathlib import Path

import pytest
import torch


path=Path(__file__).resolve().parents[1]/'tools/run/wam_native_server.py'
spec=importlib.util.spec_from_file_location('wam_patch_admission_test',path)
server=importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_original_head_without_patch_retains_no_application():
    assert server.scene_patch_admission(dict(head={'trajectory':{},'hand_query':{}}),{}, {}) == dict(applied=False)


@pytest.fixture
def admission(tmp_path,monkeypatch):
    monkeypatch.setattr(server,'ROOT',tmp_path)
    parent=tmp_path/'outputs/trajectory-policy/parent'
    parent.mkdir(parents=True)
    (parent/'head.pt').write_bytes(b'authenticated parent')
    digest=server.sha(parent/'head.pt')
    packet=dict(schema='ref2dex.wam-visited-pd-dec0.v1',head={'trajectory':{},'hand_query':{}},
        scene_patch={'backbone.dec.dec0.weight':torch.zeros(2,2)},
        scene_patch_schema='ref2dex.wam-scene-dec0.v1',parent_head_sha256=digest)
    manifest=dict(schema=packet['schema'],scene_patch_schema=packet['scene_patch_schema'],
        scene_patch_prefix='backbone.dec.dec0.',warm_head_dir=str(parent),warm_head_sha256=digest,
        input_sha256={str(parent/'head.pt'):digest})
    return packet,manifest,dict(scene_final_hash='a'*64)


def test_valid_patch_provenance_marker(admission):
    packet,manifest,result=admission
    marker=server.scene_patch_admission(packet,manifest,result)
    assert marker['applied'] and marker['scene_sha256']==result['scene_final_hash']
    assert marker['parent_head_sha256']==packet['parent_head_sha256']


@pytest.mark.parametrize('defect',['missing_patch','bad_schema','bad_prefix','unknown_stage','bad_parent','bad_hash','unknown_field'])
def test_patch_admission_rejects_unmatched_schema_provenance(admission,defect):
    packet,manifest,result=admission
    if defect=='missing_patch':
        packet.pop('scene_patch')
    elif defect=='bad_schema':
        packet['scene_patch_schema']='unknown'
    elif defect=='bad_prefix':
        manifest['scene_patch_prefix']='backbone.enc.'
    elif defect=='unknown_stage':
        packet['scene_patch']={'backbone.dec.dec1.weight':torch.zeros(2,2)}
    elif defect=='bad_parent':
        packet['parent_head_sha256']='0'*64
    elif defect=='bad_hash':
        result['scene_final_hash']='not an authenticated hash'
    else:
        packet['scene_patch_enc']={}
    with pytest.raises(ValueError):
        server.scene_patch_admission(packet,manifest,result)
