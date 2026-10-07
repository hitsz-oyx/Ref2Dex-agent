"""New route cannot relabel one actor as six or inherit unrelated qualifications."""
import json
from pathlib import Path
import runpy

import pytest

TOOLS = Path(__file__).resolve().parents[1]/'tools/run'
module = runpy.run_path(str(TOOLS/'prepare_expert_route.py'))
prepare, sha, roles = module['prepare'], module['sha'], module['ROLES']


def fixtures(root):
    runs = {}
    for role in roles:
        queue = root/('queue-'+role)
        run = queue/role
        weights = run/'train/GRAB_00000200.pth'
        weights.parent.mkdir(parents=True)
        weights.write_bytes(role.encode())
        manifest = dict(run_id=role,run_status='COMPLETED',cm_enabled=False,
                        initialization='random_scratch',source_epoch=0,source_checkpoint=None,
                        command=['--actual-epochs','200'],checkpoint=str(weights),checkpoint_sha256=sha(weights))
        (run/'run_manifest.json').write_text(json.dumps(manifest))
        qdir = queue/'qualification-s290'
        qdir.mkdir()
        (qdir/'transitions.pt').write_bytes(b'engineering trace placeholder')
        (qdir/'native-results.json').write_text('{}')
        count = 8 if role!='duck' else 0
        q = dict(episodes=64,qualified_episodes=count,data_readiness_pass=count>=8,
                 per_episode=[dict(env_id=i,qualified=i<count) for i in range(64)],
                 transitions_sha256=sha(qdir/'transitions.pt'),native_results_sha256=sha(qdir/'native-results.json'))
        (qdir/'qualification.json').write_text(json.dumps(q))
        (qdir/'run_manifest.json').write_text(json.dumps(dict(status='COMPLETED',full_frame0=True,
             reference_action_lead=None,checkpoint_sha256=sha(weights))))
        runs[role] = run
    return runs


def test_route_freezes_genuine_roles_and_preserves_failed_readiness(tmp_path):
    route = prepare(fixtures(tmp_path),tmp_path)
    assert len(route['experts']) == 6
    assert route['object_route']['cup']=='cup'
    assert route['experts']['duck']['operational_qualified_episodes']==0
    assert not route['all_experts_operationally_qualified'] and not route['training_allowed']
    assert len(route['sources']) == 6*6


@pytest.mark.parametrize('defect',['role','qualification','trace','duplicate','default'])
def test_invalid_role_or_evidence_cannot_enter_route(tmp_path,defect):
    runs=fixtures(tmp_path)
    run=runs['cup']; qdir=run.parent/'qualification-s290'
    if defect=='role':
        runs['cup']=runs['duck']
    elif defect=='qualification':
        p=qdir/'run_manifest.json';m=json.loads(p.read_text());m['reference_action_lead']=1;p.write_text(json.dumps(m))
    elif defect=='trace':
        (qdir/'transitions.pt').write_bytes(b'changed')
    elif defect=='duplicate':
        p=run/'run_manifest.json';m=json.loads(p.read_text());Path(m['checkpoint']).write_bytes(b'duck');m['checkpoint_sha256']=sha(m['checkpoint']);p.write_text(json.dumps(m));p=qdir/'run_manifest.json';q=json.loads(p.read_text());q['checkpoint_sha256']=m['checkpoint_sha256'];p.write_text(json.dumps(q))
    else:
        p=runs['airplane_base'].parent/'qualification-s290/qualification.json';q=json.loads(p.read_text());q.update(qualified_episodes=0,data_readiness_pass=False)
        for row in q['per_episode']:row['qualified']=False
        p.write_text(json.dumps(q))
    with pytest.raises(ValueError):prepare(runs,tmp_path)
