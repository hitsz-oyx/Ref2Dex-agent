"""Freeze six newly self-trained roles for observational continuous collection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from consequence_evaluator.contracts import is_within
from consequence_evaluator.provenance import self_trained_ancestry

ROLES = ('airplane_base', 'mixed12', 'train5', 'balanced5', 'duck', 'cup')
OBJECT_ROUTE = dict(airplane='airplane_base', alarmclock='mixed12', apple='airplane_base',
                    cubesmall='airplane_base', cup='cup', duck='duck', mug='train5',
                    phone='airplane_base', toothpaste='balanced5', waterbottle='airplane_base')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(runs, owned_root):
    """Verify ancestry and qualification identity without claiming all roles succeed.

    Bad/suboptimal trained actors are useful observational examples; their
    operational gate remains explicit, and the labeler must independently
    establish actual clean progress anchors and strict local preferences.
    """
    if set(runs) != set(ROLES):
        raise ValueError('exactly six distinct named expert roles required')
    experts, frozen = {}, {}
    for role in ROLES:
        run = Path(runs[role]).resolve()
        frozen.update(self_trained_ancestry(run, owned_root))
        trained = json.loads((run/'run_manifest.json').read_text())
        if trained.get('run_id') != role:
            raise ValueError('training run does not implement the declared role: '+role)
        qualification = run.parent/'qualification-s290'
        qm = json.loads((qualification/'run_manifest.json').read_text())
        q = json.loads((qualification/'qualification.json').read_text())
        rows = q.get('per_episode', [])
        if (qm.get('status') != 'COMPLETED' or not qm.get('full_frame0')
                or qm.get('reference_action_lead') is not None
                or qm.get('checkpoint_sha256') != trained['checkpoint_sha256']
                or q.get('episodes') != 64 or len(rows) != 64
                or {r['env_id'] for r in rows} != set(range(64))
                or not all(type(r.get('qualified')) is bool for r in rows)
                or q.get('qualified_episodes') != sum(r['qualified'] for r in rows)
                or q.get('data_readiness_pass') != (q['qualified_episodes'] >= 8)):
            raise ValueError('completed learned-policy qualification does not match role: '+role)
        for name, key in (('transitions.pt', 'transitions_sha256'),
                          ('native-results.json', 'native_results_sha256')):
            path = qualification/name
            if sha(path) != q[key]:
                raise ValueError('qualification evidence changed: '+str(path))
            frozen[str(path)] = q[key]
        for name in ('run_manifest.json', 'qualification.json'):
            path = qualification/name
            frozen[str(path)] = sha(path)
        experts[role] = dict(checkpoint=trained['checkpoint'], sha256=trained['checkpoint_sha256'],
                             training_run=str(run), qualification=str(qualification),
                             operational_qualified_episodes=q['qualified_episodes'],
                             data_readiness_pass=q['data_readiness_pass'])
    if len({e['sha256'] for e in experts.values()}) != 6:
        raise ValueError('six distinct self-trained endpoints required; no weight substitution')
    if not experts['airplane_base']['data_readiness_pass']:
        raise ValueError('the default learned expert must pass operational readiness')
    all_operational = all(e['data_readiness_pass'] for e in experts.values())
    return dict(description='New self-trained route; failed roles are explicitly suboptimal candidates',
                default_expert='airplane_base', experts=experts, object_route=OBJECT_ROUTE.copy(),
                all_experts_operationally_qualified=all_operational,
                training_allowed=all_operational, sources=frozen,
                limitation=('all six operational qualification passed; measured collection and pair coverage remain required'
                            if all_operational else
                            'route provenance only; failed/suboptimal experts are observational candidates'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True, help='role=/absolute/owned/training/run')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = {}
    for value in a.run:
        role, separator, run = value.partition('=')
        if not separator or role in runs:
            p.error('unique role=training-run entries required')
        runs[role] = run
    output = a.output.absolute()
    if output.exists() or not is_within(output.resolve(), ROOT/'outputs/consequence-evaluator'):
        p.error('fresh task-owned route artifact required')
    route = prepare(runs, ROOT/'outputs/consequence-evaluator')
    route['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(route,indent=2)+'\n')
    print(json.dumps(dict(route=str(output),all_experts_operationally_qualified=
                         route['all_experts_operationally_qualified'])))


if __name__ == '__main__':
    main()
