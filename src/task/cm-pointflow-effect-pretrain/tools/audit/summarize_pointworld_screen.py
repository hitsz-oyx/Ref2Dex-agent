#!/usr/bin/env python3
"""Read-only ref3 screen: equal-update final tests and frozen numeric gates.

While training is live this only reports progress; it never opens test panels.
PASS/FAIL is the predeclared numeric screen, not a scientific conclusion.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
ARMS = ('history', 'action', 'shuffle')
PRIMARY = 'model/anchor/cat0/h24/point_epe'
STATIC = 'static/anchor/cat0/h24/point_epe'


def read(path): return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''): h.update(chunk)
    return h.hexdigest()


def screen(values, static, inference_shuffle):
    if not all(math.isfinite(x) and x >= 0 for x in [*values.values(), static, inference_shuffle]):
        raise ValueError('nonfinite or negative physical errors')
    b = values['action']
    gains = {a: 1-b/values[a] if values[a] > 0 else None for a in ('history','shuffle')}
    sensitivity = inference_shuffle/b-1 if b > 0 else None
    gates = dict(versus_history=values['history'] > 0 and b <= .90*values['history'],
                 versus_train_shuffle=values['shuffle'] > 0 and b <= .90*values['shuffle'],
                 versus_static=b < static,
                 inference_shuffle=b > 0 and inference_shuffle >= 1.05*b)
    return dict(screen='PASS' if all(gates.values()) else 'FAIL', gates=gates,
                relative_improvements=gains, inference_shuffle_degradation=sensitivity,
                primary_epe_m=values, static_epe_m=static, inference_shuffle_epe_m=inference_shuffle,
                interpretation='Probe screen only; root must review implementation, coverage and attribution')


def test_panel(data):
    """Reconstruct the fixed primary panel and static error using CPU labels only."""
    import numpy as np
    sys.path.insert(0, str(TASK/'src'))
    from oakink_wm.data import Windows, balanced_indices
    dataset = Windows(data, 'test')
    indices = balanced_indices(dataset, 256, 214)
    errors, identities = [], []
    for idx in indices:
        s = dataset[int(idx)]
        if s['category'] != 0: continue
        anchor = np.flatnonzero(s['object_features'][:,13] > .5)
        if len(anchor) != 1: raise ValueError('expected exactly one anchor')
        m = int(anchor[0]);p = s['points'][m];e = s['effect'][m,-1]
        errors.append(float(np.linalg.norm(p@e[:3,:3].T+e[:3,3]-p,axis=-1).mean()))
        identities.append(tuple(map(int,s['sample_id'])))
    if not errors: raise ValueError('empty moving-anchor test stratum')
    return dict(samples=256, moving_anchor_windows=len(errors),
                unique_moving_windows=len(set(identities)),
                moving_sequences=len({x[0] for x in identities}),
                primary_static_epe_m=float(np.mean(errors)),
                balanced_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())


def summarize(root, data):
    group = read(root/'group_status.json')
    progress = {a: read(root/('train-'+a)/'progress.json') for a in ARMS}
    live = {}
    for name,pid in {'launcher': group.get('pid'), **group.get('processes',{})}.items():
        path = Path('/proc/%s/cmdline'%pid)
        cmd = path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else ''
        live[name] = ('launch_pointworld_group.py' if name=='launcher' else 'train_oakink2_pointworld.py') in cmd
    result = dict(schema='pointworld-wm24.screen.v1', run_dir=str(root),
                  group_status=group['status'], progress=progress, live_processes=live,
                  screen='NOT_READY', scientific_conclusion='UNCLEAR')
    final_ready = (group['status']=='COMPLETED' and group.get('matched_updates') is True and
                   all(progress[a]['status']=='COMPLETED' for a in ARMS) and
                   {progress[a]['step'] for a in ARMS}=={group['updates']} and
                   all((root/('train-'+a)/'test_result.json').exists() for a in ARMS))
    if not final_ready:
        result['reason'] = 'requires completed equal-update final checkpoints and test evaluation'
        return result
    identities = {a:read(root/('train-'+a)/'input_manifest.json') for a in ARMS}
    for key in ('initial_parameter_sha256','dataset_hash','stats_sha256','config_hash',
                'git_commit','script_sha256','model_sha256','data_sha256','vendor_sources'):
        if any(identities[a][key]!=identities['action'][key] for a in ARMS):
            raise ValueError('cross-arm identity mismatch: '+key)
    if any(identities[a]['smoke'] or identities[a]['arm']!=a for a in ARMS):
        raise ValueError('engineering/arm identity mismatch')
    for filename,expected in group['identity']['sources'].items():
        if digest(Path(filename)) != expected: raise ValueError('runtime source drift: '+filename)
    if digest(data/'processed/manifest.json') != identities['action']['dataset_hash']:
        raise ValueError('dataset manifest drift')
    tests = {}
    for a in ARMS:
        out = root/('train-'+a);t=read(out/'test_result.json')
        if t['split']!='test' or t['engineering_only'] or t['arm']!=a or t['step']!=group['updates']:
            raise ValueError('final test identity mismatch: '+a)
        if digest(out/'final.pt') != t['checkpoint_sha256']:
            raise ValueError('final checkpoint/test mismatch: '+a)
        tests[a]=t
    panel=test_panel(data)
    statics=[tests[a]['balanced'][STATIC] for a in ARMS]
    if not all(math.isclose(x,panel['primary_static_epe_m'],rel_tol=1e-5,abs_tol=1e-6) for x in statics):
        raise ValueError('static metric does not match frozen label panel')
    if not math.isclose(tests['action']['shuffle'][STATIC],statics[1],rel_tol=1e-5,abs_tol=1e-6):
        raise ValueError('inference shuffle changed label/static panel')
    result.update(screen({a:tests[a]['balanced'][PRIMARY] for a in ARMS},statics[1],
                         tests['action']['shuffle'][PRIMARY]))
    result.update(panel=panel, identity_checks='PASS', runtime_commit=identities['action']['git_commit'],
                  natural_primary_epe_m={a:tests[a]['natural'][PRIMARY] for a in ARMS})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.output and a.output.exists(): raise FileExistsError('preserve previous screen artifact')
    result=summarize(a.run.resolve(),a.data.resolve())
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
