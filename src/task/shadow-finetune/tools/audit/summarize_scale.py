"""Check matching and report paired native small/full results after all stages finish."""
import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ('q', 'dq', 'object', 'hand', 'hand_root', 'observation', 'goal', 'success', 'reset')


def summarize(root):
    c = json.loads((root/'controls.json').read_text())
    rows, manifests, initial = {}, {}, {}
    for arm in ('small', 'full'):
        train = root/('train-'+arm)
        m = json.loads((train/'manifest.json').read_text())
        if m['status'] != 'COMPLETED' or m['comparison'] != c or m['data'] != c['training_caches'][arm]:
            raise ValueError('incomplete training or controls mismatch: '+arm)
        if m['seed'] != c['settings']['seed'] or m['updates'] != c['settings']['updates'] or m['batch_size'] != c['settings']['batch_size']:
            raise ValueError('training budget or seed mismatch')
        if m['parent'] != c['settings']['parent']:
            raise ValueError('parent checkpoint mismatch')
        if not np.array_equal(np.load(train/'validation.npy'), c['validation_ids']):
            raise ValueError('validation panel mismatch')
        manifests[arm] = m
        curve = json.loads((train/'curve.json').read_text())
        final = next(r for r in curve if r['step'] == c['settings']['updates'])
        native = root/('native-final-'+arm)
        report = json.loads((native/'comparison.json').read_text())
        if report['status'] != 'COMPLETED' or report['seed'] != c['native_reset_seed']:
            raise ValueError('native reset seed or status mismatch')
        if max(report['initial_state_max_error'].values()) > 1e-6:
            raise ValueError('native teacher/student initial mismatch')
        if not report['roles']['student']['controller_audit']['student'] or report['roles']['teacher']['controller_audit']['student']:
            raise ValueError('native role wiring mismatch')
        identities = json.loads((native/'teacher/environment-objects.json').read_text())
        if identities != json.loads((native/'student/environment-objects.json').read_text()):
            raise ValueError('teacher/student object identity mismatch')
        with np.load(native/'teacher/episode-00.npz', allow_pickle=False) as z:
            initial[arm] = {k:z[k][0].astype(np.float64) for k in FIELDS}
        rows[arm] = dict(final=final, selected_offline=min(curve,key=lambda r:r['action']),
                         native=report, environment_objects=identities)
    a, b = manifests['small'], manifests['full']
    for key in ('parent_step','config','stats','parameters','git_commit','code_root'):
        if a[key] != b[key]:
            raise ValueError('matched implementation differs: '+key)
    with np.load(root/'train-small/initial-actor.npz') as a:
        with np.load(root/'train-full/initial-actor.npz') as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):
                raise ValueError('initial actor tensors differ')
    if rows['small']['environment_objects'] != rows['full']['environment_objects']:
        raise ValueError('cross-arm object panel differs')
    initial_error = {k:float(np.abs(initial['small'][k]-initial['full'][k]).max()) for k in FIELDS}
    if max(initial_error.values()) > 1e-6:
        raise ValueError('cross-arm native initial states differ')
    gain = 1-rows['full']['final']['command_first_mae']/max(rows['small']['final']['command_first_mae'],1e-12)
    dev_success = {arm:r['native']['roles']['student']['by_split']['dev']['official_success'] for arm,r in rows.items()}
    invalid = any(r['native']['roles']['student']['controller_audit']['invalid_envs'] for r in rows.values())
    signal = 'PROMISING' if gain >= .2 and dev_success['full']-dev_success['small'] >= 2 and not invalid else 'UNCLEAR'
    return dict(status='COMPLETED',matched_checks='PASS',initial_actor_exact_equal=True,
                cross_arm_initial_max_error=initial_error,rows=rows,first_command_mae_gain=gain,
                dev_success=dev_success,invalid_student_inputs=bool(invalid),signal=signal,
                scope='single-seed fixed-compute Probe; not full-data convergence or an Inspire data-shortage causal claim')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();result=summarize(a.run);out=a.run/'summary'
    out.mkdir(exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','matched_checks','dev_success','first_command_mae_gain','signal')}))


if __name__=='__main__':main()
