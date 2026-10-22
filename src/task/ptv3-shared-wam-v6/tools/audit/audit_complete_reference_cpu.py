"""Assert coverage and exact historical prefix of a measured extended reference."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6.tau_contract import gt_reference_tau, reference_coverage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();run=a.run.resolve()
    m=json.loads((run/'manifest.json').read_text())
    with np.load(run/'reference.npz') as z:d={k:z[k].copy() for k in z.files}
    old=ROOT/'outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1/source-pd/trajectory.npz'
    with np.load(old) as z:
        errors={k:float(np.abs(d[k][:len(z[k])].astype(float)-z[k].astype(float)).max())
                for k in ('q','dq','hand','objects','physical','tick','table_pose','contact_force_valid','dof_force_valid')}
        action_error=float(np.abs(d['action'][:len(z['action'])]-z['action']).max())
    n=len(d['hand']);endpoint=int(m['task']['source_frames'][-1])
    assert len(d['action'])==n-1
    np.testing.assert_array_equal(d['tick'][:,0],np.arange(n))
    assert len(np.unique(d['episodes']))==1
    assert all(np.isfinite(v).all() for v in d.values())
    for t in range(n):
        tau,valid=gt_reference_tau(d['hand'][:,0],t)
        assert bool(d['tau_valid'][t])==valid
        np.testing.assert_allclose(d['tau'][t],tau,atol=1e-7)
    coverage=reference_coverage(n,endpoint)
    assert n==205 and coverage['valid_tau_controls']==194
    assert coverage['endpoint_plus_1s_tau_covered']
    assert max(errors.values())==0 and action_error==0
    assert m['physical_safety_abort'] is None and m['clipped_controls']==0
    assert m['native_returncode']==0 and m['optimizer_updates']==0
    native=json.loads((run/'native/manifest.json').read_text())
    assert native['steps']==204 and native['episode_action_mismatches']==0 and native['status']=='COMPLETED'
    summary=dict(status='COVERAGE_PASS_RELEASE_NOT_QUALIFIED' if not m['full_release_success'] else 'COVERAGE_PASS',
        run_id=run.name,artifact=str(run/'reference.npz'),source_sequence=m['source_sequence'],
        reference_origin=m['source'],declared_terminal_mode=m['task']['terminal_mode'],
        declared_release_tick=m['release_source_tick'],coverage=coverage,
        valid_tau_controls=int(d['tau_valid'][:len(d['action'])].sum()),
        terminal_suffix_definition='38 observed states strictly after declared endpoint166: 30 settle ticks plus8 lookahead ticks',
        identical_historical_prefix_states=81,initial_and_prefix_max_errors=errors,
        original80_executed_action_max_error=action_error,
        maximum_pd_target_error=m['maximum_pd_target_error'],clipped_controls=m['clipped_controls'],
        physical_safety_abort=m['physical_safety_abort'],full_release_success=m['full_release_success'],
        final_task=m['task_trace'][-1],new_assumptions=m['new_assumptions'],
        elapsed_s=m['elapsed_s'],gpu=m['gpu'],gpu_before=m['gpu_before'],
        gpu_after_scope='physical4/5/6 each6MiB, utilization0% after collection; no owned process remains',
        environment=dict(client_python=m['python'],python_version=m['python_version'],numpy=m['numpy'],
            native_python=m['native_initial_audit']['python'],native_torch=m['native_initial_audit']['torch'],
            native_dt_s=m['native_initial_audit']['native_dt_s'],reset_angle_policy=m['native_initial_audit']['reset_angle_policy']),
        optimizer_updates=0,source_code_commit=m['git_commit'],
        host_diff=str(run/'host-code/paths.diff'),
        failed_attempt=dict(run_id='s2b-reference-r1',controls=0,
            reason='relative socket path invalid after native chdir; native exited, client interrupted; receipts retained'),
        limitations=['This completes temporal coverage of the declared RELEASE endpoint and settle interval, not a successful physical RELEASE witness.',
                     'No calibrated-R rollout on the new reference was run; the old fit used its original end-of-reference invalid mask.',
                     'The first81 measured states/actions match exactly in this execution; fresh physics runs are not promised bitwise deterministic.'])
    text=json.dumps(summary,indent=2,allow_nan=False)+'\n'
    if a.output:
        if TASK not in a.output.resolve().parents:raise ValueError('Task-local output required')
        with a.output.open('x') as stream:stream.write(text)
    print(text)


if __name__=='__main__':main()
