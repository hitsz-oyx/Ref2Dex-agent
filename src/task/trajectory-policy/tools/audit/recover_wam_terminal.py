"""Recover ONLY a complete trace whose terminal check failed on relative paths."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/audit'), str(TASK/'tools/run')]
from audit_wam_ref2_shadow import sha, write
from execute_wam_native_bridge import partial_grasp_outcome
from trajectory_policy.wam_calibration_provenance import absolute_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source, out = a.source.resolve(), a.output.resolve()
    owner = ROOT/'outputs/trajectory-policy'
    if owner not in source.parents or out.parent != owner or out.exists() or not out.name.endswith('-recovered'):
        raise ValueError('owned source and fresh derivative output required')
    native = source/'native'
    m = json.loads((native/'manifest.json').read_text())
    g = json.loads((source/'group_status.json').read_text())
    server = json.loads((source/'server/status.json').read_text())
    log = (source/'native.log').read_text()
    controls = m['controls']
    if (g.get('status') != 'FAILED' or m.get('status') != 'FAILED' or controls != 128
            or m.get('executed_controls') != controls or m.get('partial_controls') != controls
            or m.get('partial_measured_states') != controls+1
            or m.get('error') != "FileNotFoundError(2, 'No such file or directory')"
            or "FileNotFoundError: [Errno 2] No such file or directory: 'src/task/trajectory-policy/tools/run/fit_wam_joint_retargeter.py'" not in log
            or server.get('status') != 'COMPLETED' or server.get('queries') != controls
            or sha(native/'partial.npz') != m['partial_sha256']):
        raise ValueError('only known terminal relative-path failure after full128 completion is recoverable')
    with np.load(native/'trajectory.npz', allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    with np.load(native/'partial.npz', allow_pickle=False) as f:
        if set(f.files) != set(d) or any(not np.array_equal(f[k], d[k]) for k in d):
            raise ValueError('saved full trace and terminal snapshot differ')
    if d['q'].shape != (129, 4, 18) or d['applied'].shape != (128, 4, 18) or any(not np.isfinite(v).all() for v in d.values()):
        raise ValueError('finite complete four-role trace required')
    inputs = absolute_hashes(m['input_sha256'], ROOT)
    historical = {}
    for path, digest in inputs.items():
        if sha(path) == digest:
            continue
        path_obj = Path(path)
        if ROOT not in path_obj.parents or path_obj.suffix != '.py':
            raise ValueError('actual data/checkpoint/config input drift: '+path)
        blob = subprocess.check_output(['git', 'show', m['git_commit']+':'+str(path_obj.relative_to(ROOT))], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError('historical code identity differs')
        historical[path] = digest
    outcomes = []
    for env, role in enumerate(m['roles']):
        value = partial_grasp_outcome(dict(object_pose=d['obj'][:, env], object_velocity=d['velocity'][:, env],
                                          surface_gap=d['surface_gap'][:, env], support_gap=d['support_gap'][:, env],
                                          table_footprint=d['table_footprint'][:, env]))
        value.update(env=env, role=role, clipping_rate=float(d['clipped'][:, env].mean()))
        outcomes.append(value)
    result = dict(status='COMPLETED', outcomes=outcomes, partial_episode=True, calibration_pass=None,
                  recovery_pass=None, local_grasp_calibration=all(v['maximum_held_frames'] >= 45 for v in outcomes[:2]),
                  local_grasp_recovery=outcomes[3]['maximum_held_frames'] >= 45 and outcomes[3]['clipping_rate'] < .01,
                  scope='postprocessed existing complete128 trace; no new simulation, no placement verdict',
                  recovered_from=str(source), original_status='FAILED', source_manifest_sha256=sha(native/'manifest.json'))
    out.mkdir()
    target = out/'native'
    target.mkdir()
    for name in ('trajectory.npz', 'expert_queries', 'monitor.jsonl'):
        (target/name).symlink_to(native/name, target_is_directory=name == 'expert_queries')
    write(target/'result.json', result)
    recovered = dict(m, status='COMPLETED', run_id=out.name, input_sha256=inputs,
                     trajectory_sha256=sha(native/'trajectory.npz'), result_sha256=sha(target/'result.json'),
                     recovered_from=str(source), original_manifest_sha256=sha(native/'manifest.json'),
                     original_terminal_error=m['error'], historical_code_verified=historical,
                     postprocess_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    write(target/'manifest.json', recovered)
    write(out/'group_status.json', dict(g, status='COMPLETED', run_id=out.name,
                                      recovered_from=str(source), original_group_sha256=sha(source/'group_status.json'),
                                      original_status='FAILED', postprocessing_only=True))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
