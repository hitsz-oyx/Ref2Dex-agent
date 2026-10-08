"""Reconstruct original motion and audit native DOF/FK correspondence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(ROOT), str(TASK / 'src')]
from consequence_evaluator.contracts import HAND_LINKS, is_within
from consequence_evaluator.data import sha, validate_rigid
from consequence_evaluator.reference_motion import (REFERENCE_SCHEMA, NATIVE_DOF_NAMES,
                                                     original_reference, reconstruct_points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--audit-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or not is_within(out, ROOT / 'outputs/consequence-evaluator'):
        parser.error('fresh task-owned reference output required')
    started = time.monotonic()
    source = args.audit_source.resolve()
    manifest_path = source / 'manifest.json'
    raw = json.loads(manifest_path.read_text())
    motion, urdf = args.motion.resolve(), args.urdf.resolve()
    def frozen_original(path):
        # Collection records the owned symlink path; CLI inputs are resolved.
        # Require the same canonical file AND hash, not spelling equality.
        matches = [value for name, value in raw['sources'].items() if Path(name).resolve() == path]
        return bool(matches) and all(value == sha(path) for value in matches)
    if (raw.get('status') != 'COMPLETED' or raw.get('fps') != 30
            or raw.get('rollout_kind') != 'continuous'
            or not frozen_original(motion) or not frozen_original(urdf)):
        raise ValueError('reference must be the frozen original motion/URDF used by collection')
    frozen = {str(p): sha(p) for p in (motion, urdf, manifest_path, Path(__file__).resolve())}
    helper = ROOT / 'src/task/CmResidual/object_frame_kinematics.py'
    frozen[str(helper)] = sha(helper)
    for name in ('reference_motion.py', 'reset_kinematics.py', 'contracts.py'):
        path = TASK / 'src/consequence_evaluator' / name
        frozen[str(path)] = sha(path)
    reference = original_reference(motion, urdf)
    validate_rigid(reference['object_pose'])
    # Compare FK with independently measured native bodies at startup and
    # after actual controls, in two train episodes. Do not inspect test data.
    errors = []
    records = [r for r in raw['episodes'] if r['split'] == 'train' and r['assigned_phase'] == 'clean'][:2]
    if len(records) != 2:
        raise ValueError('two assigned-clean train episodes needed for the native FK mapping audit')
    for record in records:
        files = [source / record['path'], source / record['diagnostics']]
        for path, key in zip(files, ('sha256', 'diagnostics_sha256')):
            if not is_within(path, source) or sha(path) != record[key]:
                raise ValueError('source packet drift/path escape')
            frozen[str(path)] = record[key]
        with np.load(files[0], allow_pickle=False) as packet, np.load(files[1], allow_pickle=False) as diagnostics:
            indices = np.array([0, 1, 8, 24, 64, len(packet['object_pose']) - 1])
            points = reconstruct_points(diagnostics['q'][indices], diagnostics['hand_root'][indices], urdf)
            errors.append(float(np.max(np.abs(points - packet['hand_keypoints'][indices]))))
            if not np.allclose(packet['object_pose'][0], reference['object_pose'][0], atol=1e-5):
                raise ValueError('actual/reference initial object frames disagree')
    if max(errors) > 1e-5 or any(sha(p) != value for p, value in frozen.items()):
        raise ValueError('native FK link/joint mapping failed or source drifted')
    perturbed = next((r for r in raw['episodes'] if r['split'] == 'train' and r['perturbation_tick'] >= 0), None)
    constraint_audit = None
    if perturbed:
        files = [source / perturbed['path'], source / perturbed['diagnostics']]
        for path, key in zip(files, ('sha256', 'diagnostics_sha256')):
            if not is_within(path, source) or sha(path) != perturbed[key]:
                raise ValueError('perturbed audit packet drift')
            frozen[str(path)] = perturbed[key]
        with np.load(files[0], allow_pickle=False) as packet, np.load(files[1], allow_pickle=False) as diagnostics:
            computed = reconstruct_points(diagnostics['q'], diagnostics['hand_root'], urdf)
            physical_error = np.max(np.abs(computed - packet['hand_keypoints']), axis=(1, 2))
            constraint_audit = dict(episode=perturbed['episode'],
                maximum_coordinate_error_m=float(physical_error.max()),
                frames_above_10micrometres=int(np.sum(physical_error > 1e-5)),
                interpretation='ideal FK can differ from measured PhysX articulation; actual labels use measured points')
    if any(sha(p) != value for p, value in frozen.items()):
        raise ValueError('reference/audit source changed')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'reference.npz', **reference)
    if constraint_audit is not None:
        np.savez_compressed(out / 'perturbed-fk-diagnostic.npz', coordinate_error_m=physical_error)
    manifest = dict(schema=REFERENCE_SCHEMA, status='COMPLETED', origin='original_successful_retargeted_motion',
        task=records[0]['task'], motion=records[0]['motion'], fps=30, units='m',
        hand_links=list(HAND_LINKS), dof_names=list(NATIVE_DOF_NAMES), actor_root='identity; wrist pose in q[:6]',
        reference_joint_semantics='original retargeted q, no PD action conversion or human-keypoint substitution',
        success_semantics='user-specified successful motion reference; FK audit is not physics success validation',
        samples=len(reference['timestamps']), native_fk_max_coordinate_error_m=max(errors),
        native_mapping_audit_episodes=[r['episode'] for r in records], perturbed_fk_diagnostic=constraint_audit,
        sources=frozen, reference_sha256=sha(out / 'reference.npz'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        elapsed_s=time.monotonic() - started,
        device_reason='CPU: bounded URDF geometry reconstruction and file audit, no neural inference')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: manifest[k] for k in ('samples', 'native_fk_max_coordinate_error_m', 'elapsed_s')}))


if __name__ == '__main__':
    main()
