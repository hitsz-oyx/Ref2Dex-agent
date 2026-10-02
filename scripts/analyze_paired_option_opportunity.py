"""All-group matching checks and privileged executed candidate ceiling only."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.analyze_static_hold_feasibility import rotation
from scripts.run_contact_response_probe import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / 'run_manifest.json').read_text())
    assert manifest['experiment_id'] == 'P-20261002-paired-option-task-opportunity'
    directory = root / 's601'
    audit = json.loads((directory / 'panel_audit.json').read_text())
    assert audit['run_status'] == 'COMPLETED'
    initial = torch.load(directory / 'initial.pt', map_location='cpu', weights_only=False)
    trace = torch.load(directory / 'trace.pt', map_location='cpu', weights_only=False)
    rows = json.loads((directory / 'rows.json').read_text())
    cluster = initial['option_cluster'].numpy()
    arm = initial['policy_group'].numpy()
    motion = initial['motion'].numpy()
    lookup = np.full((192, 4), -1, dtype=np.int64)
    lookup[cluster, arm] = np.arange(768)
    assert (lookup >= 0).all() and len(np.unique(lookup)) == 768
    q = np.concatenate((initial['base_q'].numpy()[None], trace['native_q'].numpy()[:-1]), 0)
    dq = np.concatenate((initial['initial_dof_vel'].numpy()[None], trace['native_dq'].numpy()[:-1]), 0)
    obj = np.concatenate((initial['object_root'].numpy()[None], trace['object_root'].numpy()[:-1]), 0)
    maxima = dict(initial_state=0.,q_xyz=0.,q_angular=0.,translation_velocity=0.,
                  angular_joint_velocity=0.,object_xyz=0.,object_rotation_matrix=0.)
    for ids in lookup:
        decision = int(initial['decision_steps'][ids[0]])
        assert np.all(initial['decision_steps'][ids].numpy() == decision)
        for states in (initial['base_q'].numpy(), initial['initial_dof_vel'].numpy(), initial['object_root'].numpy()):
            maxima['initial_state'] = max(maxima['initial_state'], float(np.abs(states[ids] - states[ids[0]]).max()))
        pre_q = q[:decision + 1, ids]
        pre_dq = dq[:decision + 1, ids]
        pre_obj = obj[:decision + 1, ids]
        def maximum(key, values):
            maxima[key] = max(maxima[key], float(np.abs(values - values[:, :1]).max()))
        maximum('q_xyz', pre_q[..., :3])
        maximum('q_angular', pre_q[..., 3:])
        maximum('object_xyz', pre_obj[..., :3])
        maximum('translation_velocity', pre_dq[..., :3])
        maximum('translation_velocity', pre_obj[..., 7:10])
        maximum('angular_joint_velocity', pre_dq[..., 3:])
        maximum('angular_joint_velocity', pre_obj[..., 10:13])
        maximum('object_rotation_matrix', rotation(pre_obj[..., 3:7]))
    labels = np.array([r['physical105'] for r in rows], dtype=bool)[lookup]
    agreement = float((labels[:, 0] == labels[:, 1]).mean())
    matching = dict(initial=maxima['initial_state'] <= 1e-7,
                    positions=max(maxima['q_xyz'], maxima['object_xyz']) <= .0005,
                    joint_angles=maxima['q_angular'] <= .005,
                    translational_velocities=maxima['translation_velocity'] <= .01,
                    angular_joint_velocities=maxima['angular_joint_velocity'] <= .1,
                    object_rotation=maxima['object_rotation_matrix'] <= .005,
                    repeat_labels=agreement >= .95)
    candidate_only = labels[:, 2] | labels[:, 3]
    oracle = labels[:, 0] | candidate_only
    pair_motion = motion[lookup[:, 0]]
    gates = dict(pooled_gain5pp_both=all(oracle.mean() >= labels[:, k].mean() + .05 for k in (0, 1)),
                 every_motion_noninferiority_both=all(oracle[pair_motion == m].mean() >= labels[pair_motion == m, k].mean() for m in range(3) for k in (0, 1)),
                 at_least5_option_disagreements=int((labels[:, 2] != labels[:, 3]).sum()) >= 5)
    label = 'UNCLEAR' if not all(matching.values()) else ('PROMISING' if all(gates.values()) else 'UNPROMISING')
    result = dict(run_status='COMPLETED', label=label, matching_gates=matching, prefix_maximum=maxima,
                  baseline_label_agreement=agreement, opportunity_gates=gates,
                  counts_per192=labels.sum(0).tolist(), oracle_count192=int(oracle.sum()),
                  candidate_only_oracle_count192=int(candidate_only.sum()),
                  newly_successful_over_anchor_groups=int((candidate_only & ~labels[:, 0]).sum()),
                  counts_per_motion64={str(m):labels[pair_motion == m].sum(0).tolist() for m in range(3)},
                  oracle_per_motion64={str(m):int(oracle[pair_motion == m].sum()) for m in range(3)},
                  random_candidate_mean_rate=float(labels[:, 2:].mean()),
                  worst_candidate_count192=int((labels[:, 2] & labels[:, 3]).sum()),
                  option_disagreement_groups=int((labels[:, 2] != labels[:, 3]).sum()),
                  conditional_interpretation_allowed=all(matching.values()),
                  privileged_retrospective_oracle_not_deployable=True, no_cm_or_policy_training=True,
                  native_audit_sha256=sha(directory / 'panel_audit.json'))
    assert not (root / 'results.json').exists()
    (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
