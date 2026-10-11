"""Independent source, reward, PW conditioning, Q and learned-policy checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
PW = ROOT / 'src/task/cm-pointflow-effect-pretrain'
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/consequence-evaluator/src'), str(PW / 'src'),
               str(Path(__file__).parent)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    scratch = ROOT / 'tmp/trajectory-policy/pw-critic'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch),
        TRITON_CACHE_DIR=str(scratch / 'triton'), TORCH_EXTENSIONS_DIR=str(scratch / 'extensions'))
    import torch
    from audit_competent_actor import independent_input
    from trajectory_policy.reference_actor import load_reference_actor
    from trajectory_policy.pw_critic import PhysicalQ
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    fit = a.fit.resolve()
    m = json.loads((fit / 'manifest.json').read_text())
    result = json.loads((fit / 'result.json').read_text())
    cache = Path(m['cache'])
    cm = json.loads((cache / 'manifest.json').read_text())
    collection = Path(cm['collection'])
    collect_m = json.loads((collection / 'manifest.json').read_text())
    hashes = {}
    for manifest in (m, cm):
        if manifest['status'] != 'COMPLETED':
            raise ValueError('completed producer required')
        for path, digest in manifest['input_sha256'].items():
            if sha(path) != digest:
                raise ValueError('authenticated input drift: ' + path)
            hashes[path] = digest
    if not collect_m['training_allowed'] or collect_m.get('pw_mode') != 'collect' or collect_m['intervention_beta'] != 0:
        raise ValueError('fresh nonintervened physical training labels required')
    if sha(cache / 'windows.npz') != cm['windows_sha256'] or sha(fit / 'learning.npz') != m['learning_sha256']:
        raise ValueError('window/learning evidence changed')
    def load(path):
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k].copy() for k in f.files}
    d = load(cache / 'windows.npz')
    trace = load(collection / 'trajectory.npz')
    reference = load(Path(m['reference']))
    learning = load(fit / 'learning.npz')
    errors = {}
    def close(key, actual, expected, tolerance=2e-5):
        error = float(np.max(np.abs(np.asarray(actual, np.float64) - np.asarray(expected, np.float64))))
        errors[key] = error
        if not np.isfinite(error) or error > tolerance:
            raise ValueError('independent audit differs: ' + key + ' ' + str(error))
    ticks, envs = d['ticks'], d['envs']
    close('query_ticks', ticks, np.repeat(np.arange(0, 519, 4), 16), 0)
    close('query_envs', envs, np.tile(np.arange(8, 24), 130), 0)
    close('whole_env_training_split', d['train'], envs < 20, 0)
    close('measured_history_reference', d['high'], independent_input(trace, ticks, envs, reference))
    close('proposed_tau', d['tau'], trace['high_tau'][ticks, envs - 8], 0)
    close('initial_mean_tau', d['mean'], trace['high_tau_mean'][ticks, envs - 8], 0)
    # Independent next-state reference task rewards and complete discounted windows.
    ob = trace['obj'][1:]
    distance = np.sqrt(np.square(ob[..., :3, 3] - reference['obj'][1:, None, :3, 3]).sum(-1))
    rr = reference['obj'][1:, None, :3, :3].swapaxes(-1, -2) @ ob[..., :3, :3]
    angle = np.arccos(np.clip((np.trace(rr, axis1=-2, axis2=-1) - 1) * .5, -1, 1))
    near = trace['surface_gap'][1:] <= .01
    supported = trace['table_footprint'][1:] & (np.abs(trace['support_gap'][1:]) <= .02)
    held = near & ~supported & ((ob[..., 2, 3] - trace['obj'][0, :, 2, 3]) >= .03)
    rate = np.square(np.diff(np.concatenate((np.zeros_like(trace['action'][:1]), trace['action'])), axis=0)).mean(-1)
    reward = (np.exp(-distance / .05) + .2 * np.exp(-angle / .5) + .5 * held
              + .2 * np.exp(-trace['surface_gap'][1:] / .02) - .1 * trace['clipped'] - .01 * rate)
    close('actual_next_state_reward', d['reward'], reward)
    returns = np.array([sum(float(reward[t + h, e]) * .99**h for h in range(24)) for t, e in zip(ticks, envs)])
    close('actual_discounted_returns', d['returns'], returns)
    train, val = d['train'], ~d['train']
    close('return_train_mean', d['returns'][train].mean(), m['target_mean'])
    close('return_train_std', d['returns'][train].std(), m['target_scale'])
    # Rebuild the first paired PW batch using an independent fixed-frame transform.
    from consequence_evaluator.old_utility import pw_sample, deterministic_group_mean
    from oakink_wm.pointworld_temporal import model_from_config, capped_collate, VENDOR
    from oakink_wm.pointworld_performance import install_fused_hilbert
    import oakink_wm.pointworld as spatial
    import oakink_wm.pointworld_temporal as temporal
    install_fused_hilbert()
    spatial.mean_groups = temporal.mean_groups = deterministic_group_mean
    torch.backends.cuda.matmul.allow_tf32 = True
    canonical = load(ROOT / 'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz')
    checkpoint = torch.load(cm['checkpoint'], map_location='cpu', weights_only=False)
    model = model_from_config(checkpoint['identity']['stats'], checkpoint['config']).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    samples = []
    for i in range(4):
        t, e = int(ticks[i]), int(envs[i])
        past = np.maximum(t + np.arange(-3, 1), 0)
        current = trace['obj'][t, e]
        r, center = current[:3, :3], current[:3, 3]
        oh = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
        oh[:, :3, :3] = r.T @ trace['obj'][past, e, :3, :3]
        oh[:, :3, 3] = (trace['obj'][past, e, :3, 3] - center) @ r
        hh = (trace['hand'][past, e] - center) @ r
        for key in ('tau', 'mean'):
            future = hh[-1][None] + d[key][i].reshape(24, 11, 3)
            sample = pw_sample(oh, hh, future, canonical, len(samples))
            close('PW_proposed_semantic_hand_' + str(len(samples)), sample['action'][:, :11, :3],
                  future[:, [0, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10]])
            if sample['action_valid'][:, 11:].any() or sample['action'][:, 11:].any():
                raise ValueError('absent left hand was unmasked')
            samples.append(sample)
    batch = {k: v.cuda() for k, v in capped_collate(samples).items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(batch, 'action')
    z = torch.cat((prediction['rotation'][:, 0].float(), prediction['translation'][:, 0].float()[..., None]), -1)
    z = z.reshape(4, 2, 24, 12).cpu().numpy()
    close('PW_frozen_prediction', z, np.stack((d['z'][:4], d['z_mean'][:4]), 1), .003)
    # BF16 inference arithmetic is disabled before critic/actor float32 replay.
    torch.backends.cuda.matmul.allow_tf32 = False
    warm = load_reference_actor(torch.load(m['initial'], map_location='cuda', weights_only=False), 'cuda')
    high = torch.tensor(d['high'], device='cuda')
    with torch.no_grad():
        anchor = torch.cat([warm(x) for x in high.split(256)]).cpu().numpy()
    monitor = [json.loads(row) for row in (fit / 'monitor.jsonl').read_text().splitlines()]
    for name, use_pw in (('tau_only', False), ('joint', True)):
        z = d['z'].reshape(-1, 288) if use_pw else np.zeros((len(ticks), 288), np.float32)
        xx = np.concatenate((d['high'], d['tau'].reshape(-1, 792), z), -1)
        packet = torch.load(fit / (name + '_q.pt'), map_location='cuda', weights_only=False)
        qstate = packet['model']
        close(name + '_Q_train_mean', qstate['mean'].cpu().numpy(), xx[train].mean(0, dtype=np.float64))
        close(name + '_Q_train_std', qstate['scale'].cpu().numpy(), np.maximum(xx[train].std(0, dtype=np.float64), .001))
        q = PhysicalQ(qstate['mean'], qstate['scale']).cuda()
        q.load_state_dict(qstate)
        xm = np.concatenate((d['high'], d['mean'].reshape(-1, 792),
                             d['z_mean'].reshape(-1, 288) if use_pw else np.zeros_like(z)), -1)
        with torch.no_grad():
            qp = q(torch.tensor(xx, device='cuda')).cpu().numpy() * m['target_scale'] + m['target_mean']
            qm = q(torch.tensor(xm, device='cuda')).cpu().numpy() * m['target_scale'] + m['target_mean']
        close(name + '_Q_proposal', qp, learning[name + '_q_prediction'])
        close(name + '_Q_mean', qm, learning[name + '_q_mean_prediction'])
        temperature = max(float((qp - qm)[train].std()), .001)
        weights = np.exp(np.clip((qp - qm) / temperature, np.log(.5), np.log(2.)))
        weights /= weights[train].mean()
        close(name + '_advantage_weights', weights, learning[name + '_weight'], .002)
        selected = min((row for row in monitor if row['arm'] == name), key=lambda row: row['validation_mse'])['update']
        if selected != packet['selected_update']:
            raise ValueError('Q validation selection differs')
        final_packet = torch.load(fit / (name + '.pt'), map_location='cuda', weights_only=False)
        actor = load_reference_actor(final_packet, 'cuda')
        trained = torch.load(fit / (name + '_before_backtrack.pt'), map_location='cuda', weights_only=False)
        for key, value in actor.state_dict().items():
            expected = warm.state_dict()[key] + final_packet['backtrack_factor'] * (trained[key] - warm.state_dict()[key])
            close(name + '_trust_parameter_' + key, value.cpu().numpy(), expected.cpu().numpy(), 0)
        with torch.no_grad():
            final = torch.cat([actor(x) for x in high.split(256)]).cpu().numpy()
        close(name + '_learned_tau', final, learning[name + '_final_tau'])
        rms = float(np.sqrt(np.mean((final - anchor)**2)))
        close(name + '_trust_rms', rms, result['reports'][name]['trajectory_change_rms_m'], 1e-7)
        if rms > .00025 or rms <= 0 or result['reports'][name]['actor_gradient_l1'] <= 0:
            raise ValueError('actor update/trust contract differs')
        if sha(fit / (name + '.pt')) != result['checkpoint_sha256'][name]:
            raise ValueError('learned checkpoint changed')
    if result['weight_arm_difference_rms'] <= 0 or result['actor_arm_difference_rms_m'] <= 0:
        raise ValueError('PW did not actually affect policy training')
    evidence = [fit / name for name in ('manifest.json', 'result.json', 'learning.npz', 'tau_only.pt', 'joint.pt',
                'tau_only_q.pt', 'joint_q.pt', 'tau_only_before_backtrack.pt', 'joint_before_backtrack.pt')]
    report = dict(**{'pass': True}, max_errors=errors, source_sha256=sha(__file__),
        evidence_sha256={str(path): sha(path) for path in evidence},
        scope='eligible fresh labels, independent rewards/PW first batch, Q weights, actual actor/trust update; no benefit claim')
    (fit / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'max_errors'}), flush=True)


if __name__ == '__main__':
    main()
