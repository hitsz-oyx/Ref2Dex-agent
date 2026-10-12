"""Matched GT-trajectory execution-horizon development diagnostic; WAM unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/audit')]
from audit_wam_ref2_shadow import sha, write

EXPERIMENT = 'P-20261012-goal-point-wam-ref22-r-horizon'
HORIZONS = (1, 4, 8, 16, 24)
PANEL = (0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 518)
SOURCES = (
    ('t8', 'wam-ref2-execution-startup-anchor-20261012-r1', 8, 128, 2316,
     '7c586a79eb8c859405cca1be9cda3b7a28505136d4464f2a24a2e8f80bf343a3'),
    ('t16', 'wam-ref2-execution-recovery-20261012-r2', 16, 542, 9768,
     'b0edec52551bc9e7f528c80022f7aad946793d02e450fc234bac8eabde8c32ee'),
    ('t32', 'wam-ref2-execution-aggcollect-20261012-r1', 32, 192, 3468,
     '27da16275bdbc2c482ac6bfa164bf206dba4781cebc0c6b07c9a01d180329102'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engineering-smoke', action='store_true')
    a = parser.parse_args()
    owner = ROOT/'outputs/trajectory-policy'
    out = a.output.resolve()
    if out.exists() or out.parent != owner or not out.name.startswith('wam-ref2-r-horizon-'):
        raise ValueError('fresh owned horizon output required')
    if subprocess.check_output(['nvidia-smi', '-i', '2', '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU2 occupied')
    scratch = ROOT/'tmp/trajectory-policy/wam-r-horizon'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='2', TMPDIR=str(scratch))
    import torch
    from trajectory_policy.pd_retargeter import load_model, windows, features, encode_target, decode_target, command, pd_from_action
    from trajectory_policy.wam_retargeter_admission import BASE_R_SHA
    from trajectory_policy.wam_joint import buffer_hash, state_hash
    torch.set_num_threads(2)
    torch.manual_seed(290)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(291)
    started = time.monotonic()
    evidence = {}

    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('input identity differs: '+str(path))
        evidence[str(path)] = digest
        return digest

    def read(path):
        bind(path)
        return json.loads(path.read_text())

    base = owner/'competent-pd-fit-20261011-r1/tau.pt'
    bind(base, BASE_R_SHA)
    packet = torch.load(base, map_location='cpu', weights_only=False)
    source = owner/'competent-pd-teacher-20261011-r2'
    sm, sr, sa = [read(source/k) for k in ('manifest.json', 'result.json', 'audit.json')]
    if sm.get('status') != 'COMPLETED' or sm.get('training_allowed') is not True or not sr.get('teacher_readiness') or not sa.get('pass'):
        raise ValueError('qualified nominal teacher required')
    for k in ('trajectory.npz', 'manifest.json', 'result.json', 'audit.json'):
        bind(source/k, packet['input_sha256'][str(source/k)])
    with np.load(source/'trajectory.npz', allow_pickle=False) as f:
        nominal = {k: f[k][:, :28].copy() for k in ('q', 'dq', 'hand', 'obj', 'velocity', 'pd_targets')}
    arrays = {}
    row_ids = {}
    values = windows(nominal, list(range(16)))
    arrays['nominal_train'] = values[:3]
    row_ids['nominal_train'] = np.asarray([(t, e) for t in range(519) for e in range(16)])
    values = windows(nominal, list(range(24, 28)))
    rows = np.asarray([(t, e) for t in range(519) for e in range(24, 28)])
    for name, mask in [('nominal_report', np.isin(values[3], PANEL)),
                       ('startup_all', values[3] < 16), ('startup_t0', values[3] == 0)]:
        arrays[name] = tuple(v[mask] for v in values[:3])
        row_ids[name] = rows[mask]
    admissions = {}
    for name, folder, takeover, controls, checks, digest in SOURCES:
        native = owner/folder/'native'
        nm, nr, na = [read(native/k) for k in ('manifest.json', 'result.json', 'root_recovery_audit.json')]
        bind(native/'trajectory.npz', digest)
        if (nm.get('status') != 'COMPLETED' or nr.get('status') != 'COMPLETED'
                or nm.get('controls') != controls or nm.get('takeover_tick') != takeover
                or nm.get('expert_PD_training_allowed') is not True
                or nm.get('mixed_policy_future_hand_training_allowed') is not False
                or nm.get('trajectory_sha256') != digest or na.get('native_data_sha256') != digest
                or na.get('pass') is not True or na.get('checks_max') != 0
                or na.get('check_count') != checks or nm['native_controller'] != sm['native_controller']):
            raise ValueError('authenticated own-row expert source required: '+name)
        with np.load(native/'trajectory.npz', allow_pickle=False) as f:
            data = {k: f[k].copy() for k in ('q', 'dq', 'hand', 'obj', 'velocity', 'action', 'applied',
                                           'pd_targets', 'expert_action', 'expert_PD', 'expert_applied', 'expert_clipped')}
        expected = np.zeros((controls, 2), bool)
        expected[takeover:, 1] = True
        if (data['q'].shape != (controls+1, 4, 18) or not np.array_equal(data['action'], data['applied'])
                or data['expert_applied'].dtype != bool or not np.array_equal(data['expert_applied'], expected)
                or data['expert_clipped'][takeover:, 1].any()
                or not np.array_equal(data['applied'][takeover:, 3], data['expert_action'][takeover:, 1])
                or not np.array_equal(data['pd_targets'][takeover:, 3], data['expert_PD'][takeover:, 1])
                or any(not np.isfinite(v).all() for v in data.values())):
            raise ValueError('actual expert controls/futures differ: '+name)
        offset, scale = [torch.as_tensor(nm['native_controller'][k]) for k in ('offset', 'scale')]
        rebuilt = pd_from_action(torch.from_numpy(data['expert_action'][takeover:, 1]),
                                torch.from_numpy(data['q'][takeover:controls, 3]), offset, scale).numpy()
        error = float(np.abs(rebuilt-data['expert_PD'][takeover:, 1]).max())
        if error > 1e-6:
            raise ValueError('current-q action label differs')
        ticks = np.arange(takeover, min(controls-23, takeover+137))
        if any(not data['expert_applied'][t:t+24, 1].all() for t in ticks):
            raise ValueError('mixed-policy future denied')
        future = data['hand'][ticks[:, None]+np.arange(1, 25), 3]
        state, tau = features(*(data[k][ticks, 3] for k in ('q', 'dq', 'hand', 'obj', 'velocity')), future)
        label = encode_target(data['pd_targets'][ticks, 3], data['q'][ticks, 3])
        arrays['recovery_'+name] = (state, tau, label)
        row_ids['recovery_'+name] = np.c_[ticks, np.full(len(ticks), 3)]
        admissions[name] = dict(source=str(native), sha256=digest, role=3, ticks=ticks.tolist(),
                                current_q_PD_reconstruction_max=error,
                                training_allowed=name == 't8', scope='own actual expert future only; development diagnostic')
    for path in (Path(__file__), TASK/'src/trajectory_policy/pd_retargeter.py',
                 TASK/'src/trajectory_policy/retargeter_data.py', TASK/'src/trajectory_policy/wam_joint.py'):
        bind(path)
    if any(not all(np.isfinite(v).all() for v in values) for values in arrays.values()):
        raise ValueError('nonfinite supervision')
    tensors = {k: tuple(torch.as_tensor(v, device='cuda') for v in values) for k, values in arrays.items()}
    models = {k: load_model(dict(packet, execution_horizon=k), 'cuda') for k in HORIZONS}
    hashes = {k: state_hash(m.state_dict()) for k, m in models.items()}
    norms = {k: buffer_hash(m) for k, m in models.items()}
    if len(set(hashes.values())) != 1:
        raise ValueError('matched initial weights required')
    for m in models.values():
        for p in m.parameters():
            p.requires_grad_(True)
    opts = {k: torch.optim.AdamW(m.net.parameters(), lr=1e-4, weight_decay=.01) for k, m in models.items()}
    out.mkdir()
    manifest = dict(schema='ref2dex.wam-r-horizon-fit.v1', status='RUNNING', experiment_id=EXPERIMENT,
                    run_id=out.name, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    physical_gpu=2, seed=290, draw_seed=291, input_sha256=evidence, base_R_sha256=BASE_R_SHA,
                    horizons=list(HORIZONS), updates=1 if a.engineering_smoke else 1000,
                    batch=128, loss='encoded current PD L1 / original target_scale',
                    sampling='16 nominal startup +48 nominal later +64 actual t8 expert recovery',
                    admissions=admissions, WAM_training_allowed=False, physics_allowed=False,
                    limitation='all takeover traces previously inspected; development diagnostic only')
    write(out/'manifest.json', manifest)
    raw = {}
    offset, scale = [torch.as_tensor(sm['native_controller'][k], device='cuda') for k in ('offset', 'scale')]

    def evaluate(model, tag):
        report = {}
        with torch.no_grad():
            for name in ('nominal_report', 'startup_all', 'startup_t0', 'recovery_t8', 'recovery_t16', 'recovery_t32'):
                state, tau, label = tensors[name]
                pred = model(state, tau)
                q = state[:, :18]
                pd, true = decode_target(pred, q), decode_target(label, q)
                delta = (pd-true).cpu().numpy()
                _, _, clipped = command(pd, q, offset, scale)
                report[name] = dict(count=len(q), xyz_mean_mm=float(np.linalg.norm(delta[:, :3], axis=-1).mean()*1000),
                                    xyz_p95_mm=float(np.percentile(np.linalg.norm(delta[:, :3], axis=-1), 95)*1000),
                                    rotation_p95_rad=float(np.percentile(np.abs(delta[:, 3:6]).max(-1), 95)),
                                    finger_p95_rad=float(np.percentile(np.abs(delta[:, [6, 8, 10, 12, 14, 15]]).max(-1), 95)),
                                    clipping_rate=float(clipped.float().mean()))
                raw[tag+'_'+name+'_pd'] = pd.cpu().numpy()
                raw[name+'_true_pd'] = true.cpu().numpy()
                raw[name+'_rows'] = row_ids[name]
        return report

    initial = {str(k): evaluate(m, 'initial_K'+str(k)) for k, m in models.items()}
    steps = manifest['updates']
    nominal_ticks = row_ids['nominal_train'][:, 0]
    startup, later = np.flatnonzero(nominal_ticks < 16), np.flatnonzero(nominal_ticks >= 16)
    sampled = []
    curve = []
    train_started = time.monotonic()
    for step in range(1, steps+1):
        ni = np.r_[rng.choice(startup, 16), rng.choice(later, 48)]
        ri = rng.choice(len(tensors['recovery_t8'][0]), 64)
        sampled.append(np.r_[ni, ri])
        state, tau, target = [torch.cat((tensors['nominal_train'][i][ni], tensors['recovery_t8'][i][ri])) for i in range(3)]
        losses = {}
        for k, model in models.items():
            opts[k].zero_grad(set_to_none=True)
            loss = ((model(state, tau)-target)/model.target_scale).abs().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('nonfinite gradient')
            opts[k].step()
            losses[str(k)] = float(loss)
        if step == 1 or step % 100 == 0:
            pids = set(map(int, subprocess.check_output(['nvidia-smi', '-i', '2', '--query-compute-apps=pid',
                                                       '--format=csv,noheader,nounits'], text=True).split()))
            if pids-{os.getpid()} or time.monotonic()-started > 600:
                raise RuntimeError('foreign GPU process or600s deadline')
            if any(sha(p) != h for p, h in evidence.items()):
                raise ValueError('actual input drift')
            own_bytes = int(subprocess.check_output(['du', '-sb', str(out)], text=True).split()[0])
            if own_bytes >= 100_000_000:
                raise RuntimeError('100MB stage boundary')
            entry = dict(step=step, losses=losses, elapsed_s=time.monotonic()-started,
                         eta_s=(time.monotonic()-train_started)/step*(steps-step),
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            curve.append(entry)
            print(json.dumps(entry), flush=True)
    final = {str(k): evaluate(m, 'final_K'+str(k)) for k, m in models.items()}
    # Shorten only for a material worst-panel improvement without nominal regression.
    scores = {k: max(final[str(k)][p]['xyz_p95_mm'] for p in
                     ('nominal_report', 'recovery_t8', 'recovery_t16', 'recovery_t32')) for k in HORIZONS}
    retained = {k: (final[str(k)]['nominal_report']['xyz_p95_mm'] <= 30
                    and final[str(k)]['startup_all']['xyz_p95_mm'] <= 20
                    and all(v['clipping_rate'] < .01 and v['rotation_p95_rad'] <= .15
                            and v['finger_p95_rad'] <= .3 for v in final[str(k)].values())) for k in HORIZONS}
    eligible = [k for k in HORIZONS if retained[k] and scores[k] <= .8*scores[24]
                and final[str(k)]['nominal_report']['xyz_p95_mm'] <= 1.2*final['24']['nominal_report']['xyz_p95_mm']]
    selected = min(eligible, key=lambda k: (scores[k], k)) if eligible else 24
    if any(buffer_hash(m) != norms[k] for k, m in models.items()):
        raise ValueError('frozen normalization drift')
    checkpoints = {}
    if not a.engineering_smoke:
        for k, model in models.items():
            torch.save(dict(packet, model={n: v.detach().cpu() for n, v in model.state_dict().items()},
                            execution_horizon=k, base_R_sha256=BASE_R_SHA, input_sha256=evidence,
                            run_id=out.name, horizon_diagnostic_only=True), out/('tau_K%d.pt'%k))
            checkpoints[str(k)] = sha(out/('tau_K%d.pt'%k))
    np.savez_compressed(out/'predictions.npz', **raw, sampled_indices=np.asarray(sampled))
    result = dict(status='COMPLETED', initial=initial, final=final, scores=scores, retention=retained,
                  selected_horizon=selected, shortened=selected < 24, normalization_buffers_unchanged=True,
                  initial_hash=hashes, final_hash={k: state_hash(m.state_dict()) for k, m in models.items()},
                  checkpoint_sha256=checkpoints, elapsed_s=time.monotonic()-started, curve=curve,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  conclusion='engineering PASS' if a.engineering_smoke else 'UNCLEAR; development evidence only')
    write(out/'result.json', result)
    manifest.update(status='COMPLETED', result_sha256=sha(out/'result.json'), checkpoint_sha256=checkpoints,
                    predictions_sha256=sha(out/'predictions.npz'))
    write(out/'manifest.json', manifest)
    print(json.dumps(dict(status='COMPLETED', selected_horizon=selected, elapsed_s=result['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
