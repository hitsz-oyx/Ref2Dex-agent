"""Replay saved native observations and independently reconstruct control/metrics."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ACTIVE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]
COUPLING = [(7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05), (13, 12, 1.05), (16, 15, .6), (17, 15, .8)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2')
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    # Recreate architecture directly, independently of the deployment adapter.
    def load_model(path):
        packet = torch.load(path, map_location='cpu', weights_only=False)
        layers = []
        for index in [0, 2, 4, 6, 8]:
            weight = packet['model'][str(index) + '.weight']
            layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
            if index < 8:
                layers.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*layers).cuda().eval()
        model.load_state_dict(packet['model'], strict=True)
        return model, {k: packet[k].cuda() for k in ('xmean', 'xscale', 'ymean', 'yscale')}

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    directories = [a.run / 'smoke'] if a.smoke else sorted(a.run.glob('s*_*'))
    check(len(directories) == (1 if a.smoke else 6), 'Complete fixed panel required')
    models, results, audits = {}, {}, {}
    for folder in directories:
        meta = json.loads((folder / 'manifest.json').read_text())
        check(meta['status'] == 'COMPLETED' and meta['engineering_only'] == a.smoke, 'Complete/scope')
        with np.load(folder / 'trajectory.npz') as stream:
            d = {k: stream[k] for k in stream.files}
        count = meta['controls']
        check(all(np.isfinite(v).all() for v in d.values()), 'Nonfinite trajectory')
        check(np.array_equal(d['progress'], np.tile(np.arange(count + 1)[:, None], (1, 4))), 'Clock')
        check(d['raw_observation'].shape == (count, 4, 1442), 'Full observation shape')
        initial_spread = {k: float(abs(d[k][0] - d[k][0, 0]).max()) for k in ('q', 'obj', 'wrist', 'raw_observation')}
        check(max(initial_spread.values()) < 1e-5, 'Matched reset')
        source = np.load(meta['native_input']['reference'])
        reference = source['reference_obj'][:count + 1]
        check(np.max(abs(d['reference_object'] - reference[:, None, :3, 3])) < 1e-6, 'Object reference/clock')
        check(np.max(abs(d['reference_wrist'] - source['reference_current'][:count + 1, None, 4:7])) < 1e-6, 'Wrist reference/clock')
        check(np.array_equal(d['applied'][:, 0], d['teacher_query_action'][:, 0]), 'Teacher calibration action')
        offset = np.array(meta['native_controller']['offset'], np.float32)
        scale = np.array(meta['native_controller']['scale'], np.float32)
        native_action = d['applied'].copy()
        native_action[..., 6:] = (native_action[..., 6:] + 1) / 2
        pd = offset + scale * native_action
        pd[..., :6] += d['q'][:-1, :, :6]
        for child, parent, ratio in COUPLING:
            pd[..., child] = pd[..., parent] * ratio
        pd_difference = float(abs(pd - d['pd_targets']).max())
        check(pd_difference < 1e-6, 'Actual native PD reconstruction')
        replay = {}
        metrics = {}
        for row, role in enumerate(meta['roles']):
            if row:
                if role not in models:
                    models[role] = load_model(meta['checkpoints'][role])
                model, norm = models[role]
                x = torch.tensor(d['raw_observation'][:, row], device='cuda')
                with torch.inference_mode():
                    # Batch64 matches native arithmetic while each saved state remains independent.
                    chunks = []
                    for chunk in x.split(64):
                        n = len(chunk)
                        chunk = chunk.repeat(63 // n + 1, 1)[:64]
                        value = model(((chunk - norm['xmean']) / norm['xscale']).clamp(-5, 5))
                        chunks.append((value * norm['yscale'] + norm['ymean'])[:n])
                    prediction = torch.cat(chunks).cpu().numpy()
                difference = abs(prediction - d['student_prediction'][:, row - 1])
                replay[role] = dict(xyz_max_mm=float(difference[:, :3].max() * 1000),
                                    other_max=float(difference[:, 3:].max()))
                check(replay[role]['xyz_max_mm'] < .1 and replay[role]['other_max'] < 1e-4,
                      'Independent raw-observation checkpoint replay ' + role)
                encoded = d['student_prediction'][:, row - 1]
                # Algebraically invert relative wrist/absolute finger labels without adapter helpers.
                expected = np.zeros((count, 18), np.float32)
                expected[:, ACTIVE[:6]] = (encoded[:, :6] - offset[:6]) / scale[:6]
                expected[:, ACTIVE[6:]] = (encoded[:, 6:] - offset[ACTIVE[6:]]) / scale[ACTIVE[6:]] * 2 - 1
                check(np.max(abs(expected - d['intended'][:, row])) < 1e-6, 'Student action encoding ' + role)
                check(np.max(abs(np.clip(expected, -1, 1) - d['applied'][:, row])) < 1e-6, 'Student actual action ' + role)
            wrist = np.linalg.norm(d['wrist'][1:, row] - d['reference_wrist'][1:, row], axis=-1) * 1000
            obj = np.linalg.norm(d['obj'][1:, row, :3, 3] - reference[1:, :3, 3], axis=-1) * 1000
            trace = np.einsum('tij,tij->t', d['obj'][1:, row, :3, :3], reference[1:, :3, :3])
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi
            support = d['table_footprint'][:, row].astype(bool) & (abs(d['support_gap'][:, row]) <= .02)
            height = d['obj'][:, row, 2, 3] - d['obj'][0, row, 2, 3]
            held = (d['surface_gap'][:, row] <= .01) & ~support & (height >= .03)
            longest = current = 0
            for value in held[1:]:
                current = current + 1 if value else 0
                longest = max(longest, current)
            oracle_pd = offset + scale * np.concatenate((d['teacher_query_action'][:, row, :6],
                (d['teacher_query_action'][:, row, 6:] + 1) / 2), axis=-1)
            oracle_pd[:, :6] += d['q'][:-1, row, :6]
            local = np.linalg.norm(d['pd_targets'][:, row, :3] - oracle_pd[:, :3], axis=-1) * 1000
            metrics[role] = dict(object_mean_mm=float(obj.mean()), object_p95_mm=float(np.percentile(obj, 95)),
                wrist_mean_mm=float(wrist.mean()), wrist_p95_mm=float(np.percentile(wrist, 95)),
                object_rotation_mean_deg=float(angle.mean()), startup_object_mm=float(obj[:16].mean()),
                startup_wrist_mm=float(wrist[:16].mean()), maximum_held_frames=longest, hold45=longest >= 45,
                clipped_fraction=float(d['clipped'][:, row].mean()), own_state_teacher_pd_mean_mm=float(local.mean()),
                own_state_teacher_pd_startup_mm=float(local[:16].mean()))
        outcome = json.loads((folder / 'result.json').read_text())['outcomes']
        check(all(r['maximum_held_frames'] == metrics[r['role']]['maximum_held_frames'] for r in outcome), 'Hold metric replay')
        results[folder.name] = metrics
        audits[folder.name] = dict(initial_spread=initial_spread, pd_max_abs=pd_difference, replay=replay)
    macro = {role: {key: float(np.mean([r[role][key] for r in results.values()]))
                    for key in next(iter(results.values()))[role]} for role in ['teacher', 'original', 'expanded', 'balanced']}
    out = a.run / ('audit-smoke' if a.smoke else 'audit')
    out.mkdir(exist_ok=False)
    (out / 'results.json').write_text(json.dumps(dict(status='PASS', engineering_only=a.smoke,
        sequence_results=results, macro=macro, audits=audits), indent=2) + '\n')
    print(json.dumps(dict(status='PASS', macro=macro), indent=2))


if __name__ == '__main__':
    main()
