"""Recompute fixed-panel results and independently measure TRAIN-bank coverage."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT / 'src/task/ptv3-shared-wam-v2'
BASE = ROOT / 'outputs/ptv3-shared-wam-v2'
ACTIVE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output-name', default='audit')
    a = p.parse_args()
    run = a.run.resolve()
    out = run / a.output_name
    if out.parent != run:
        raise ValueError('Audit directory must be directly inside run')
    if out.exists() or BASE not in run.parents:
        raise ValueError('Fresh task audit output required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='1', TMPDIR=str(ROOT / 'tmp'))
    sys.path.insert(0, str(TASK / 'src'))
    import numpy as np
    import torch
    from ptv3_shared_wam_v2.feature_transfer import FeatureControl
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600s audit')))
    signal.alarm(600)
    cache = BASE / 'full-teacher-cache-20261015-r1'
    previous = BASE / 'teacher-sequence-generalization-20261016-r1'
    member = json.loads((previous / 'membership.json').read_text())['groups']
    init = torch.load(previous / 'step0.pt', map_location='cpu', weights_only=False)
    old = torch.load(previous / 'step24000.pt', map_location='cpu', weights_only=False)
    original_ids = np.array(member['seen']['cache_indices'])
    seq = np.load(cache / 'train-sequence.npy')
    tick = np.load(cache / 'train-tick.npy')
    held = np.array(member['interpolation']['cache_indices'])
    result = dict(arms={}, coverage={}, scope='Probe descriptive coverage, no causal equivalence claim')
    out.mkdir()
    arms = [arm for arm in ('original', 'expanded', 'diverse', 'balanced') if (run / arm / 'final.pt').exists()]
    if not {'original', 'expanded'} <= set(arms):
        raise ValueError('Both primary arms required')
    for arm in arms:
        cp = torch.load(run / arm / 'final.pt', map_location='cpu', weights_only=False)
        ids = np.load(run / arm / 'train-cache-indices.npy')
        if arm == 'original':
            assert np.array_equal(ids, original_ids)
        if arm == 'diverse':
            assert len(ids) == len(original_ids) == len(np.unique(ids))
            assert np.isin(ids, np.load(run / 'expanded/train-cache-indices.npy')).all()
        if arm == 'balanced':
            assert np.array_equal(ids, np.load(run / 'expanded/train-cache-indices.npy'))
            draw = np.load(run / arm / 'first-draw.npy')
            assert np.isin(draw, original_ids).sum() == 256
            assert (tick[draw[:256]] < 16).all() and (tick[draw[256:]] >= 16).all()
        # Independent brute support check, without calling the train membership helper.
        for s in np.unique(seq[held]):
            train_ticks = tick[ids[seq[ids] == s]]
            held_ticks = tick[held[seq[held] == s]]
            assert np.min(abs(train_ticks[:, None] - held_ticks[None])) >= 28
        for k in ('xmean', 'xscale', 'ymean', 'yscale'):
            assert torch.equal(cp[k], init[k]), k
        model = FeatureControl().raw.cuda()
        model.load_state_dict(cp['model'], strict=True)
        model.eval()
        mean, scale, ym, ys = [cp[k].cuda() for k in ('xmean', 'xscale', 'ymean', 'yscale')]
        saved = np.load(run / arm / 'predictions-final.npz')
        reported = json.loads((run / arm / 'result.json').read_text())['final']['groups']
        checks = {}
        for group, entry in member.items():
            split, rows = entry['cache_split'], np.array(entry['cache_indices'])
            assert np.array_equal(saved[group + '_cache_indices'], rows)
            q = np.load(cache / (split + '-q.npy'), mmap_mode='r')[rows]
            label = np.load(cache / (split + '-pd.npy'), mmap_mode='r')[rows][:, ACTIVE]
            x = torch.tensor(np.load(cache / (split + '-full.npy'), mmap_mode='r')[rows], device='cuda')
            x = ((x - mean) / scale).clamp(-5, 5)
            with torch.no_grad():
                prediction = (torch.cat([model(b) for b in x.split(512)]) * ys + ym).cpu().numpy()
            prediction[:, :6] += q[:, :6]
            replay_max = float(abs(prediction - saved[group + '_pd']).max())
            assert replay_max < 1e-5, replay_max
            error = np.linalg.norm(saved[group + '_pd'][:, :3] - label[:, :3], axis=1) * 1000
            groups = np.load(cache / (split + '-sequence.npy'))[rows]
            macro = float(np.mean([error[groups == s].mean() for s in np.unique(groups)]))
            assert abs(macro - reported[group]['sequence_mean_mm']) < 1e-4
            checks[group] = dict(macro_mm=macro, p95_mm=float(np.percentile(error, 95)),
                                 replay_max_abs=replay_max)
        checks['max_parameter_difference_from_old'] = max(float((cp['model'][k]-old['model'][k]).abs().max())
                                                           for k in cp['model'])
        checks['training_membership'] = dict(rows=len(ids), sequences=len(np.unique(seq[ids])),
            startup_rows=int((tick[ids] < 16).sum()), later_rows=int((tick[ids] >= 16).sum()),
            original_seen_overlap=int(np.isin(ids, original_ids).sum()))
        if arm == 'original':
            assert checks['max_parameter_difference_from_old'] == 0
        result['arms'][arm] = checks
    # Distances use the model's teacher RMS/clamp. Full obs and a physically named
    # subset are reported separately: reference wrist XYZ error/orientation error,
    # measured world wrist linear/angular velocity, and five fingertip contacts.
    # Native obs layout: first15 root feedback, contact[252:257], repeated at+721.
    subsets = dict(full1442=np.arange(1442), wrist_feedback_contact40=np.r_[
        np.arange(15), np.arange(252, 257), np.arange(721, 736), np.arange(973, 978)])
    train = np.load(cache / 'train-full.npy', mmap_mode='r')
    expanded_ids = np.load(run / 'expanded/train-cache-indices.npy')
    with torch.no_grad():
        for name, dims in subsets.items():
            mu, sd = init['xmean'][dims].cuda(), init['xscale'][dims].cuda()
            banks = {key: ((torch.tensor(np.array(train[np.load(run / key / 'train-cache-indices.npy')][:, dims]),
                                        device='cuda') - mu) / sd).clamp(-5, 5) for key in arms}
            panel = {}
            for group in ('interpolation', 'unseen'):
                entry = member[group]
                raw = np.load(cache / (entry['cache_split'] + '-full.npy'), mmap_mode='r')[entry['cache_indices']][:, dims]
                x = ((torch.tensor(raw, device='cuda') - mu) / sd).clamp(-5, 5)
                minima = {}
                for key, bank in banks.items():
                    values = []
                    for batch in x.split(128):
                        best = torch.full((len(batch),), float('inf'), device='cuda')
                        for block in bank.split(8192):
                            best = torch.minimum(best, torch.cdist(batch, block).min(1).values)
                        values.append(best.cpu().numpy() / np.sqrt(len(dims)))
                    minima[key] = np.concatenate(values)
                panel[group] = {}
                for key in arms:
                    ratio = minima[key] / np.maximum(minima['original'], 1e-8)
                    panel[group][key] = dict(median=float(np.median(minima[key])),
                        median_ratio=float(np.median(ratio)), fraction_improved_10pct=float((ratio < .9).mean()))
            result['coverage'][name] = panel
    result['status'] = 'PASS'
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
