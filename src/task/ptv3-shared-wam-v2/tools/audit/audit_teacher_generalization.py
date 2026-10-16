"""Independently recompute probe memberships, normalizers and physical metrics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT / 'src/task/ptv3-shared-wam-v2'
ACTIVE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    if subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                               '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('assigned GPU occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2',
                      TMPDIR=str(ROOT / 'tmp'))
    sys.path.insert(0, str(TASK / 'src'))
    import numpy as np
    import torch
    from ptv3_shared_wam_v2.teacher_generalization import assert_support_disjoint
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    run = args.run.resolve()
    if (run / 'independent-audit.json').exists():
        raise FileExistsError('audit already recorded')
    manifest = json.loads((run / 'manifest.json').read_text())
    cache = Path(manifest['cache'])
    membership = json.loads((run / 'membership.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    if manifest['status'] != 'COMPLETED' or result['status'] != 'COMPLETED':
        raise ValueError('completed run required')
    predictions = np.load(run / 'predictions-final.npz')
    packet = torch.load(run / ('step%d.pt' % manifest['updates']), map_location='cpu', weights_only=False)
    layers = []
    for i in [0, 2, 4, 6, 8]:
        w = packet['model'][str(i) + '.weight']
        layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
        if i != 8:
            layers.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*layers).cuda().eval()
    model.load_state_dict(packet['model'], strict=True)
    xm, xs, ym, ys = [packet[k].cuda() for k in ['xmean', 'xscale', 'ymean', 'yscale']]
    replay_max = 0.
    groups = {}
    split_arrays = {}
    with torch.inference_mode():
        for split in ['train', 'val']:
            indices = predictions[split + '_cache_indices']
            x = np.load(cache / (split + '-full.npy'), mmap_mode='r')
            q = np.load(cache / (split + '-q.npy'), mmap_mode='r')
            pd = np.load(cache / (split + '-pd.npy'), mmap_mode='r')
            saved = predictions[split + '_pd']
            for start in range(0, len(indices), 512):
                ix = indices[start:start + 512]
                normalized = ((torch.tensor(x[ix], device='cuda') - xm) / xs).clamp(-5, 5)
                actual = (model(normalized) * ys + ym).cpu().numpy()
                actual[:, :6] += q[ix, :6]
                replay_max = max(replay_max, float(abs(actual - saved[start:start + len(ix)]).max()))
            split_arrays[split] = dict(
                sequence=np.load(cache / (split + '-sequence.npy')),
                tick=np.load(cache / (split + '-tick.npy')))
            for name, group in membership['groups'].items():
                if group['cache_split'] != split:
                    continue
                ix = np.asarray(group['cache_indices'], dtype=np.int64)
                local = np.searchsorted(indices, ix)
                assert np.array_equal(indices[local], ix)
                errors = np.sqrt(np.square(saved[local, :3] - pd[ix, :3]).sum(axis=1)) * 1000
                sequence = split_arrays[split]['sequence'][ix]
                means = [float(errors[sequence == s].mean()) for s in np.unique(sequence)]
                report = dict(count=len(ix), mean_mm=float(errors.mean()),
                              p95_mm=float(np.percentile(errors, 95)),
                              sequence_mean_mm=float(np.mean(means)))
                expected = result['final']['groups'][name]['all']
                assert report['count'] == expected['count']
                for metric in ['mean_mm', 'p95_mm', 'sequence_mean_mm']:
                    assert abs(report[metric] - expected[metric]) < .001, (name, metric)
                groups[name] = report
    assert replay_max < 2e-6, replay_max
    fit = np.array(membership['groups']['seen']['cache_indices'])
    held = np.array(membership['groups']['interpolation']['cache_indices'])
    assert not np.intersect1d(fit, held).size
    t, s = split_arrays['train']['tick'], split_arrays['train']['sequence']
    separations = [assert_support_disjoint(t[fit[s[fit] == seq]], t[held[s[held] == seq]])
                   for seq in np.unique(s[held])]
    train_names = {r['sequence'] for r in membership['sequences'] if r['split'] == 'train'}
    unseen_names = {r['sequence'] for r in membership['sequences'] if r['split'] == 'val'}
    assert not train_names & unseen_names
    target = np.load(cache / 'train-pd.npy')[fit][:, ACTIVE].copy()
    target[:, :6] -= np.load(cache / 'train-q.npy')[fit, :6]
    np.testing.assert_allclose(target.mean(0), packet['ymean'].numpy(), atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(target.std(0).clip(.001), packet['yscale'].numpy(), atol=2e-6, rtol=1e-5)
    # Verify the old student's architecture and random initialization were preserved.
    initial = torch.load(run / 'step0.pt', map_location='cpu', weights_only=False)
    old_path = ROOT / 'outputs/ptv3-shared-wam-v2/teacher-fit-generalization-20261015-r1/full_relative-cosine_tail/step0.pt'
    old = torch.load(old_path, map_location='cpu', weights_only=False)
    assert all(torch.equal(initial['model'][k], old['model'][k]) for k in initial['model'])
    report = dict(status='PASS', groups=groups, model_replay_max_abs=replay_max,
                  minimum_query_separation=min(separations),
                  target_stats_train_only=True, initialization_matches_parent_recipe=True,
                  sequence_disjoint=True, cache=str(cache))
    (run / 'independent-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
