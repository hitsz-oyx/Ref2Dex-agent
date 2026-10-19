"""Compare cached Euclidean contact and an explicit normal-aware audit candidate."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v5.contact_audit import (
    AUDIT_VERSION, SOURCE_NAMES, future_exposure, normal_candidate, opposite_surface_proxy)
from ptv3_shared_wam_v5.contracts import LABEL_SEMANTICS
from ptv3_shared_wam_v5.storage import read_json, load_array
from audit_contact_prevalence import write

METRIC_NAMES = ('queries', 'invalid_normals', 'baseline_positive', 'candidate_positive',
                'changed_gt01', 'delta_sum', 'positive_backside', 'changed_backside',
                'thin_proxy_queries', 'changed_thin_proxy', 'thin_proxy_positive', 'suppressed_positive')


def choose_frames(eligible, positive, count, rng):
    """Half uniform, half contact-enriched, with unique frame/object pairs."""
    eligible = np.asarray(eligible)
    first = rng.choice(len(eligible), min((count+1)//2, len(eligible)), replace=False)
    chosen = {tuple(x): 'uniform' for x in eligible[first]}
    positive = np.asarray(positive)
    if len(positive):
        for i in rng.permutation(len(positive)):
            if len(chosen) >= count: break
            chosen.setdefault(tuple(positive[i]), 'contact_enriched')
    for i in rng.permutation(len(eligible)):
        if len(chosen) >= count: break
        chosen.setdefault(tuple(eligible[i]), 'uniform_fill')
    return [(int(f), int(o), kind) for (f, o), kind in chosen.items()]


def metrics(result, thin):
    valid = result['normal_valid']; base = result['baseline'].astype(np.float32)
    candidate = result['candidate'].astype(np.float32); delta = base-candidate
    changed = valid & (delta > .1); positive = valid & (base > .25)
    return dict(queries=int(valid.sum()), invalid_normals=int((~valid).sum()),
        baseline_positive=int(positive.sum()), candidate_positive=int((valid & (candidate > .25)).sum()),
        changed_gt01=int(changed.sum()), delta_sum=float(delta[valid].sum()),
        positive_backside=int((positive & (result['signed'] < -.001)).sum()),
        changed_backside=int((changed & (result['signed'] < -.001)).sum()),
        thin_proxy_queries=int((valid & thin).sum()), changed_thin_proxy=int((changed & thin).sum()),
        thin_proxy_positive=int((positive & thin).sum()),
        suppressed_positive=int((positive & (candidate <= .25)).sum()))


def render(output, examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for rank, item in enumerate(examples):
        stem = 'example-%02d-s%d' % (rank, item['source_id'])
        fields = {k: item[k] for k in ('queries', 'normals', 'vertices', 'baseline', 'candidate', 'thin_proxy')}
        np.savez_compressed(output/(stem+'.npz'), **fields)
        fig = plt.figure(figsize=(13, 4))
        q, h = item['queries'], item['vertices']
        center = q.mean(0); span = max(float(np.ptp(q, axis=0).max())*.6, .01)
        for i, (label, colors) in enumerate((('Distance only', item['baseline']),
                ('Normal candidate', item['candidate']), ('Suppression', item['baseline']-item['candidate']))):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            ax.scatter(*h.T, c='gray', s=.3, alpha=.12)
            art = ax.scatter(*q.T, c=colors, s=3, cmap='magma', vmin=0, vmax=1)
            for axis, value in zip(('x', 'y', 'z'), center):
                getattr(ax, 'set_'+axis+'lim')(value-span, value+span)
            ax.set_box_aspect((1, 1, 1)); ax.set_title(label)
            ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
        fig.colorbar(art, ax=fig.axes, shrink=.65)
        fig.suptitle('%s %s frame%d object%d: %s' % (SOURCE_NAMES[item['source_id']],
            item['key'], item['frame'], item['slot'], item['object']))
        fig.savefig(output/(stem+'.png'), dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'hands', 'contacts', 'geometry', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--git-commit', required=True)
    p.add_argument('--frames-per-source', type=int, default=36)
    p.add_argument('--sequences-per-source', type=int, default=8)
    p.add_argument('--seconds', type=int, default=900)
    a = p.parse_args()
    if not 1 <= a.frames_per_source <= 55 or not 1 <= a.sequences_per_source <= 16 or a.seconds <= 0:
        p.error('bounded panel/runtime required (at most495 source-balanced frame/object pairs)')
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); rng = np.random.default_rng(a.seed)
    plan = read_json(a.plan); registry = read_json(a.geometry/'manifest.json')
    registry = {r['key']: r for r in registry['objects']}; geometries = {}
    available = set(os.listdir(a.contacts)); jobs = defaultdict(list)
    for job in plan['jobs']:
        if 'train' in job['windows'] and job['key'] in available:
            jobs[job['source_id']].append(job)
    write(a.output/'run-manifest.json', dict(work_version=AUDIT_VERSION, git_commit=a.git_commit,
        run_id=a.output.name, pid=os.getpid(), python=sys.version, numpy=np.__version__, seed=a.seed,
        device='CPU offline geometry; no model inference', seconds=a.seconds, min_free_gib=20,
        max_output_mib=256, frames_per_source=a.frames_per_source, sequences_per_source=a.sequences_per_source,
        inputs={k: str(getattr(a, k).resolve()) for k in ('plan', 'hands', 'contacts', 'geometry')},
        candidate='d_eff=d+4*max(-dot(h-o,n)-0.001m,0); original 3mm/10mm/33mm confidence transfer',
        candidate_identity='normal-backside-penalty4-tolerance1mm.v1; NOT CEDex reproduction',
        normal_orientation='Original CAD normals, rotated to world. Outward winding not certified; no auto flip.',
        selection='TRAIN future-target frames only. Half uniform valid, half c>0.25 enriched; available-cache subset, not population prevalence.'))
    rows = []; examples = {}; source_frames = defaultdict(int)

    def check():
        if time.monotonic()-started > a.seconds: raise InterruptedError('deadline')
        if (a.output/'STOP').exists(): raise InterruptedError('STOP')
        if shutil.disk_usage(a.output).free < 20*2**30: raise InterruptedError('free space')

    status = 'COMPLETED'; reason = None
    try:
        for sid in sorted(jobs):
            candidates = [jobs[sid][i] for i in rng.permutation(len(jobs[sid]))]
            candidates = candidates[:a.sequences_per_source]
            for number, job in enumerate(candidates):
                check(); key = job['key']; folder = a.hands/'sequences'/key; cf = a.contacts/key
                meta = read_json(cf/'meta.json')
                if meta['label_semantics'] != LABEL_SEMANTICS: raise ValueError('contact label identity changed')
                c = load_array(cf/'contact.npy', mmap_mode='r'); cv = load_array(cf/'valid.npy', mmap_mode='r')
                exposure = future_exposure(load_array(a.plan.parent/job['windows']['train']), len(c), c.shape[1])
                eligible = np.argwhere(cv & (exposure > 0))
                if not len(eligible): continue
                # Only reduce candidate frames; no full sequence advanced-index copy.
                positive = np.array([(f, o) for f, o in eligible if (c[f, o] > .25).any()], dtype=np.int64)
                quota = int(np.ceil((a.frames_per_source-source_frames[sid])/(len(candidates)-number)))
                selections = choose_frames(eligible, positive, quota, rng)
                seq = {name: load_array(folder/(name+'.npy'), mmap_mode='r') for name in (
                    'joints', 'surface_offsets', 'surface_valid', 'surface_parts', 'object_poses')}
                for frame, slot, selection in selections:
                    check(); name = job['objects'][slot]; gkey = 's%d:%s' % (sid, name) if sid < 3 else name
                    if gkey not in geometries:
                        with load_array(a.geometry/registry[gkey]['path'], allow_pickle=False) as z:
                            geo = {k: z[k] for k in z.files}
                        geo['thin_proxy'] = opposite_surface_proxy(geo['surface'], geo['normals'])
                        geometries[gkey] = geo
                    geo = geometries[gkey]; pose = seq['object_poses'][frame, slot]
                    queries = geo['surface']@pose[:3, :3].T+pose[:3, 3]
                    normals = geo['normals']@pose[:3, :3].T
                    sides = seq['surface_valid'][frame]
                    vertices = (seq['surface_offsets'][frame].astype(np.float32)+seq['joints'][frame, :, None, 0])[sides].reshape(-1, 3)
                    parts = seq['surface_parts'][sides].reshape(-1)
                    result = normal_candidate(vertices, parts, queries, normals)
                    if not np.array_equal(result['baseline'], c[frame, slot]):
                        raise ValueError('baseline regeneration disagrees with cached teacher: '+key)
                    row = dict(key=key, source_id=sid, object=name, frame=frame, slot=slot, selection=selection,
                               **metrics(result, geo['thin_proxy']))
                    rows.append(row); source_frames[sid] += 1
                    previous = examples.get(sid)
                    if previous is None or row['changed_gt01'] > previous['changed_gt01']:
                        examples[sid] = dict(row, queries=queries, normals=normals, vertices=vertices,
                            baseline=result['baseline'].astype(np.float32), candidate=result['candidate'].astype(np.float32),
                            thin_proxy=geo['thin_proxy'])
                print(json.dumps(dict(source_id=sid, frames=source_frames[sid], elapsed_s=time.monotonic()-started)), flush=True)
    except InterruptedError as exc:
        status, reason = 'PARTIAL', str(exc)
    except Exception as exc:
        write(a.output/'summary.json', dict(status='FAILED', reason=str(exc), frames=len(rows))); raise
    sources = {}
    for sid in sorted(jobs):
        subset = [r for r in rows if r['source_id'] == sid]
        totals = {k: sum(r[k] for r in subset) for k in METRIC_NAMES}
        n = totals['queries']
        sources[str(sid)] = dict(name=SOURCE_NAMES[sid], frames=len(subset),
            sequences=len({r['key'] for r in subset}), objects=len({r['object'] for r in subset}), **totals,
            changed_fraction=totals['changed_gt01']/n if n else None,
            mean_suppression=totals['delta_sum']/n if n else None)
    expected = set(j['source_id'] for j in plan['jobs'])
    if set(source_frames) != expected or any(source_frames[s] < a.frames_per_source for s in expected):
        status = 'PARTIAL'; reason = reason or 'not enough available TRAIN frames for requested balanced panel'
    write(a.output/'frames.json', rows)
    selected = sorted(examples.values(), key=lambda x: -x['changed_gt01'])[:6]
    rendered = False
    try:
        check(); render(a.output, selected); rendered = True
    except InterruptedError as exc:
        status, reason = 'PARTIAL', str(exc)
    write(a.output/'summary.json', dict(status=status, reason=reason, frames=len(rows), sources=sources,
        elapsed_s=time.monotonic()-started, baseline_matches_cache=True, examples_rendered=rendered,
        limitation='An enriched, source-balanced diagnostic, not a prevalence estimate or proof of normal correctness. Concavity and true physical contact are not inferred. Same nearest discrete vertex metric for MANO and Shadow still has morphology-dependent sampling density.',
        decision='Review changes, thin/opposing-face proxies and visuals before changing teacher. Audit never changes caches, model, loss, C dimensions or PPO.'))
    write(a.output/'examples.json', [{k: r[k] for k in ('source_id', 'key', 'frame', 'slot', 'object', 'changed_gt01')} for r in selected])


if __name__ == '__main__': main()
