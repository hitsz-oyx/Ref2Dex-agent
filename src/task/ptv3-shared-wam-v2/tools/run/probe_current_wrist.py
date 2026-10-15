"""Matched wrist-only MLPs: state/goal versus additional captured current delta0."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[5]


def write(path, value):
    tmp = path.with_suffix('.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['state_goal', 'plus_current'], required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=6000)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v2' not in out.parents:
        raise ValueError('Fresh Task output required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(196)
    torch.backends.cuda.matmul.allow_tf32 = True
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('Disk reserve')
    out.mkdir(parents=True)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('900s probe budget')))
    signal.alarm(900)
    source = ROOT/'outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1'
    corpus = json.loads((source/'manifest.json').read_text())
    contract = corpus['observation_contract']
    assert corpus['capture_current_observation'] and contract['dimension'] == 721
    assert contract['reference_delta'] == 0 and contract['future_reference_inputs_saved'] is False
    assert contract['live_future_invariance_passed'] is True
    audit = json.loads((source/corpus['native_observation_audit']).read_text())
    assert audit['status'] == 'PASS' and audit['sequences'] == 660
    snapshot = json.loads((ROOT/'outputs/ptv3-shared-wam-v2/shared-actor-shared-20261015-r1/encoder_workspace_admission.json').read_text())
    lookup = {r['sequence']: r for r in corpus['records']}
    panels = json.loads((ROOT/'outputs/ptv3-shared-wam-v2/shared-execution-gap-20261015-r1/panels.json').read_text())
    data, identities, pools = {}, {}, [[], []]
    for split in ('train', 'val'):
        xs, ys, ticks_all, names_all = [], [], [], []
        offset = 0
        for record in snapshot[split]['records']:
            ticks = np.asarray(record['admitted_ticks'], dtype=np.int64)
            if not len(ticks):
                continue
            name = record['sequence']
            item = lookup[name]
            assert item['split'] == split
            with np.load(source/item['path']/'trajectory.npz') as z:
                assert np.array_equal(z['frame_tick'], np.arange(item['frames']))
                q, dq, hand, obj, velocity = (z[k][ticks] for k in ('q', 'dq', 'hand', 'obj', 'velocity'))
                center = obj[:, :3, 3]
                local = np.einsum('npj,njk->npk', hand-center[:, None], obj[:, :3, :3])
                state = np.concatenate((q, dq, local.reshape(-1, 33), obj[:, :3, :].reshape(-1, 12), velocity), -1)
                goal = z['reference_obj'][ticks+24, :3, 3]-center
                extra = z['native_current_obs'][ticks]
                x = np.concatenate((state, goal, extra), -1).astype(np.float32)
                y = z['applied'][ticks, :6].astype(np.float32)
            assert x.shape == (len(ticks), 811) and np.isfinite(x).all() and np.isfinite(y).all()
            xs.append(x); ys.append(y); ticks_all.append(ticks); names_all.extend([name]*len(ticks))
            if split == 'train':
                for phase, mask in enumerate((ticks < 16, ticks >= 16)):
                    if mask.any():
                        pools[phase].append(np.flatnonzero(mask)+offset)
            offset += len(ticks)
        x, y, tick = np.concatenate(xs), np.concatenate(ys), np.concatenate(ticks_all)
        identities[split] = {(name, int(t)): i for i, (name, t) in enumerate(zip(names_all, tick))}
        data[split] = dict(x=x, y=y, tick=tick)
    assert [len(data[s]['y']) for s in ('train', 'val')] == [93680, 10376]
    xmean = data['train']['x'].mean(0)
    xstd = data['train']['x'].std(0).clip(1e-3)
    ymean = data['train']['y'].mean(0)
    ystd = data['train']['y'].std(0).clip(.01)
    for d in data.values():
        x = (d['x']-xmean)/xstd
        if a.arm == 'state_goal':
            x[:, 90:] = 0
        d['x'] = torch.as_tensor(x, device='cuda')
        d['target'] = torch.as_tensor((d['y']-ymean)/ystd, device='cuda')
    model = nn.Sequential(nn.Linear(811, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
                          nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 6)).cuda()
    nn.init.zeros_(model[-1].weight); nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scale = torch.as_tensor(corpus['native_controller']['scale'][:6], device='cuda')
    ym, ys = (torch.as_tensor(v, device='cuda') for v in (ymean, ystd))
    panel_ids = {}
    for panel, rows in panels.items():
        split = 'train' if panel.startswith('TRAIN') else 'val'
        panel_ids[panel] = (split, np.array([identities[split][(r['sequence'], r['tick'])] for r in rows]))

    @torch.no_grad()
    def evaluate(full=False):
        model.eval()
        result = {}
        sets = dict(panel_ids)
        if full:
            sets.update({s+'_all': (s, np.arange(len(d['y']))) for s, d in data.items()})
        for panel, (split, ix) in sets.items():
            d = data[split]
            delta = torch.cat([(model(d['x'][part])-d['target'][part])*ys*scale for part in np.array_split(ix, max(1, (len(ix)+1023)//1024))]).cpu().numpy()
            xyz = np.linalg.norm(delta[:, :3], axis=-1)*1000
            masks = dict(all=np.ones(len(ix), bool), startup=d['tick'][ix]<16, later=d['tick'][ix]>=16)
            result[panel] = {k: dict(count=int(mask.sum()), xyz_mean_mm=float(xyz[mask].mean()),
                xyz_p95_mm=float(np.percentile(xyz[mask], 95)), euler_mae_rad=float(np.abs(delta[mask, 3:]).mean()))
                for k, mask in masks.items() if mask.any()}
        model.train()
        return result

    manifest = dict(experiment_id='P-20261015-current-wrist-mlp', arm=a.arm, physical_gpu=a.gpu,
        pid=os.getpid(), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        widths=[811, 1024, 1024, 1024, 512, 6], seeds=[196, 197], updates=a.updates, batch=512,
        learning_rate=5e-4, loss='TRAIN standardized first native wrist6 L1',
        input='state87 + existing goal24 delta3; extra721 zero-masked in state_goal',
        source=str(source), membership='shared-actor-shared-20261015-r1/encoder_workspace_admission.json',
        no_future_hand_reference=True, selection='fixed final endpoint; no DEV selection', status='RUNNING')
    write(out/'manifest.json', manifest)
    initial = evaluate()
    write(out/'initial.json', initial)
    rng = np.random.default_rng(197)
    # Sequence-uniform within each half, matching the original half-startup policy.
    draw_tables = []
    for phase in pools:
        table = np.full((len(phase), max(map(len, phase))), -1, np.int64)
        for i, rows in enumerate(phase): table[i, :len(rows)] = rows
        draw_tables.append((table, np.asarray(list(map(len, phase)))))
    first_draw = None
    for step in range(1, a.updates+1):
        selected = []
        for table, lengths in draw_tables:
            seq = rng.integers(len(table), size=256)
            selected.append(table[seq, (rng.random(256)*lengths[seq]).astype(np.int64)])
        ix = np.concatenate(selected)
        if first_draw is None:
            first_draw = ix.tolist(); write(out/'first-draw.json', first_draw)
        pred = model(data['train']['x'][ix])
        loss = (pred-data['train']['target'][ix]).abs().mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 1000 == 0 or step == a.updates:
            metrics = evaluate(full=step==a.updates)
            progress = dict(step=step, elapsed_s=time.monotonic()-started, loss=float(loss), panels=metrics)
            write(out/'progress.json', progress)
            with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(progress)+'\n')
            torch.save(dict(model=model.state_dict(), step=step, xmean=xmean, xstd=xstd, ymean=ymean, ystd=ystd,
                arm=a.arm, optimizer=optimizer.state_dict()), out/f'step{step:05d}.pt')
            print(json.dumps(dict(step=step, elapsed_s=progress['elapsed_s'], TRAIN=metrics['TRAIN_balanced256']['all'], DEV=metrics['DEV_fixed256']['all'])), flush=True)
            if shutil.disk_usage(ROOT).free < 20*2**30: raise RuntimeError('Disk reserve')
    write(out/'result.json', dict(status='COMPLETED', **progress))
    signal.alarm(0)


if __name__ == '__main__':
    main()
