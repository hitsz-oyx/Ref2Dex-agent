#!/usr/bin/env python3
"""Single/DDP mixed training with common point supervision and a shared backbone."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
sys.path.insert(0, str(TASK.parent/'cm-pointflow-effect-pretrain/src'))
from goal_point_wam.point_supervision import PointSupervisedWAM
from goal_point_wam.video_data import VideoPointWindows, video_collate
from goal_point_wam.data import GoalWindows, goal_collate
from oakink_wm.multisource import MixedWindows, mixed_indices
from train_three_domain import digest, atomic_json, gpu_snapshot, move_batch, evaluate
from train_egodex_auxiliary import goal_sensitivity
from video_validation_panel import balanced_video_panel


@torch.no_grad()
def evaluate_video(model, dataset, device, batch_size, panel):
    model.eval()
    total, zero, count, actor, actor_count = 0., 0., 0, 0., 0
    sessions = {}; offset = 0
    for cpu in DataLoader(Subset(dataset, [r['index'] for r in panel]), batch_size=batch_size, collate_fn=video_collate):
        batch = move_batch(cpu, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(batch)
        valid = batch['point_target_valid']
        error = torch.linalg.vector_norm(pred['world_points']-batch['point_target'], dim=-1)
        reference = torch.linalg.vector_norm(batch['query_xyz'][:, None]-batch['point_target'], dim=-1)
        total += float(error[valid].sum())
        zero += float(reference[valid].sum())
        count += int(valid.sum())
        hv = batch['trajectory_valid']
        he = torch.linalg.vector_norm(pred['trajectory']-batch['trajectory_target'], dim=-1)
        actor += float(he[hv].sum())
        actor_count += int(hv.sum())
        for row in range(len(error)):
            session = panel[offset+row]['session']
            values = sessions.setdefault(session, dict(error=0., stationary=0., count=0, windows=0))
            values['error'] += float(error[row][valid[row]].sum())
            values['stationary'] += float(reference[row][valid[row]].sum())
            values['count'] += int(valid[row].sum()); values['windows'] += 1
        offset += len(error)
    per_session = {s:dict(point_epe_mm=v['error']/max(v['count'],1)*1000,
                         stationary_point_epe_mm=v['stationary']/max(v['count'],1)*1000,
                         valid_points=v['count'],windows=v['windows']) for s,v in sessions.items()}
    return dict(point_epe_mm=total/max(count, 1)*1000, stationary_point_epe_mm=zero/max(count, 1)*1000,
                valid_points=count, actor_epe_mm=actor/max(actor_count, 1)*1000, valid_hand_points=actor_count,
                sessions=per_session, session_macro_point_epe_mm=sum(v['point_epe_mm'] for v in per_session.values())/len(per_session))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'stats', 'video-data', 'init-wam', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='engineering only;4 updates, no checkpoint')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--main-batch', type=int, default=16)
    p.add_argument('--video-batch', type=int, default=4)
    p.add_argument('--seed', type=int, default=287)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--artifact-cap-bytes', type=int, default=1_500_000_000)
    p.add_argument('--validation-indices', type=Path)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    a = p.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda', local_rank)
    if rank == 0:
        if a.output.exists():
            raise FileExistsError(a.output)
        a.output.mkdir(parents=True)
    if world > 1:
        dist.barrier()
    steps = 4 if a.smoke else a.steps
    if steps < 4 or min(a.main_batch, a.video_batch, a.seconds, a.artifact_cap_bytes,
                         a.eval_every, a.checkpoint_every) < 1:
        raise ValueError('positive bounded run required')
    video = VideoPointWindows(a.video_data, 'train', engineering=a.smoke)
    # Scientific training cannot silently use an engineering-only corpus or one session.
    if not a.smoke:
        held = VideoPointWindows(a.video_data, 'val')
        if a.validation_indices is None:
            raise ValueError('scientific mixed fit requires the frozen main-three panel')
    else:
        held = None
    video_panel = balanced_video_panel(held.records) if held is not None else []
    base = MixedWindows(a.data, 'train')
    native = GoalWindows(base, 'object_points', 32)
    native_held, native_panel = None, None
    if not a.smoke:
        val_base = MixedWindows(a.data, 'val')
        native_held = GoalWindows(val_base, 'object_points', 32)
        panel_meta = json.loads((a.validation_indices.parent/'panel.json').read_text())
        if (panel_meta['validation_indices_sha256'] != digest(a.validation_indices)
                or panel_meta['data_manifest_sha256'] != digest(a.data/'processed/manifest.json')
                or panel_meta['stats_sha256'] != digest(a.stats)):
            raise ValueError('frozen main panel identity mismatch')
        all_panel = np.load(a.validation_indices)
        source_ids = np.searchsorted(val_base.offsets, all_panel, side='right')-1
        native_panel = np.concatenate([all_panel[source_ids == s][:64] for s in range(3)])
    parent = torch.load(a.init_wam, map_location='cpu', weights_only=False)
    if parent['identity']['stats_sha256'] != digest(a.stats):
        raise ValueError('parent normalization mismatch')
    config = dict(parent['config'], rigid_native_head=True, point_loss_scale_m=.05, point_query_chunk=256)
    model = PointSupervisedWAM(json.loads(a.stats.read_text()), config).to(device)
    transfer = model.transfer_wam(parent)
    parent_step = parent['step']
    del parent
    identities = {str(f.resolve()): digest(f) for f in (
        a.stats, a.init_wam, a.video_data/'manifest.json', a.data/'processed/manifest.json',
        Path(__file__), TASK/'src/goal_point_wam/model.py', TASK/'src/goal_point_wam/point_supervision.py',
        TASK/'src/goal_point_wam/video_data.py', Path(__file__).with_name('video_validation_panel.py'))}
    for record in video.meta['records']:
        path = a.video_data/record['path']
        if digest(path) != record['sha256']:
            raise ValueError('video source identity drift')
        identities[str(path.resolve())] = record['sha256']
    # Immutable parent weights/arrays are checked on admission and array load.
    # Rehash only small control files during fitting, avoiding repeated NAS
    # scans of every clip and the full parent checkpoint every ten updates.
    controls = {path: sha for path, sha in identities.items()
                if Path(path).suffix != '.pt' and Path(path).suffix != '.npz'}
    native_indices = mixed_indices(base, steps*a.main_batch, a.seed+1+1000*rank).astype('int32')
    native_loader = iter(DataLoader(Subset(native, native_indices.tolist()), batch_size=a.main_batch, collate_fn=goal_collate))
    generator = torch.Generator().manual_seed(a.seed+1+1000*rank)
    video_indices = torch.randint(len(video), (steps*a.video_batch,), generator=generator).tolist()
    video_loader = iter(DataLoader(Subset(video, video_indices), batch_size=a.video_batch, collate_fn=video_collate))
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    np.save(a.output/f'rank{rank}_native_indices.npy', native_indices)
    np.save(a.output/f'rank{rank}_video_indices.npy', np.asarray(video_indices, dtype='int32'))
    manifest = dict(git_commit=__import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    run_id=a.output.name, input_identity=identities, model_seed=a.seed,
                    world_size=world, main_global_batch=world*a.main_batch, video_global_batch=world*a.video_batch,
                    recipe='three native point-supervised updates then one video point-supervised update',
                    optimizer_reset=True, checkpoint_kind='goal-point-wam.point-supervised.v1',
                    parent_step=parent_step, engineering_only=a.smoke, ranks=[],video_validation_panel=video_panel)
    ranks = [dict(rank=rank, pid=os.getpid())]
    if world > 1:
        ranks = [None]*world
        dist.all_gather_object(ranks, dict(rank=rank, pid=os.getpid()))
    manifest['ranks'] = ranks
    if rank == 0:
        atomic_json(a.output/'input_manifest.json', manifest)
    started = time.time()
    def validation(step):
        # Every rank enters/exits the same barriers. Only rank0 evaluates a
        # bare model; metric presence must never decide collective scheduling.
        if world > 1:
            dist.barrier()
        report = None
        if rank == 0 and held is not None:
            report = dict(step=step, video=evaluate_video(model, held, device, a.video_batch, video_panel),
                          main=evaluate(model, native_held, native_panel, min(a.main_batch, 16), 0, device),
                          main_goal=goal_sensitivity(model, native_held, native_panel, device))
            with (a.output/'validation.jsonl').open('a') as f:
                f.write(json.dumps(report, allow_nan=False)+'\n')
        if world > 1:
            dist.barrier()
        model.train()
        return report

    def save_checkpoint(step):
        rng = dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), numpy_rng=np.random.get_state())
        states = [rng]
        if world > 1:
            states = [None]*world
            dist.all_gather_object(states, rng)
        if rank == 0:
            part = a.output/'latest.pt.part'
            torch.save(dict(checkpoint_kind='goal-point-wam.point-supervised.v1', model=model.state_dict(),
                       optimizer=optimizer.state_dict(), distributed_rng=states, config=config,
                       identity=manifest, step=step, optimizer_resumable=True), part)
            if sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file()) > a.artifact_cap_bytes:
                raise ValueError('atomic checkpoint exceeds declared output budget')
            part.replace(a.output/'latest.pt')
        if world > 1:
            dist.barrier()

    initial_validation = validation(0)
    held_before = initial_validation['video'] if initial_validation else None
    stop = [False]
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    video_shared_gradient = 0.
    log = []
    model.train()
    for step in range(1, steps+1):
        # Three native batches and one video batch; smoke exercises both twice.
        is_video = (step % 2 == 0) if a.smoke else (step % 4 == 0)
        batch = move_batch(next(video_loader if is_video else native_loader), device)
        if a.smoke and is_video:
            # Engineering intervention proves the VIDEO WORLD loss itself
            # reaches the shared backbone, not merely its native hand loss.
            batch['trajectory_valid'] = torch.zeros_like(batch['trajectory_valid'])
        optimizer.zero_grad(set_to_none=True)
        tick = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = wrapped(batch)
        loss, terms = model.loss(prediction, batch)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad):
            raise ValueError('nonfinite gradient')
        if is_video:
            backbone_grad = sum(float(p.grad.float().square().sum()) for p in model.scene.backbone.parameters() if p.grad is not None)**.5
            video_shared_gradient = max(video_shared_gradient, backbone_grad)
        else:
            backbone_grad = None
        optimizer.step()
        row = dict(step=step, video=is_video, loss=float(loss), gradient_norm=float(grad),
                   video_backbone_gradient_norm=backbone_grad, seconds=time.time()-tick,
                   elapsed_seconds=time.time()-started, terms={k:float(v) for k,v in terms.items()})
        log.append(row)
        if rank == 0:
            row['gpu'] = gpu_snapshot(device)
            row['eta_seconds'] = row['elapsed_seconds']/step*(steps-step)
            print(json.dumps(row), flush=True)
            atomic_json(a.output/'progress.json', dict(status='RUNNING', **row))
            with (a.output/'train.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        if not a.smoke and step % a.eval_every == 0:
            validation(step)
        if not a.smoke and step % a.checkpoint_every == 0:
            save_checkpoint(step)
        local_stop = torch.tensor(int(stop[0] or time.time()-started >= a.seconds), device=device)
        if rank == 0 and (step == steps or step % 10 == 0):
            if any(digest(Path(path)) != sha for path,sha in controls.items()):
                local_stop.fill_(1)
            total_bytes = sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())
            if total_bytes >= a.artifact_cap_bytes:
                local_stop.fill_(1)
        if world > 1:
            dist.all_reduce(local_stop, op=dist.ReduceOp.MAX)
        if local_stop.item():
            break
    h = hashlib.sha256()
    for parameter in model.parameters():
        h.update(parameter.detach().float().cpu().numpy().tobytes())
    hashes = [None]*world
    if world > 1:
        dist.all_gather_object(hashes, h.hexdigest())
    else:
        hashes[0] = h.hexdigest()
    if len(set(hashes)) != 1 or video_shared_gradient <= 0:
        raise ValueError('shared backbone did not train, or DDP ranks diverged')
    final_validation = validation(step)
    if not a.smoke:
        save_checkpoint(step)
    if rank == 0:
        held_after = final_validation['video'] if final_validation else None
        result = dict(status='COMPLETED' if step == steps else 'STOPPED', engineering_only=a.smoke,
            run_id=a.output.name, steps=step, world_size=world, elapsed_seconds=time.time()-started,
            final_rank_parameter_sha256=hashes, video_shared_backbone_gradient_norm=video_shared_gradient,
            transfer=transfer, input_identity=identities, history=log, gpu=gpu_snapshot(device),
            held_session_before=held_before, held_session_after=held_after,
            limitations='Engineering wiring only; video actor loss masked to isolate point-world gradient' if a.smoke else 'Exploratory mixed fit, up to4 clock-spaced clips per held session; no matched control')
        if not a.smoke and step == steps:
            # Final is the same inode as rolling latest, including optimizer/RNG.
            os.link(a.output/'latest.pt', a.output/'final.pt')
        atomic_json(a.output/'result.json', result)
        atomic_json(a.output/'progress.json', dict(status=result['status'], step=step,
                    elapsed_seconds=result['elapsed_seconds'], gpu=result['gpu']))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
