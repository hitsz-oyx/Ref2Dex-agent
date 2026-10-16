"""Full-token frozen-WAM query-head ablations; no pooled-feature substitution."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('REF2DEX_RUN_ROOT', TASK.parents[2])).resolve()
sys.path.insert(0, str(TASK/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('extract', 'train'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--share-owned-pid', type=int, help='Explicitly colocate with this same-experiment worker only')
    p.add_argument('--width', type=int, choices=(128, 512), default=128)
    p.add_argument('--direct-goal', action='store_true')
    p.add_argument('--bypass', action='store_true')
    p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--seconds', type=int, default=2400)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v2' not in out.parents:
        raise ValueError('Fresh Task output required')
    if shutil.disk_usage(ROOT).free < (80 if a.stage == 'extract' else 20)*2**30:
        raise RuntimeError('Disk reserve')
    occupied=subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        if a.share_owned_pid is None or set(occupied.splitlines())!={str(a.share_owned_pid)}:
            raise RuntimeError('Assigned GPU occupied by an unapproved process')
        owner=Path('/proc')/str(a.share_owned_pid)
        command=(owner/'cmdline').read_bytes().split(b'\0')
        if owner.stat().st_uid!=os.getuid() or not any(b'probe_actor_capacity.py' in v for v in command) or b'train' not in command:
            raise RuntimeError('Colocation requires our capacity training worker')
        if b'--output' not in command or Path(command[command.index(b'--output')+1].decode()).parent!=out.parent:
            raise RuntimeError('Colocation must remain within this experiment')
    scratch = ROOT/'tmp'/out.name
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      TMPDIR=str(scratch), REF2DEX_RUN_ROOT=str(ROOT))
    import numpy as np
    import torch
    from ptv3_shared_wam_v2.capacity_actor import CapacityActor
    from ptv3_shared_wam_v2.wrist_shared_probe import phase_draw, write_json
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Probe deadline')))
    signal.alarm(a.seconds)
    out.mkdir(parents=True)
    started = time.monotonic()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TASK, text=True).strip()
    status = dict(status='RUNNING', stage=a.stage, pid=os.getpid(), gpu=a.gpu, git_commit=commit,
                  max_seconds=a.seconds)
    write_json(out/'process.json', status)
    try:
        if a.stage == 'extract':
            from ptv3_shared_wam_v2.robot_data import V2RLRobotWindows, robot_collate
            from ptv3_shared_wam_v2.model import SharedWAMV2
            from ptv3_shared_wam.loading import to_device
            from torch.utils.data import DataLoader
            raw_meta = json.loads((a.cache/'manifest.json').read_text())
            packet = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
            model = SharedWAMV2(packet['stats'], packet['config'])
            model.load_state_dict({k[len('forecaster.'):]:v for k,v in packet['model'].items() if k.startswith('forecaster.')}, strict=True)
            config = packet['config']
            max_tokens = config['scene_tokens'] + 22 + config['object_points']
            model.cuda().eval().requires_grad_(False)
            actor_stats = {k[len('retargeter.'):]:v.numpy() for k,v in packet['model'].items()
                           if k.startswith('retargeter.') and k.endswith(('_mean', '_scale'))}
            del packet
            for split in ('train', 'val'):
                ds = V2RLRobotWindows(Path(raw_meta['source']), split, config['object_points'], config['action_representation'])
                lookup = {r['sequence']:i for i,r in enumerate(ds.records)}
                keys = json.loads((a.cache/(split+'-keys.json')).read_text())
                original = set(ds.keys)
                ds.keys = [(int(t), lookup[name]) for name,t in keys]
                assert set(ds.keys).issubset(original) and len(ds) == raw_meta['windows'][split]
                tokens = np.lib.format.open_memmap(out/(split+'-tokens.npy'), mode='w+', dtype=np.float16,
                                                 shape=(len(ds), max_tokens, config['shared_width']))
                valid = np.lib.format.open_memmap(out/(split+'-valid.npy'), mode='w+', dtype=bool, shape=(len(ds), max_tokens))
                targets = np.lib.format.open_memmap(out/(split+'-target.npy'), mode='w+', dtype=np.float32, shape=(len(ds),8,15))
                cursor = 0
                for b in DataLoader(ds, batch_size=a.batch, collate_fn=robot_collate, num_workers=2, pin_memory=True):
                    b = to_device(b, 'cuda')
                    with torch.no_grad():
                        z, mask, _ = model.encode(b)
                    n, count = z.shape[:2]
                    assert count <= max_tokens and torch.isfinite(z).all()
                    tokens[cursor:cursor+n] = 0
                    valid[cursor:cursor+n] = False
                    tokens[cursor:cursor+n,:count] = z.half().cpu().numpy()
                    valid[cursor:cursor+n,:count] = mask.cpu().numpy()
                    targets[cursor:cursor+n] = b['action_target'].cpu().numpy()
                    if cursor == 0:
                        # Preserve FP32 sample for independent cache-quantization/head replay.
                        np.savez(out/(split+'-fp32-replay.npz'), tokens=z[:16].cpu().numpy(),
                                 valid=mask[:16].cpu().numpy(), indices=np.arange(min(n,16)))
                    cursor += n
                    if cursor == n or cursor % (20*a.batch) == 0 or cursor == len(ds):
                        write_json(out/'progress.json', dict(split=split, rows=cursor, total=len(ds),
                            elapsed_s=time.monotonic()-started, token_count=count,
                            gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()))
                assert cursor == len(ds)
                tokens.flush();valid.flush();targets.flush()
                if split == 'train':
                    actor_stats['action_mean'] = targets.mean(0)
                    actor_stats['action_scale'] = targets.std(0).clip(.01)
            raw = np.load(a.cache/'train.npz')['x']
            actor_stats.update(state_mean=raw[:,:87].mean(0),state_scale=raw[:,:87].std(0).clip(1e-3),
                current_mean=raw[:,90:].mean(0),current_scale=raw[:,90:].std(0).clip(1e-3),
                goal_mean=raw[:,87:90].mean(0),goal_scale=raw[:,87:90].std(0).clip(1e-3))
            np.savez(out/'stats.npz', **actor_stats)
            write_json(out/'manifest.json', dict(status='COMPLETED', schema='ref2dex.query-capacity-cache.v1',
                raw_cache=str(a.cache.resolve()), checkpoint=str(a.checkpoint.resolve()), git_commit=commit,
                windows=raw_meta['windows'], max_tokens=max_tokens, precision='FP32 TF32 frozen encoder; FP16 stored full tokens',
                config=config, normalization='TRAIN-only state/current/goal/action'))
        else:
            meta = json.loads((a.cache/'manifest.json').read_text())
            assert meta['status'] == 'COMPLETED'
            raw_cache = Path(meta['raw_cache'])
            stats = dict(np.load(a.cache/'stats.npz'))
            data = {}
            for split in ('train','val'):
                d = dict(np.load(raw_cache/(split+'.npz')))
                d['x'] = torch.as_tensor(d['x'], device='cuda')
                d.update(tokens=np.load(a.cache/(split+'-tokens.npy'), mmap_mode='r'),
                         valid=np.load(a.cache/(split+'-valid.npy'), mmap_mode='r'),
                         target=torch.as_tensor(np.load(a.cache/(split+'-target.npy')),device='cuda'))
                data[split] = d
            torch.manual_seed(221)
            model = CapacityActor(a.width, a.direct_goal, a.bypass).cuda()
            with torch.no_grad():
                for key,value in stats.items():
                    if hasattr(model,key):getattr(model,key).copy_(torch.as_tensor(value,device='cuda'))
            np.savez(out/'initial-model.npz', **{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
            draw = phase_draw(data['train']['tick'], data['train']['sequence'], a.batch, np.random.default_rng(222))
            panels = json.loads((raw_cache/'panels.json').read_text())
            def predict(d, ix):
                z = torch.as_tensor(np.array(d['tokens'][ix],copy=True), device='cuda').float()
                mask = torch.as_tensor(np.array(d['valid'][ix],copy=True),device='cuda')
                x = d['x'][ix]
                return model(x[:,:87], z, mask, x[:,90:], x[:,87:90])
            @torch.no_grad()
            def evaluate():
                model.eval();result={}
                for name,entry in panels.items():
                    d=data[entry['split']];ids=np.asarray(entry['indices']);pred=[]
                    for start in range(0,len(ids),a.batch):
                        ix=ids[start:start+a.batch]
                        with torch.autocast('cuda',dtype=torch.bfloat16):value=predict(d,ix)
                        pred.append(value.float())
                    predicted=torch.cat(pred);target=d['target'][ids]
                    xyz=((predicted[:,0,:3]-target[:,0,:3]).norm(dim=-1)*1000).cpu().numpy()
                    result[name]=dict(xyz_mean_mm=float(xyz.mean()),xyz_p95_mm=float(np.percentile(xyz,95)),
                        action_l1=float(((predicted-target)/model.action_scale).abs().mean()))
                    for phase,mask in [('startup',d['tick'][ids]<16),('later',d['tick'][ids]>=16)]:
                        if mask.any():result[name][phase]=dict(count=int(mask.sum()),xyz_mean_mm=float(xyz[mask].mean()))
                model.train();return result
            manifest=dict(status='RUNNING',width=a.width,direct_goal=a.direct_goal,bypass=a.bypass,
                share_owned_pid=a.share_owned_pid,
                params=sum(p.numel() for p in model.parameters()),active_params=sum(p.numel() for n,p in model.named_parameters()
                    if not (n.startswith('bypass.') and not a.bypass)),batch=a.batch,updates=a.updates,lr=5e-4,weight_decay=1e-4,
                model_seed=221,draw_seed=222,git_commit=commit,cache=str(a.cache.resolve()),
                frozen_wam=True,labels='all8x15 q_pd geometry-lift',precision='BF16 actor, full FP16 cached tokens',
                selection='fixed final endpoint; no DEV selection')
            write_json(out/'manifest.json',manifest)
            write_json(out/'initial-metrics.json',evaluate())
            for step in range(1,a.updates+1):
                ix=draw();d=data['train']
                if step==1:np.save(out/'first-draw.npy',ix)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):predicted=predict(d,ix)
                loss=((predicted.float()-d['target'][ix])/model.action_scale).abs().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if step==1 or step%1000==0 or step==a.updates:
                    torch.cuda.synchronize();elapsed=time.monotonic()-started
                    row=dict(step=step,loss=float(loss),elapsed_s=elapsed,eta_s=elapsed/step*(a.updates-step),panels=evaluate())
                    write_json(out/'progress.json',row)
                    with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True)
            # FP16 cache admission: compare FP32 source tokens with exactly their quantized copies.
            replay=np.load(a.cache/'val-fp32-replay.npz');ids=replay['indices'];x=data['val']['x'][ids]
            z=torch.as_tensor(replay['tokens'],device='cuda');mask=torch.as_tensor(replay['valid'],device='cuda')
            model.eval()
            with torch.no_grad():
                fp=model(x[:,:87],z,mask,x[:,90:],x[:,87:90])
                hp=model(x[:,:87],z.half().float(),mask,x[:,90:],x[:,87:90])
            quant_mm=float(((fp[:,0,:3]-hp[:,0,:3]).norm(dim=-1)*1000).max())
            write_json(out/'cache-replay.json',dict(status='PASS' if quant_mm<.1 else 'FAIL',max_xyz_difference_mm=quant_mm,rows=len(ids)))
            if quant_mm>=.1:raise ValueError('FP16 cache changed wrist output by >=0.1mm')
            torch.save(dict(schema='ref2dex.query-capacity-actor.v1',step=a.updates,model=model.state_dict(),
                config=manifest,optimizer=optimizer.state_dict(),feature_checkpoint=meta['checkpoint']),out/'final.pt')
            manifest['status']='COMPLETED';write_json(out/'manifest.json',manifest)
            row['status']='COMPLETED';write_json(out/'result.json',row)
        status.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
    except BaseException as error:
        status.update(status='FAILED',error=repr(error));raise
    finally:
        write_json(out/'process.json',status);signal.alarm(0)


if __name__=='__main__':main()
