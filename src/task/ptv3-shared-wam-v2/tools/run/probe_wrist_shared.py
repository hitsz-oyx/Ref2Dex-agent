"""Prepare, extract and train bounded matched811D wrist +/-shared probes."""
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
sys.path.insert(0, str(TASK / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'extract', 'merge', 'train', 'benchmark'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int)
    p.add_argument('--mode', choices=['none', 'aligned', 'shuffled'], default='none')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--model-seed', type=int, default=196)
    p.add_argument('--draw-seed', type=int, default=197)
    p.add_argument('--shuffle-seed', type=int, default=199)
    p.add_argument('--seconds', type=int, default=1800)
    a = p.parse_args()
    a.output = a.output.resolve()
    distributed_train = int(os.environ.get('WORLD_SIZE', '1')) > 1 and a.stage in ('train', 'benchmark')
    if a.output.exists() and (not distributed_train or int(os.environ.get('RANK', '0')) == 0):
        raise FileExistsError(a.output)
    if not (ROOT / 'outputs/ptv3-shared-wam-v2') in a.output.parents:
        raise ValueError('Fresh Task output required')
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError('20GiB disk reserve')
    rank = int(os.environ.get('RANK', a.rank))
    world = int(os.environ.get('WORLD_SIZE', a.world_size))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if a.stage in ('extract', 'train', 'benchmark') and (world == 1 or a.stage == 'extract'):
        if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('Assigned GPU occupied')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', REF2DEX_RUN_ROOT=str(ROOT))
    scratch = ROOT / 'tmp' / a.output.name
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/shared-wam-v2-extensions'))
    import numpy as np
    import torch
    from ptv3_shared_wam_v2.wrist_shared_probe import (
        FEATURE_SCHEMA, MODEL_SCHEMA, WristSharedMLP, pool_shared, phase_draw, condition_rows, write_json)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Probe time budget')))
    signal.alarm(a.seconds)
    if world > 1 and a.stage in ('train', 'benchmark'):
        torch.cuda.set_device(local)
        torch.distributed.init_process_group('nccl')
        if rank == 0:
            a.output.mkdir(parents=True)
        torch.distributed.barrier()
    else:
        a.output.mkdir(parents=True)
    started = time.monotonic()
    git_commit = os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    status = dict(status='RUNNING', stage=a.stage, pid=os.getpid(), rank=rank, world_size=world,
                  gpu=a.gpu, git_commit=git_commit, training=a.stage == 'train', budget_seconds=a.seconds)
    if rank == 0:
        write_json(a.output/'process.json', status)
    def gpu_status():
        return subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                       '--format=csv,noheader'], text=True).strip()
    try:
        if a.stage == 'prepare':
            source = ROOT/'outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1'
            corpus = json.loads((source/'manifest.json').read_text())
            assert corpus['capture_current_observation'] and corpus['observation_contract']['live_future_invariance_passed']
            assert json.loads((source/corpus['native_observation_audit']).read_text())['status'] == 'PASS'
            original = ROOT/'outputs/ptv3-shared-wam-v2/shared-actor-shared-20261015-r1/encoder_workspace_admission.json'
            snapshot = json.loads(original.read_text())
            panels_path = ROOT/'outputs/ptv3-shared-wam-v2/shared-execution-gap-20261015-r1/panels.json'
            panels = json.loads(panels_path.read_text())
            lookup = {r['sequence']: r for r in corpus['records']}
            all_data, panel_indices = {}, {}
            for split in ('train', 'val'):
                xs, ys, ticks_all, seqids, names, keys = [], [], [], [], [], []
                for seq, record in enumerate(snapshot[split]['records']):
                    ticks = np.asarray(record['admitted_ticks'], np.int64)
                    if not len(ticks):
                        continue
                    name = record['sequence']; item = lookup[name]
                    assert item['split'] == split
                    with np.load(source/item['path']/'trajectory.npz') as z:
                        assert np.array_equal(z['frame_tick'], np.arange(len(z['q'])))
                        q, dq, hand, obj, vel = (z[k][ticks] for k in ('q','dq','hand','obj','velocity'))
                        center = obj[:, :3, 3]
                        local_hand = np.einsum('npj,njk->npk', hand-center[:,None], obj[:,:3,:3])
                        state = np.concatenate((q,dq,local_hand.reshape(-1,33),obj[:,:3,:].reshape(-1,12),vel),-1)
                        goal = z['reference_obj'][ticks+24,:3,3]-center
                        xs.append(np.concatenate((state,goal,z['native_current_obs'][ticks]),-1).astype(np.float32))
                        ys.append(z['applied'][ticks,:6].astype(np.float32))
                    ticks_all.extend(ticks.tolist());seqids.extend([seq]*len(ticks));names.append(name)
                    keys.extend((name,int(t)) for t in ticks)
                data = dict(x=np.concatenate(xs),y=np.concatenate(ys),tick=np.asarray(ticks_all),sequence=np.asarray(seqids))
                assert data['x'].shape == (snapshot[split]['retained_windows'],811)
                assert all(np.isfinite(v).all() for v in data.values())
                np.savez(a.output/(split+'.npz'), **data)
                write_json(a.output/(split+'-keys.json'),keys)
                identities = {tuple(k):i for i,k in enumerate(keys)}
                for name, rows in panels.items():
                    if name.startswith('TRAIN') == (split=='train'):
                        panel_indices[name] = dict(split=split,indices=[identities[(r['sequence'],r['tick'])] for r in rows])
                all_data[split] = data
            assert [len(all_data[s]['x']) for s in ('train','val')] == [93680,10376]
            d=all_data['train']
            stats=dict(xmean=d['x'].mean(0),xstd=d['x'].std(0).clip(1e-3),ymean=d['y'].mean(0),ystd=d['y'].std(0).clip(.01),
                       controller_scale=np.asarray(corpus['native_controller']['scale'][:6],np.float32))
            np.savez(a.output/'raw-stats.npz',**stats)
            write_json(a.output/'snapshot.json',snapshot)
            write_json(a.output/'panels.json',panel_indices)
            write_json(a.output/'manifest.json',dict(schema=FEATURE_SCHEMA,source=str(source),membership=str(original),
                windows={s:len(v['x']) for s,v in all_data.items()},raw_width=811,shared_width=2304,git_commit=git_commit))
        elif a.stage == 'extract':
            from ptv3_shared_wam_v2.robot_data import V2RLRobotWindows,robot_collate
            from ptv3_shared_wam_v2.model import SharedWAMV2
            from ptv3_shared_wam.loading import to_device
            from torch.utils.data import DataLoader,Subset
            meta=json.loads((a.cache/'manifest.json').read_text());snap=json.loads((a.cache/'snapshot.json').read_text())
            packet=torch.load(a.checkpoint,map_location='cpu',weights_only=False);config=packet['config']
            model=SharedWAMV2(packet['stats'],config)
            weights={k[len('forecaster.'):]:v for k,v in packet['model'].items() if k.startswith('forecaster.')}
            model.load_state_dict(weights,strict=True);del packet
            model.cuda().eval();progress={}
            for split in ('train','val'):
                ds=V2RLRobotWindows(Path(meta['source']),split,config['object_points'],config['action_representation'])
                seq={r['sequence']:i for i,r in enumerate(ds.records)}
                keys=json.loads((a.cache/(split+'-keys.json')).read_text())
                valid_keys=set(ds.keys);ds.keys=[(t,seq[name]) for name,t in keys]
                assert set(ds.keys).issubset(valid_keys) and len(ds)==meta['windows'][split]
                indices=np.array_split(np.arange(len(ds)),world)[rank]
                if a.limit:indices=indices[:a.limit]
                loader=DataLoader(Subset(ds,indices.tolist()),batch_size=a.batch,num_workers=a.workers,
                    collate_fn=robot_collate,pin_memory=True,shuffle=False)
                z=np.lib.format.open_memmap(a.output/(split+'-z.npy'),mode='w+',dtype=np.float32,shape=(len(indices),2304))
                cursor=0
                with torch.inference_mode():
                    for b in loader:
                        b=to_device(b,'cuda');tokens,valid,scene=model.encode(b)
                        value=pool_shared(tokens,valid,scene['tokens'].shape[1]).cpu().numpy()
                        z[cursor:cursor+len(value)]=value;cursor+=len(value)
                        if cursor==len(value) or cursor//(a.batch*20)!=(cursor-len(value))//(a.batch*20):
                            progress[split]=dict(rows=cursor,total=len(indices),elapsed_s=time.monotonic()-started,gpu=gpu_status())
                            write_json(a.output/'progress.json',progress);print(json.dumps(progress[split]),flush=True)
                assert cursor==len(indices);z.flush();np.save(a.output/(split+'-indices.npy'),indices)
            write_json(a.output/'manifest.json',dict(schema=FEATURE_SCHEMA,status='COMPLETED',checkpoint=str(a.checkpoint.resolve()),
                cache=str(a.cache.resolve()),rank=rank,world_size=world,git_commit=git_commit,limit=a.limit,
                pooling='separate masked mean scene/hand/goal, concatenated2304D',precision='FP32 TF32 as historical offline evaluation'))
        elif a.stage == 'merge':
            meta=json.loads((a.cache/'manifest.json').read_text());shards=[]
            for rank_id in range(world):
                s=a.cache/('features-rank'+str(rank_id));m=json.loads((s/'manifest.json').read_text())
                assert m['status']=='COMPLETED' and m['rank']==rank_id and m['world_size']==world and m['limit'] is None
                if shards:assert m['checkpoint']==shards[0][1]['checkpoint'] and m['git_commit']==shards[0][1]['git_commit']
                shards.append((s,m))
            for split in ('train','val'):
                z=np.lib.format.open_memmap(a.output/(split+'-z.npy'),mode='w+',dtype=np.float32,shape=(meta['windows'][split],2304))
                seen=np.zeros(len(z),bool)
                for s,_ in shards:
                    ix=np.load(s/(split+'-indices.npy'));values=np.load(s/(split+'-z.npy'),mmap_mode='r')
                    assert not seen[ix].any() and values.shape==(len(ix),2304) and np.isfinite(values).all()
                    z[ix]=values;seen[ix]=True
                assert seen.all();z.flush()
                if split=='train':mean=z.mean(0,dtype=np.float64);std=z.std(0,dtype=np.float64).clip(1e-3)
            np.savez(a.output/'shared-stats.npz',zmean=mean.astype(np.float32),zstd=std.astype(np.float32))
            write_json(a.output/'manifest.json',dict(schema=FEATURE_SCHEMA,status='COMPLETED',raw_cache=str(a.cache.resolve()),
                checkpoint=shards[0][1]['checkpoint'],pooling=shards[0][1]['pooling'],normalization='TRAIN only',git_commit=git_commit))
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP
            torch.manual_seed(a.model_seed)
            net=WristSharedMLP().cuda()
            initial={k:v.cpu().numpy().copy() for k,v in net.state_dict().items()}
            optimizer=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=a.weight_decay)
            if world>1:net=DDP(net,device_ids=[local])
            raw=net.module if world>1 else net
            if a.stage=='benchmark':
                torch.manual_seed(17)
                x=torch.randn(4096,811,device='cuda');z=torch.randn(4096,2304,device='cuda');y=torch.randn(4096,6,device='cuda')
                data={'train':dict(x=x,z=z,y=y)}
                draw=lambda: np.arange(512)
                panels={}
            else:
                fm=json.loads((a.cache/'manifest.json').read_text());raw_cache=Path(fm['raw_cache'])
                stats=dict(np.load(raw_cache/'raw-stats.npz'));stats.update(dict(np.load(a.cache/'shared-stats.npz')))
                panels=json.loads((raw_cache/'panels.json').read_text());data={}
                for split in ('train','val'):
                    d=dict(np.load(raw_cache/(split+'.npz')));z=np.load(a.cache/(split+'-z.npy'))
                    z=(z-stats['zmean'])/stats['zstd'];z=condition_rows(z,a.mode,a.shuffle_seed+(split=='val'))
                    data[split]={k:torch.as_tensor(v,device='cuda') for k,v in dict(x=(d['x']-stats['xmean'])/stats['xstd'],
                        z=z,y=(d['y']-stats['ymean'])/stats['ystd']).items()}
                    data[split]['tick']=d['tick']
                    if split=='train':draw=phase_draw(d['tick'],d['sequence'],512,np.random.default_rng(a.draw_seed))
                factor=torch.as_tensor(stats['ystd']*stats['controller_scale'],device='cuda')
            if rank==0:
                np.savez(a.output/'initial-model.npz',**initial)
                write_json(a.output/'manifest.json',dict(schema=MODEL_SCHEMA,mode=a.mode,checkpoint=str(a.cache) if a.cache else None,
                    model_seed=a.model_seed,draw_seed=a.draw_seed,shuffle_seed=a.shuffle_seed,global_batch=512,
                    lr=a.lr,weight_decay=a.weight_decay,updates=a.updates,world_size=world,git_commit=git_commit,
                    selection='fixed final endpoint; no DEV selection',status='RUNNING',engineering_only=a.stage=='benchmark'))
            @torch.no_grad()
            def evaluate(full=False):
                raw.eval();result={};sets=dict(panels)
                if full:sets.update({s+'_all':dict(split=s,indices=list(range(len(d['x'])))) for s,d in data.items()})
                for name,entry in sets.items():
                    d=data[entry['split']];ix=np.asarray(entry['indices']);chunks=[]
                    for offset in range(0,len(ix),1024):
                        ids=ix[offset:offset+1024];chunks.append(((raw(d['x'][ids],d['z'][ids])-d['y'][ids])*factor).cpu().numpy())
                    delta=np.concatenate(chunks);xyz=np.linalg.norm(delta[:,:3],axis=-1)*1000;ticks=d['tick'][ix]
                    result[name]={}
                    for phase,mask in [('all',np.ones(len(ix),bool)),('startup',ticks<16),('later',ticks>=16)]:
                        if mask.any():result[name][phase]=dict(count=int(mask.sum()),xyz_mean_mm=float(xyz[mask].mean()),
                            xyz_p95_mm=float(np.percentile(xyz[mask],95)),euler_mae_rad=float(abs(delta[mask,3:]).mean()))
                    if full:np.save(a.output/(name+'-XYZ.npy'),xyz)
                raw.train();return result
            if rank==0 and a.stage=='train':write_json(a.output/'initial-metrics.json',evaluate())
            warm_start=None
            for step in range(1,a.updates+1):
                ix=draw()
                if rank==0 and step==1:np.save(a.output/'first-draw.npy',ix)
                part=np.array_split(ix,world)[rank];d=data['train']
                prediction=net(d['x'][part],d['z'][part]);loss=(prediction-d['y'][part]).abs().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                if step==10:torch.cuda.synchronize();warm_start=time.monotonic()
                if step==1 or step%1000==0 or step==a.updates:
                    if world>1:torch.distributed.barrier()
                    if rank==0:
                        metrics=evaluate(full=step==a.updates) if a.stage=='train' else {}
                        progress=dict(step=step,loss=float(loss),elapsed_s=time.monotonic()-started,panels=metrics,gpu=gpu_status())
                        write_json(a.output/'progress.json',progress)
                        with (a.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(progress)+'\n')
                        print(json.dumps(dict(step=step,loss=float(loss),elapsed_s=progress['elapsed_s'],DEV=metrics.get('DEV_fixed256'))),flush=True)
                        if a.stage=='train' and step==a.updates:
                            torch.save(dict(schema=MODEL_SCHEMA,step=step,mode=a.mode,model=raw.state_dict(),stats=stats,
                                feature_checkpoint=fm['checkpoint'],feature_pooling=fm['pooling'],optimizer=optimizer.state_dict()),a.output/'final.pt')
                        if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('Disk reserve')
                    if world>1:torch.distributed.barrier()
            torch.cuda.synchronize()
            if rank==0:
                progress.update(status='COMPLETED',steady_updates_per_second=(a.updates-10)/(time.monotonic()-warm_start) if warm_start else None)
                write_json(a.output/'result.json',progress)
                manifest=json.loads((a.output/'manifest.json').read_text());manifest['status']='COMPLETED';write_json(a.output/'manifest.json',manifest)
            if world>1:torch.distributed.destroy_process_group()
        status.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
    except BaseException as error:
        status.update(status='FAILED',error=repr(error));raise
    finally:
        if rank==0 or a.stage=='extract':write_json(a.output/'process.json',status)
        signal.alarm(0)


if __name__=='__main__':
    main()
