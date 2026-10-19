"""Bounded real-TRAIN WAM timing and 1 mm output gate; never update a policy."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seconds', type=int, default=240)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--single-check', action='store_true')
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.repeats <= 10 or not 30 <= a.seconds <= 600:
        raise ValueError('fresh bounded inference audit required')
    if a.checkpoint and not a.checkpoint.is_file(): raise ValueError('existing actor checkpoint required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), REF2DEX_DEEP_OPT='all',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('inference audit budget')))
    signal.alarm(a.seconds)
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v2.wam_v4_runtime import V4Runtime
    from wam_rl_policy_v2.batched_geometry import workspace_valid
    from wam_rl_policy_v2.inspire_shadow import expanded_model, CURRENT_DIM, tau_world
    spec = importlib.util.spec_from_file_location('inspire_full_bc_inputs', TASK/'tools/run/inspire_full_bc.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.set_num_threads(2); torch.manual_seed(246)
    meta = json.loads((a.cache/'manifest.json').read_text())
    runtime = V4Runtime(meta['wam'], meta['records'], a.cache,
                        dense_manifest=a.cache/'surface-manifest.json', allow_geometry_replacement=True)
    # Eight real H4 windows per object; cover all 50 TRAIN object identities.
    if a.fixture and a.fixture.exists():
        fixture=torch.load(a.fixture,map_location='cpu',weights_only=False)
        if fixture['wam']!=runtime.conditioning_identity: raise ValueError('fixture WAM identity mismatch')
        cpu,current,objects=fixture['batch'],fixture['current'],fixture['objects']
    else:
        rows=[]; currents=[]
        train_x=np.load(a.cache/'train-x.npy',mmap_mode='r')
        for obj in sorted({r['object'] for r in meta['sequences'] if r['split']=='train'}):
            row = next(r for r in meta['sequences'] if r['split']=='train' and r['object']==obj)
            item = module.RecordSamples([row], runtime.geometry, 8)[0]
            if not item['batches']: raise ValueError('empty TRAIN history')
            batch = item['batches'][0]
            keep = workspace_valid(batch, runtime.config)
            rows.append({k:v[keep] for k,v in batch.items()})
            currents.append(torch.from_numpy(train_x[row['lo']+3:row['lo']+3+len(keep)].copy())[keep])
        cpu = {k:torch.cat([b[k] for b in rows]) for k in rows[0]}
        current=torch.cat(currents);objects=len(rows)
        if a.fixture:
            a.fixture.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(wam=runtime.conditioning_identity,batch=cpu,current=current,objects=objects),a.fixture)
    if len(cpu['xyz']) < 256: raise ValueError('insufficient real in-workspace fixtures')
    actor = None
    if a.checkpoint:
        packet = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        state = packet['model']
        if 'actor.net.0.weight' in state:
            state = {k[6:]:v for k,v in state.items() if k.startswith('actor.')}
        actor = expanded_model(True).cuda().eval().requires_grad_(False)
        actor.load_state_dict(state)
    report = dict(status='RUNNING', fixture='8 measured H4 windows per TRAIN object',
                  objects=objects, samples=len(cpu['xyz']), gpu=a.gpu, torch=torch.__version__,
                  python=sys.executable, wam=runtime.conditioning_identity,
                  optimizer_steps=0, gate_m=.001, rows=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): a.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    references = []
    def run(mode, size):
        output = []
        for lo in range(0, len(cpu['xyz']), size):
            batch = {k:v[lo:lo+size].cuda() for k,v in cpu.items()}
            fn = runtime.predict if mode=='full_fp32' else runtime.predict_hand
            options=dict(shared_tf32=True) if mode=='hand_shared_tf32' else {}
            output.append(fn(batch,**options)['trajectory'].float().cpu())
        return torch.cat(output)
    for mode, size in [('full_fp32',128), ('hand_fp32',256), ('hand_shared_tf32',128), ('hand_shared_tf32',256)]:
        torch.backends.cuda.matmul.allow_tf32 = mode=='hand_tf32'
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            warm = run(mode, size)
            if not references: references.append(warm)
            durations=[]; torch.cuda.reset_peak_memory_stats()
            for _ in range(a.repeats):
                torch.cuda.synchronize(); start=time.perf_counter()
                predicted=run(mode,size); torch.cuda.synchronize()
                durations.append(time.perf_counter()-start)
            error=float((predicted-references[0]).abs().max())
            repeat_error=float((predicted-warm).abs().max())
            single_error=None
            if a.single_check and mode=='hand_shared_tf32':
                singles=[]
                for ix in range(0,len(cpu['xyz']),8):
                    b={k:v[ix:ix+1].cuda() for k,v in cpu.items()}
                    singles.append(runtime.predict_hand(b,shared_tf32=True)['trajectory'].float().cpu())
                single_error=float((torch.cat(singles)-predicted[::8]).abs().max())
            action_error=None
            if actor is not None:
                # Same measured current for both trajectories: isolate tau drift.
                x=actor.x_mean.expand(len(predicted),-1).clone();x[:,:CURRENT_DIM]=current.cuda();x[:,-1]=1
                rot=Rotation.from_quat(current[:,287:291].numpy()).as_matrix()
                x[:,CURRENT_DIM:-1]=torch.as_tensor(tau_world(references[0].numpy(),rot).reshape(len(predicted),264),device='cuda')
                old=actor(x)
                x[:,CURRENT_DIM:-1]=torch.as_tensor(tau_world(predicted.numpy(),rot).reshape(len(predicted),264),device='cuda')
                action_error=float((actor(x)-old).abs().max())
            row=dict(mode=mode,batch=size,seconds=durations,median_s=float(np.median(durations)),
                     max_trajectory_error_m=error,max_repeat_error_m=repeat_error,
                     measured_state_action_delta=action_error,max_batch_single_error_m=single_error,
                     peak_mib=torch.cuda.max_memory_allocated()/2**20,
                     passed=bool(max(error,repeat_error,single_error or 0)<=.001 and torch.isfinite(predicted).all()))
            report['rows'].append(row);save();print(json.dumps(row),flush=True)
    report['status']='COMPLETED';save()


if __name__=='__main__': main()
