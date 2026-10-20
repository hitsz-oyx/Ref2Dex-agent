"""Actually train task10: random Current310, then learned-goal Scheduled575."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('REF2DEX_RUN_ROOT', TASK.parents[2])).resolve()
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/run')]


def write(path, value):
    part = path.with_suffix(path.suffix+'.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('current', 'scheduled'), required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--batch', type=int, default=2048, help='global batch, not per rank')
    p.add_argument('--seed', type=int, default=241)
    p.add_argument('--seconds', type=int, default=7200)
    p.add_argument('--physical-gpus', type=int, nargs='+', required=True)
    p.add_argument('--engineering-smoke', action='store_true')
    a = p.parse_args()
    if not 1 <= a.steps <= 50000 or not 3 <= a.batch <= 8192 or not 30 <= a.seconds <= 7200:
        raise ValueError('bounded BC updates/batch/deadline required')
    if (a.stage == 'scheduled') != bool(a.parent):
        raise ValueError('only Scheduled575 loads this run\'s learned Current310 parent')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from train_current_bc import load_resident_array
    from wam_rl_policy_v3.bc_mlp import TaskShadowMLP
    from wam_rl_policy_v3.bc_sampling import sample_rows, startup_rows
    from wam_rl_policy_v3.shadow_task_contract import conditioning, CURRENT_SCHEMA, SCHEDULED_SCHEMA
    from wam_rl_policy_v3.task_bc_training import global_mean_loss

    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 3 or len(a.physical_gpus) != world:
        raise ValueError('this reproduction requires exactly three DDP ranks')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=600))
    try:
        out = a.output.resolve(); data_root = a.data.resolve()
        if ROOT/'outputs/wam-rl-policy-v3' not in out.parents:
            raise ValueError('task-local fresh output required')
        if rank == 0:
            if out.exists(): raise ValueError('fresh BC output required')
            out.mkdir(parents=True)
        dist.barrier()
        torch.set_num_threads(2); torch.manual_seed(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        cache = json.loads((data_root/'manifest.json').read_text())
        with_tau = a.stage == 'scheduled'
        labels = cache.get('schema') == 'ref2dex.shadow-task-labels.v1'
        if (labels and with_tau) or (not labels and cache.get('schema') != 'ref2dex.shadow-task-current-cache.v1'):
            raise ValueError('Current310 task labels or Scheduled575 cache required')
        if (cache.get('status') != 'COMPLETED' or cache.get('conditioning') != conditioning(not labels)):
            raise ValueError('completed same-TaskController575 cache required')
        if bool(cache.get('engineering_fixture')) != a.engineering_smoke or (a.engineering_smoke and a.steps>10):
            raise ValueError('engineering fixtures require an explicit <=10-update smoke')
        started = time.monotonic(); data = {}
        for split in ('train', 'dev'):
            if labels:
                robot=load_resident_array(Path(cache['base_cache'])/(split+'-x.npy'),device)
                task=load_resident_array(data_root/(split+'-task.npy'),device)
                data[split]=dict(x=torch.cat((robot,task),-1),y=load_resident_array(data_root/(split+'-y.npy'),device))
                del robot,task
            else:
                data[split] = {k: load_resident_array(data_root/(split+'-'+k+'.npy'), device)
                               for k in ('x', 'y')}
            if data[split]['x'].shape[1] != (310 if labels else 575): raise ValueError('real task10 input required')
        model = TaskShadowMLP(with_tau).to(device)
        stats = dict(np.load(data_root/'stats.npz'))
        with torch.no_grad():
            for key in ('x_mean','x_scale','y_mean','y_scale'):
                value = stats[key][:model.input_dim] if key.startswith('x_') else stats[key]
                getattr(model,key).copy_(torch.as_tensor(value,device=device))
        inherited = False
        if a.parent:
            parent = torch.load(a.parent.resolve(), map_location='cpu', weights_only=False)
            if parent.get('config',{}).get('conditioning') != conditioning(False):
                raise ValueError('Current310 parent must share the exact task contract')
            inherited = bool(parent.get('task_columns_trained'))
            if not inherited: raise ValueError('parent task10 columns have not actually learned')
            model.initialize_task_current(parent)
        initial_goal = model.net[0].weight[:,300:310].detach().clone()
        initial_tau = model.net[0].weight[:,310:].detach().clone() if with_tau else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        rng = np.random.default_rng(a.seed+1)
        ticks = np.load(data_root/'train-tick.npy', mmap_mode='r')
        early = startup_rows(ticks, 60)
        panel = torch.as_tensor(np.load(data_root/'dev-panel.npy'), device=device)
        schema = SCHEDULED_SCHEMA if with_tau else CURRENT_SCHEMA
        commit = os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        manifest = dict(status='RUNNING',schema=schema,stage=a.stage,
            experiment_id='P-20261019-inspire-task-contract',run_id=out.parent.name+'/'+out.name,
            git_commit=commit,branch='main',python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
            physical_gpus=a.physical_gpus,world_size=world,global_batch=a.batch,
            local_batches=[len(x) for x in np.array_split(np.arange(a.batch),world)],
            gradient_reduction='DDP mean of (world*local/global)*local mean L1',
            source_cache=str(data_root),data=cache['splits'],conditioning=conditioning(with_tau),
            wam=None if labels else cache['identity']['wam'],dense=cache['dense'] if labels else cache['identity']['dense'],input_dim=model.input_dim,
            base_cache=cache['base_cache'] if labels else cache['identity']['base'],
            source=cache['source'],engineering_smoke=a.engineering_smoke,
            parent=str(a.parent.resolve()) if a.parent else None,
            initialization='newly learned Current310' if a.parent else 'fresh random including task10',
            sampler='half startup tick<8; half all TRAIN',seed=a.seed,steps=a.steps,
            lr=3e-4,weight_decay=1e-5,hidden_dims=list(model.hidden_dims),
            preparation_s=time.monotonic()-started,stop_seconds=a.seconds,optimizer='AdamW',
            normalization='original TRAIN current300/action; explicit task10/tau scales')
        if rank == 0: write(out/'manifest.json',manifest)
        curve=[]; best=float('inf'); first_goal_gradient=None; begin=time.monotonic()

        @torch.no_grad()
        def evaluate():
            model.eval(); errors=[]
            for lo in range(0,len(panel),2048):
                ids=panel[lo:lo+2048]
                errors.append((model(data['dev']['x'][ids,:model.input_dim])-data['dev']['y'][ids]).abs())
            error=torch.cat(errors); model.train()
            return dict(command_mae=float(error.mean()),root_mae=float(error[:,:6].mean()),
                        finger_mae=float(error[:,6:].mean()))

        def save(step, metrics, latest=True):
            nonlocal best
            movement=float((model.net[0].weight[:,300:310]-initial_goal).abs().max())
            learning=dict(max_task_weight_change=movement,first_nonzero_task_gradient=first_goal_gradient,
                max_tau_weight_change=float((model.net[0].weight[:,310:]-initial_tau).abs().max()) if with_tau else None)
            packet=dict(schema=schema,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
                config=manifest,source_cache=str(data_root),task_learning=learning,
                task_columns_trained=bool(inherited or (step>0 and movement>0 and first_goal_gradient is not None)),
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
            names=['latest.pt'] if latest else []
            if metrics['command_mae']<best:best=metrics['command_mae'];names.append('best.pt')
            for name in names:
                torch.save(packet,out/(name+'.part'));(out/(name+'.part')).replace(out/name)
            return learning

        if rank==0:
            baseline=evaluate();write(out/'initial-dev.json',baseline);save(0,baseline)
        dist.barrier()
        draw=torch.empty(a.batch,dtype=torch.long,device=device)
        completed=0;stop_reason=None
        for step in range(1,a.steps+1):
            if rank==0:draw.copy_(torch.from_numpy(sample_rows(rng,len(ticks),early,a.batch)))
            dist.broadcast(draw,src=0)
            ids=torch.tensor_split(draw,world)[rank]
            prediction=ddp(data['train']['x'][ids,:model.input_dim])
            loss=global_mean_loss(prediction,data['train']['y'][ids],model.y_scale,a.batch,world)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite task BC loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if rank==0 and first_goal_gradient is None:
                value=float(model.net[0].weight.grad[:,300:310].abs().max())
                if value>0:first_goal_gradient=dict(step=step,max_abs=value)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();completed=step
            if step==1 or step%1000==0 or step==a.steps:
                # All ranks synchronize around rank0-only DEV and checkpoint I/O.
                reported_loss=loss.detach().clone();dist.all_reduce(reported_loss);reported_loss.div_(world)
                dist.barrier()
                if rank==0:
                    metrics=evaluate();learning=save(step,metrics)
                    elapsed=time.monotonic()-begin
                    row=dict(step=step,loss=float(reported_loss),dev=metrics,elapsed_s=elapsed,
                        eta_s=elapsed/step*(a.steps-step),task_learning=learning,
                        gpu=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,a.physical_gpus)),
                            '--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
                    curve.append(row);write(out/'curve.json',curve);write(out/'progress.json',row)
                    print(json.dumps(row),flush=True)
                    if (out/'STOP').exists() or (out.parent/'STOP').exists():stop_reason='owned STOP'
                    elif shutil.disk_usage(ROOT).free<20*2**30:stop_reason='disk floor'
                    elif elapsed>a.seconds:stop_reason='wall-clock limit'
                flag=torch.tensor([int(stop_reason is not None) if rank==0 else 0],device=device)
                dist.broadcast(flag,src=0)
                if flag.item():break
        # Check actual parameter agreement, not just successful process exits.
        largest=0.
        for parameter in model.parameters():
            reference=parameter.detach().clone();dist.broadcast(reference,src=0)
            largest=max(largest,float((parameter-reference).abs().max()))
        disagreement=torch.tensor(largest,device=device);dist.all_reduce(disagreement,op=dist.ReduceOp.MAX)
        if rank==0:
            manifest.update(status='COMPLETED' if completed==a.steps else 'STOPPED',steps_completed=completed,
                stop_reason=stop_reason,elapsed_s=time.monotonic()-begin,best_dev_command_mae=best,
                maximum_rank_parameter_difference=float(disagreement))
            write(out/'manifest.json',manifest)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
