"""Supervise bounded parallel full-corpus collection; freeze after660 receipts."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
RUNTIME = '/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'


def write(path,value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def recover_batches(out, batches, seed, capture_current=False, capture_force=False):
    """Reuse only complete, owned batch receipts; partial attempts stay intact."""
    recovered, completed, controller = set(), [], None
    for receipt_path in sorted((out/'batches').glob('*/manifest.json')):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('status') != 'COMPLETED':
            continue
        if capture_force and receipt.get('force_contract',{}).get('schema')!='ref2dex.inspire-measured-forces.v1':
            raise ValueError('resume receipt lacks same-execution measured force contract')
        if capture_current and (not receipt.get('capture_current_observation')
                or receipt.get('observation_contract',{}).get('dimension') != 721
                or not receipt.get('observation_contract',{}).get('live_future_invariance_passed')):
            raise ValueError('resume receipt lacks native current observation evidence')
        names = receipt['sequences']
        indices = [i for i, batch in enumerate(batches) if names == batch]
        if (len(indices) != 1 or receipt['seed'] != seed+indices[0]
                or receipt.get('schema') != 'ref2dex.rl660-batch.v1'
                or sorted(r['sequence'] for r in receipt['records']) != sorted(names)):
            raise ValueError('resume receipt differs from frozen batches/seed')
        index = indices[0]
        if index in recovered:
            raise ValueError('duplicate completed batch receipt')
        if controller is not None and receipt['native_controller'] != controller:
            raise ValueError('native controller changes across resume receipts')
        controller = receipt['native_controller']
        for record in receipt['records']:
            folder = (receipt_path.parent/record['path']).resolve()
            if out not in folder.parents or not all((folder/f).is_file() for f in ('trajectory.npz','canonical.npz')):
                raise ValueError('resume artifacts must exist inside corpus')
            completed.append(dict(record, path=str(folder.relative_to(out))))
        recovered.add(index)
    return recovered, completed, controller


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experiment-card',type=Path,required=True)
    p.add_argument('--gpus',default='4,5')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=120)
    p.add_argument('--seconds',type=int,default=10800)
    p.add_argument('--detach',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--capture-current-observation',action='store_true')
    p.add_argument('--capture-dof-force',action='store_true')
    p.add_argument('--allow-monica-sharing',action='store_true')
    p.add_argument('--split-from',type=Path)
    p.add_argument('--inherit-completed',type=Path,help='Reuse completed force batches from an owned partial corpus; collect missing sequences only')
    p.add_argument('--artifact-budget-bytes',type=int,default=4_000_000_000)
    a = p.parse_args()
    if a.capture_dof_force and not a.capture_current_observation:raise ValueError('force capture requires current measured states')
    if a.inherit_completed and (not a.capture_dof_force or a.resume):raise ValueError('inherited completion requires a fresh force collection')
    out = a.output.resolve()
    gpus = a.gpus.split(',')
    allowed = [ROOT/'outputs/ptv3-shared-wam']
    if a.capture_current_observation:
        allowed.append(ROOT/'outputs/ptv3-shared-wam-v2')
    if a.capture_dof_force:allowed.extend([ROOT/'outputs/wam-rl-policy-v2',ROOT/'outputs/wam-rl-policy-v3'])
    if ((out.exists() != a.resume) or not any(p in out.parents for p in allowed)
            or not 1 <= len(gpus) <= 6 or len(set(gpus)) != len(gpus)
            or not all(g.isdigit() for g in gpus) or not 1<=a.batch_size<=32
            or not a.experiment_card.is_file()):
        raise ValueError('fresh output or explicit resume, distinct GPUs and existing card required')
    inherited_split = None
    if a.split_from:
        parent = json.loads(a.split_from.read_text())
        if parent.get('status') != 'COMPLETED' or parent.get('schema') != 'ref2dex.rl660-corpus.v1':
            raise ValueError('frozen completed original sequence split required')
        inherited_split = {r['sequence']:r['split'] for r in parent['records']}
        if len(inherited_split) != 660 or set(inherited_split.values()) != {'train','val'}:
            raise ValueError('complete original 660-member split required')
    if a.capture_current_observation and inherited_split is None:
        raise ValueError('native current observation collection must preserve the original sequence split')
    scratch = ROOT/'tmp'/out.name
    previous = None
    if a.resume:
        previous = json.loads((out/'manifest.json').read_text())
        if (previous.get('schema') != 'ref2dex.rl660-corpus.v1'
                or previous['status'] != 'FAILED' or previous['seed'] != a.seed):
            raise ValueError('only failed owned corpus with unchanged seed can resume')
        if (previous.get('capture_current_observation',False) != a.capture_current_observation
                or previous.get('capture_dof_force',False)!=a.capture_dof_force
                or previous.get('inherited_split') != inherited_split):
            raise ValueError('resume must preserve input contract and split')
        attempt = len(previous.get('resume_history',[]))+1
        scratch = scratch/('resume%02d'%attempt)
    scratch.mkdir(parents=True,exist_ok=True)
    if a.detach:
        command = [sys.executable,str(Path(__file__).resolve()),'--output',str(out),
            '--experiment-card',str(a.experiment_card.resolve()),'--gpus',a.gpus,
            '--batch-size',str(a.batch_size),'--seed',str(a.seed),'--seconds',str(a.seconds),
            '--artifact-budget-bytes',str(a.artifact_budget_bytes)]
        if a.capture_current_observation:
            command.append('--capture-current-observation')
        if a.capture_dof_force:command.append('--capture-dof-force')
        if a.allow_monica_sharing:command.append('--allow-monica-sharing')
        if a.split_from:
            command.extend(['--split-from',str(a.split_from.resolve())])
        if a.inherit_completed:command.extend(['--inherit-completed',str(a.inherit_completed.resolve())])
        if a.resume:
            command.append('--resume')
        with (scratch/'supervisor.log').open('x') as log:
            process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1'))
        write(scratch/'launch.json',dict(pid=process.pid,command=command))
        print(json.dumps(dict(supervisor_pid=process.pid,scratch=str(scratch))),flush=True)
        return
    from collect_rl660_batch import check_gpu_occupancy
    allowed_occupants={gpu:set(map(int,check_gpu_occupancy(gpu,a.allow_monica_sharing).split())) for gpu in gpus}
    if shutil.disk_usage(ROOT).free < 20*1024**3:
        raise RuntimeError('less than20GiB filesystem headroom')
    source = ROOT/'data/processed_data/inspire_rl_object_dexplore'
    names = sorted(p.parent.name for p in source.glob('*/interaction_hand_inspire.pt'))
    if len(names)!=660:
        raise ValueError('frozen 660-member source required')
    if inherited_split is not None and set(inherited_split) != set(names):
        raise ValueError('new membership differs from inherited original split')
    out.mkdir(parents=True,exist_ok=a.resume)
    manifest = dict(schema='ref2dex.rl660-corpus.v1',status='RUNNING',training_allowed=False,
        engineering_only=False,num_sequences=660,fps=30,records=[],physical_gpus=gpus,
        experiment_card=str(a.experiment_card.resolve()),seed=a.seed,
        source_membership=str(source.resolve()),supervisor_pid=os.getpid(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        split_policy='complete sequence; stratified by object, 10% dev with at least one per object')
    manifest.update(capture_current_observation=a.capture_current_observation,
        capture_dof_force=a.capture_dof_force,allow_monica_sharing=a.allow_monica_sharing,
        task_goal_source='same_execution_teacher_object_rollout' if a.capture_dof_force else 'legacy',
        inherited_split=inherited_split,artifact_budget_bytes=a.artifact_budget_bytes,
        split_from=str(a.split_from.resolve()) if a.split_from else None)
    reused, completed, controller = set(), [], None
    remaining=names
    if a.inherit_completed:
        parent=a.inherit_completed.resolve();prior=json.loads((parent/'manifest.json').read_text())
        if (prior.get('status') not in ('FAILED','COMPLETED') or not prior.get('capture_dof_force')
                or prior.get('inherited_split')!=inherited_split or prior.get('source_membership')!=str(source.resolve())):
            raise ValueError('inherited corpus membership/force/split contract differs')
        (out/'batches').mkdir()
        for record in prior['records']:
            folder=(parent/record['path']).resolve();batch_root=folder.parents[1]
            receipt=json.loads((batch_root/'manifest.json').read_text())
            if receipt.get('status')!='COMPLETED' or receipt.get('force_contract',{}).get('schema')!='ref2dex.inspire-measured-forces.v1':
                raise ValueError('cannot inherit incomplete or old force data')
            link=out/'batches'/('inherited-'+batch_root.parents[1].name+'-'+batch_root.name)
            if not link.exists():link.symlink_to(batch_root)
            if link.resolve()!=batch_root:raise ValueError('inherited batch name collision')
            path=(link/folder.relative_to(batch_root)).relative_to(out)
            completed.append(dict(record,path=str(path)))
        existing={r['sequence'] for r in completed}
        if len(existing)!=len(completed) or existing-set(names):raise ValueError('duplicate/outside inherited membership')
        remaining=[name for name in names if name not in existing]
        manifest.update(inherited_completed_from=str(parent),records=completed)
        for key in ('native_controller','observation_contract','force_contract'):
            if key in prior:manifest[key]=prior[key]
    batches = [remaining[i:i+a.batch_size] for i in range(0,len(remaining),a.batch_size)]
    prior_elapsed = 0.
    if previous is not None:
        if previous['source_membership'] != str(source.resolve()):
            raise ValueError('source membership changed during resume')
        reused,completed,controller = recover_batches(out,batches,a.seed,a.capture_current_observation,a.capture_dof_force)
        if a.capture_current_observation and 'observation_contract' in previous:
            manifest['observation_contract'] = previous['observation_contract']
        if controller is not None:
            manifest['native_controller'] = controller
        prior_elapsed = previous.get('elapsed_s',json.loads((out/'progress.json').read_text()).get('elapsed_s',0.))
        history = list(previous.get('resume_history',[]))
        history.append(dict(status=previous['status'],error=previous.get('error'),
            completed_sequences=len(completed),elapsed_s=prior_elapsed,
            physical_gpus=previous['physical_gpus'],git_commit=previous['git_commit']))
        manifest.update(resume_history=history,records=completed)
        write(out/('prior-resume%02d.json'%attempt),previous)
    manifest['batch_size'] = a.batch_size
    write(out/'manifest.json',manifest)
    pending = [(index,batch) for index,batch in enumerate(batches) if index not in reused]
    active = {}
    attempts={};failed_batches=[]
    initial_completed = len(completed)
    started = time.monotonic()
    storage_checked = 0.
    stopped = []
    signal.signal(signal.SIGTERM,lambda *_:stopped.append(True))
    signal.signal(signal.SIGINT,lambda *_:stopped.append(True))
    try:
        while pending or active:
            if stopped or prior_elapsed+time.monotonic()-started>a.seconds:
                raise TimeoutError('requested stop or collection deadline')
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                index,batch = pending.pop(0)
                attempts[index]=attempts.get(index,0)+1
                selection = scratch/('selection%03d.json'%index)
                write(selection,batch)
                folder = out/'batches'/('batch%03d'%index)
                if a.resume:
                    folder = folder.with_name(folder.name+('-resume%02d'%attempt))
                if attempts[index]>1:folder=folder.with_name(folder.name+('-attempt%02d'%attempts[index]))
                folder.parent.mkdir(exist_ok=True)
                command = [RUNTIME,str(TASK/'tools/run/collect_rl660_batch.py'),
                    '--selection',str(selection),'--output',str(folder),'--gpu',gpu,
                    '--seed',str(a.seed+index),'--seconds','900']
                if a.capture_current_observation:
                    command.append('--capture-current-observation')
                if a.capture_dof_force:command.append('--capture-dof-force')
                if a.allow_monica_sharing:command.append('--allow-monica-sharing')
                logname=('batch%03d'%index)+('' if attempts[index]==1 else '-attempt%02d'%attempts[index])+'.log'
                log = (scratch/logname).open('x')
                process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1'))
                active[gpu] = (process,log,folder,index)
            for gpu,(process,log,folder,index) in list(active.items()):
                if process.poll() is None:
                    continue
                log.close()
                if process.returncode:
                    if a.capture_dof_force:
                        del active[gpu]
                        if attempts[index]<3:
                            pending.append((index,batches[index]))
                            print(json.dumps(dict(event='BATCH_RETRY',index=index,attempt=attempts[index],returncode=process.returncode)),flush=True)
                        else:
                            failed_batches.append(dict(index=index,sequences=batches[index],attempts=attempts[index],last_receipt=str(folder)))
                        continue
                    raise RuntimeError('collection batch%d failed; see supervisor scratch'%index)
                receipt = json.loads((folder/'manifest.json').read_text())
                if receipt['status']!='COMPLETED':
                    raise ValueError('incomplete batch receipt')
                if a.capture_dof_force:
                    contract=receipt.get('force_contract',{})
                    if contract.get('schema')!='ref2dex.inspire-measured-forces.v1':raise ValueError('missing measured forces')
                    if manifest.get('force_contract',contract)!=contract:raise ValueError('force contract changed')
                    manifest['force_contract']=contract
                if a.capture_current_observation:
                    contract = receipt.get('observation_contract',{})
                    if (not receipt.get('capture_current_observation') or contract.get('dimension') != 721
                            or not contract.get('live_future_invariance_passed')):
                        raise ValueError('batch missing native current-only observation evidence')
                    if manifest.get('observation_contract',contract) != contract:
                        raise ValueError('native observation contract changes across batches')
                    manifest['observation_contract'] = contract
                controller = receipt['native_controller']
                if manifest.get('native_controller',controller)!=controller:
                    raise ValueError('native controller changes across sequences')
                manifest['native_controller'] = controller
                for record in receipt['records']:
                    record['path'] = str((folder/record['path']).relative_to(out))
                    completed.append(record)
                del active[gpu]
            elapsed = prior_elapsed+time.monotonic()-started
            for gpu,(process,_,_,_) in active.items():
                consumers = set(map(int,subprocess.check_output(['nvidia-smi','-i',gpu,
                    '--query-compute-apps=pid','--format=csv,noheader'],text=True).split()))
                if consumers-{process.pid}-allowed_occupants[gpu]:
                    raise RuntimeError('foreign GPU process appeared on assigned GPU'+gpu)
            new_count = len(completed)-initial_completed
            row = dict(status='RUNNING',completed_sequences=len(completed),target=660,
                elapsed_s=elapsed,eta_s=((elapsed-prior_elapsed)/new_count*(660-len(completed)) if new_count else None),
                active_pids=[v[0].pid for v in active.values()],
                failed_batches=failed_batches,
                gpu=subprocess.check_output(['nvidia-smi','-i',a.gpus,
                    '--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
            write(out/'progress.json',row)
            print(json.dumps(row),flush=True)
            if len(completed)!=len({r['sequence'] for r in completed}):
                raise ValueError('duplicate collected sequence')
            size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
            if size>a.artifact_budget_bytes:
                raise RuntimeError('bounded corpus storage exceeded')
            if elapsed-storage_checked>=120 or storage_checked==0:
                if shutil.disk_usage(ROOT).free < 20*1024**3:
                    raise RuntimeError('less than20GiB filesystem headroom')
                storage_checked=elapsed
            if pending or active:
                time.sleep(15)
        if failed_batches:raise RuntimeError('bounded retries exhausted; completed data retained; failed batches='+str(failed_batches))
        if sorted(r['sequence'] for r in completed)!=names:
            raise ValueError('incomplete 660-member execution corpus')
        # Stable per-object sequence split independent of GPU completion order.
        import random
        rng = random.Random(a.seed+1)
        objects = sorted({r['object'] for r in completed})
        if inherited_split is not None:
            for record in completed:
                record['split'] = inherited_split[record['sequence']]
        else:
            for obj in objects:
                group=sorted([r for r in completed if r['object']==obj],key=lambda r:r['sequence'])
                rng.shuffle(group)
                count=max(1,round(len(group)*.1))
                for index,record in enumerate(group):
                    record['split']='val' if index<count else 'train'
        manifest.update(status='COMPLETED',training_allowed=not a.capture_current_observation,
            capture_complete=True,
            records=sorted(completed,key=lambda r:r['sequence']),elapsed_s=prior_elapsed+time.monotonic()-started)
        import numpy as np
        manifest['max_reference_error_m'] = None if a.capture_dof_force else .1
        if a.capture_dof_force:manifest.update(admission_pending=True,admission_policy='measured rollout audit; no training-reference distance gate')
        for record in manifest['records']:
            with np.load(out/record['path']/'trajectory.npz') as data:
                error = np.linalg.norm(data['obj'][:,:3,3]-data['reference_obj'][:,:3,3],axis=-1)
            count = len(error)-24
            if a.capture_dof_force:
                record.update(raw_windows=count,raw_commands=len(error)-1)
                continue
            peak = np.maximum.reduce([error[i:i+count] for i in range(25)])
            ticks = np.flatnonzero(np.isfinite(peak) & (peak<=.1))
            record.update(raw_windows=count,admitted_windows=len(ticks),
                admitted_ticks=ticks.tolist(),excluded_windows=count-len(ticks))
        manifest['admitted_windows'] = sum(r.get('admitted_windows',0) for r in manifest['records'])
        write(out/'manifest.json',manifest)
        write(out/'progress.json',dict(status='COMPLETED',completed_sequences=660,elapsed_s=manifest['elapsed_s']))
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error),records=completed,
            elapsed_s=prior_elapsed+time.monotonic()-started)
        write(out/'manifest.json',manifest)
        write(out/'progress.json',dict(status='FAILED',error=repr(error),
            completed_sequences=len(completed),target=660,elapsed_s=manifest['elapsed_s'],active_pids=[]))
        raise
    finally:
        for process,log,_,_ in active.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            log.close()


if __name__=='__main__':
    main()
