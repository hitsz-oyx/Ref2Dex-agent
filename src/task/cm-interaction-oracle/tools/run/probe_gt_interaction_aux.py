#!/usr/bin/env python3
"""Bounded matched PPO GT-aux training and actor-only evaluation campaign."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import yaml
ROOT=Path(__file__).resolve().parents[5]
PYTHON=Path('/home2/wyy/miniconda3/envs/graspenv/bin/python')
TASK=ROOT/'src/task/cm-interaction-oracle'
SOURCE=ROOT/'outputs/Dexplore/agent_v139_s3_backtrack_s70_e260'
CHECKPOINT=SOURCE/'train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth'
MOTIONS=ROOT/'outputs/CmResidual/agent_contact_option_airplane_motions'
SOURCE_SHA='16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f'
ARMS=('plain','conditioned','shuffle','stopgrad')
PROTOCOL=TASK/'docs/experiments/probes/P-20261006-gt-interaction-aux.md'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data): Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def launch(command, gpu, folder, deadline):
    folder.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
               TMPDIR=str(ROOT/'tmp'),TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/torch_extensions'),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT/'tmp/torchinductor'),PYTHONPATH=':'.join((str(ROOT),str(TASK/'src'),str(ROOT/'src/task/CmResidual/tools'),env.get('PYTHONPATH',''))))
    started=time.monotonic()
    record=dict(command=command,gpu=gpu,status='RUNNING',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(folder/'manifest.json',record)
    try:
        with (folder/'run.log').open('w') as out:
            proc=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,
                                  stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=proc.pid;write(folder/'manifest.json',record)
            try: code=proc.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise TimeoutError('own task subprocess exceeded fixed deadline')
        if code: raise RuntimeError(f'worker exited {code}: {folder}/run.log')
        record['status']='COMPLETED'
    except BaseException as exc:
        record.update(status='FAILED',error=str(exc));raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started;write(folder/'manifest.json',record)
    return record


def gpu_check(gpus):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.splitlines()}
    if any(used[g]>1024 for g in gpus): raise RuntimeError('requested GPU occupied')


def training(root,gpus,smoke):
    count=1 if smoke else 64;seed=42 if smoke else 292
    source=json.loads((SOURCE/'config.json').read_text())
    cfg=yaml.safe_load((ROOT/'third_party/DExplore/dexplore/data/cfg/train/rlg/inspire.yaml').read_text())
    cfg['params']['config']['save_intermediate']=False
    cfg['params']['config']['save_frequency']=100000
    traincfg=root/'training.yaml';traincfg.write_text(yaml.safe_dump(cfg,sort_keys=False))
    end=260+count
    jobs=[]
    for i,arm in enumerate(ARMS):
        folder=root/arm
        command=[str(PYTHON),'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
                 str(TASK/'tools/run/gt_aux_bootstrap.py'),'--gt-arm',arm,'--cm-distill-coef','0',
                 '--actual-epochs',str(end),'--approach-reward-coef','2',
                 '--held-lift-reward-coef','10','--lift-progress-reward-coef','5',
                 '--scratch-resume-checkpoint',str(CHECKPOINT.resolve()),
                 '--scratch-resume-sha256',SOURCE_SHA,'--learning-rate','1e-5',
                 '--contact-before','3','--contact-after','3','--contact-fraction','.5',
                 '--lift-fraction','.25','--curriculum-backtrack-start','180',
                 '--curriculum-backtrack-end','220',
                 '--task','Dexplore_Inspire','--cfg_env','dexplore/data/cfg/inspire.yaml',
                 '--cfg_train',str(traincfg),'--motion_file',str(MOTIONS.resolve()),
                 '--output_path',str(folder/'train'),'--headless','--sim_device','cuda:0',
                 '--rl_device','cuda:0','--graphics_device_id','0','--num_envs','64',
                 '--horizon_length','32','--minibatch_size','256',
                 '--max_iterations',str(end),'--seed',str(seed),'--horovod','--resume','1',
                 '--checkpoint',str(CHECKPOINT.resolve())]
        jobs.append((command,gpus[i%len(gpus)],folder))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=[1,2,3,4])
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--stage',choices=('train','eval'),default='train')
    a=p.parse_args();root=a.run_dir.resolve();gpu_check(a.gpus)
    if len(a.gpus)!=4 or len(set(a.gpus))!=4: raise ValueError('four distinct idle GPUs required')
    if sha(CHECKPOINT)!=SOURCE_SHA: raise ValueError('source checkpoint changed')
    start=time.monotonic();budget=600 if a.smoke else (1800 if a.stage=='train' else 900)
    if a.stage=='train':
        root.mkdir(parents=True,exist_ok=False)
        (root/'protocol.md').write_bytes(PROTOCOL.read_bytes())
        jobs=training(root,a.gpus,a.smoke)
    else:
        source=json.loads((SOURCE/'config.json').read_text())
        jobs=[]
        for i,arm in enumerate((*ARMS,'source')):
            if arm=='source': checkpoint=CHECKPOINT
            else:
                found=list((root/arm/'train').rglob('GRAB.pth'))
                if len(found)!=1: raise ValueError('missing final checkpoint')
                checkpoint=found[0]
            folder=root/'evaluation'/arm
            command=[str(PYTHON),str(TASK/'tools/run/evaluate_gt_aux.py'),
                '--task','Dexplore_Inspire','--cfg_env','dexplore/data/cfg/inspire.yaml',
                '--cfg_train','dexplore/data/cfg/train/rlg/inspire.yaml',
                '--motion_file',str(MOTIONS.resolve()),'--checkpoint',str(checkpoint.resolve()),
                '--disable-early-termination','--headless','--sim_device','cuda:0',
                '--rl_device','cuda:0','--graphics_device_id','0','--num_envs','96',
                '--seed','293','--output',str(folder/'results.json')]
            jobs.append((command,a.gpus[i%4],folder))
    paths=[PROTOCOL,Path(__file__),TASK/'src/gt_interaction_aux.py',TASK/'src/dexplore_gt_aux_agent.py',
           TASK/'tools/run/gt_aux_bootstrap.py',CHECKPOINT,SOURCE/'config.json',
           ROOT/'third_party/DExplore/dexplore/data/cfg/inspire.yaml',
           ROOT/'third_party/DExplore/dexplore/data/cfg/train/rlg/inspire.yaml']
    if a.stage=='eval':paths.append(TASK/'tools/run/evaluate_gt_aux.py')
    paths += sorted(MOTIONS.rglob('*.pt'))
    inputs={str(v.resolve()):sha(v) for v in paths}
    record=dict(status='RUNNING',stage=a.stage,smoke=a.smoke,run_id=root.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=inputs,gpus=a.gpus,budget_seconds=budget,events=[])
    manifest=root/f'{a.stage}_manifest.json';write(manifest,record)
    try:
        # source eval follows trained plain on the same GPU; no two workers per GPU.
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending={pool.submit(launch,*job,start+budget):job for job in jobs[:4]}
            for future in as_completed(pending):
                result=future.result();record['events'].append(result);write(manifest,record)
                print(json.dumps(dict(folder=str(pending[future][2]),elapsed=result['elapsed_seconds'])),flush=True)
        if len(jobs)>4:
            record['events'].append(launch(*jobs[4],start+budget))
        if any(sha(v)!=h for v,h in inputs.items()):raise ValueError('input drift during stage')
        record['status']='COMPLETED'
    except BaseException as exc:
        record.update(status='FAILED',error=str(exc));raise
    finally:
        record['elapsed_seconds']=time.monotonic()-start;write(manifest,record)


if __name__=='__main__':main()
