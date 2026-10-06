#!/usr/bin/env python3
"""Bounded frozen-critic capture campaign and paired one-shot Z comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src'),str(Path(__file__).parent)]
from run_rolling_gt_y import CHECKPOINT,MOTIONS,PYTHON,sha
from critic_ranking import bootstrapped_score,selection_summary
from oracle_y_utility import utility,stable_grasp_z,CANDIDATES
from rolling_y_audit import short_y
PROTOCOL=ROOT/'src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-frozen-critic-ranking.md'


def assemble(root):
    records=[]; y_all=[]; z_all=[]; rewards=[]; values=[]; dones=[]; motions=[]
    inputs={}
    for base,seed in [('oracle-y-utility-s263',263),('oracle-y-utility-extra-s264',264)]:
        schedule=json.loads((ROOT/'outputs/cm-interaction-oracle'/f'{base}-sync-schedule.json').read_text())
        for group in range(2):
            arrays={k:[] for k in ('y','z','reward','value','done')}; reference=None
            for k in range(7):
                folder=root/f's{seed}-g{group}-k{k}'
                source=ROOT/'outputs/cm-interaction-oracle'/(base+'-sync-reference' if k==0 else f'{base}-sync-g{group}-candidate{k}')/'panel.pt'
                old=torch.load(source,map_location='cpu',weights_only=False)
                captured=torch.load(folder/'critic.pt',map_location='cpu',weights_only=False)
                report=json.loads((folder/'critic.json').read_text())
                for p,h in [(source,captured['original_candidate_sha256']),(folder/'critic.pt',report['critic_sha256'])]:
                    if sha(p)!=h:raise ValueError('captured source/output hash drift')
                    inputs[str(p.resolve())]=h
                rows=torch.tensor([i for i,g in enumerate(schedule['groups']) if g==group])
                if not torch.equal(rows,captured['rows']) or captured['candidate']!=k or captured['group']!=group:
                    raise ValueError('candidate identity mismatch')
                if reference is not None and not torch.equal(reference,old['actor_obs'][rows]):
                    raise ValueError('source candidate current obs differs')
                reference=old['actor_obs'][rows]
                y,_=short_y(old['before'][rows,2],old['before'][rows,71]>.5,old['height'][rows,:32],old['pair'][rows,:32],old['rest_height'][rows])
                z,_=stable_grasp_z(old['height'][rows],old['pair'][rows],old['rest_height'][rows])
                for key,value in [('y',y),('z',z),('reward',captured['raw_rewards8']),('value',captured['live_values8']),('done',captured['dones8'])]:
                    arrays[key].append(value)
            y_all.append(torch.stack(arrays['y'],1));z_all.append(torch.stack(arrays['z'],1))
            rewards.append(torch.stack(arrays['reward'],1));values.append(torch.stack(arrays['value'],1));dones.append(torch.stack(arrays['done'],1))
            motions.extend(old['motion_id'][rows].tolist())
            records.extend([dict(seed=seed,group=group,row=int(r),trigger=int(old['triggers'][r]),motion=int(old['motion_id'][r])) for r in rows])
    y,z,r,v,d=[torch.cat(x).numpy() for x in (y_all,z_all,rewards,values,dones)]
    score,rpart,vpart=bootstrapped_score(r,v,d);yscore=utility(y)
    reports={name:selection_summary(s,z,yscore) for name,s in [('Critic_Q8',score),('Value8_only',v),('Reward8_only',rpart),('GT_Y',yscore)]}
    if int(z[:,0].sum())!=23 or reports['GT_Y']['successes']!=24 or int(z.max(1).sum())!=25:
        raise ValueError('original Y/Z counts fail exact reconstruction')
    primary=reports['Critic_Q8'];adequate=len(z)>=30 and len(set(motions))>=2
    passed=adequate and primary['successes']>=reports['GT_Y']['successes'] and primary['rescued']>=1 and primary['harmed']==0
    opportunities=[]
    for i,row in enumerate(records):
        if not z[i,0] and z[i].any():
            opportunities.append(dict(**row,candidate_z=z[i].tolist(),utility_y=yscore[i].tolist(),critic_q8=score[i].tolist(),
                reward_component=rpart[i].tolist(),bootstrap_component=vpart[i].tolist(),value8=v[i].tolist(),
                chosen={a:res['selection'][i] for a,res in reports.items()}))
    result=dict(status='PROMISING' if passed else ('UNPROMISING' if adequate else 'UNCLEAR'),
                anchors=len(z),groups=4,candidate_names=list(CANDIDATES),records=records,arms=reports,
                candidate_z_upper=int(z.max(1).sum()),opportunities=opportunities,input_sha256=inputs,
                rolling_gt_y_reference=dict(baseline=23,rolling_gt_y=27,scope='Separate actually executed rolling experiment, not directly comparable to this one-shot mosaic'),
                scope='Same-prefix one-shot potential-outcome mosaic; no real critic rolling execution or training')
    np.savez_compressed(root/'scores.npz',y=y,z=z,rewards8=r,value8=v,score=score,reward_component=rpart,bootstrap_component=vpart)
    (root/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--gpus',type=int,nargs='+',default=[0,1]);ap.add_argument('--smoke',action='store_true');ap.add_argument('--smoke-candidate',type=int,default=0,choices=range(7));ap.add_argument('--spent-seconds',type=float,default=0)
    args=ap.parse_args();args.run_dir=args.run_dir.resolve();torch.set_num_threads(2)
    if len(args.gpus)>2 or len(set(args.gpus))!=len(args.gpus):raise ValueError('one or two distinct GPUs required')
    args.run_dir.mkdir(parents=True,exist_ok=False);start=time.monotonic();slots=queue.Queue()
    for gpu in args.gpus:slots.put(gpu)
    paths=[Path(__file__),Path(__file__).with_name('collect_critic_candidates.py'),Path(__file__).with_name('collect_oracle_y_candidates.py'),
           ROOT/'src/task/cm-interaction-oracle/src/critic_ranking.py',PROTOCOL,CHECKPOINT]
    inputs={str(p.resolve()):sha(p) for p in paths}
    manifest=dict(status='STARTED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  run_id=args.run_dir.name,input_sha256=inputs,gpus=args.gpus,no_training=True,budget_seconds=240 if args.smoke else 1200-args.spent_seconds,prior_spent_seconds=args.spent_seconds,smoke=args.smoke)
    (args.run_dir/'protocol.md').write_bytes(PROTOCOL.read_bytes())
    (args.run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    events=[]
    def worker(base,seed,group,k):
        gpu=slots.get()
        try:
            if time.monotonic()-start>manifest['budget_seconds']:raise TimeoutError('campaign deadline')
            root=ROOT/'outputs/cm-interaction-oracle';reference=root/(base+'-sync-reference')
            schedule=root/(base+'-sync-schedule.json');name=f's{seed}-g{group}-k{k}';folder=args.run_dir/name
            old=reference/'panel.pt' if k==0 else root/(base+f'-sync-g{group}-candidate{k}')/'panel.pt'
            initial=torch.load(reference/'initial_state.pt',map_location='cpu',weights_only=False)
            config=ROOT/'src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7'
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR=str(ROOT/'tmp'),TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/torch_extensions'),
                MAX_JOBS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LD_LIBRARY_PATH='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH',''))
            cmd=[PYTHON,str(Path(__file__).resolve().with_name('collect_critic_candidates.py')),'--expected-panel',str(old),
                '--run-dir',str(folder),'--reference',str(reference),'--anchor-schedule',str(schedule),'--group-id',str(group),
                '--candidate',str(k),'--post-window','90','--wall-seconds','180','--num_threads','1',
                '--task','Dexplore_Inspire','--cfg_env',str(config/'environment.yaml'),'--cfg_train',str(config/'training.yaml'),
                '--checkpoint',str(CHECKPOINT),'--motion_file',str(MOTIONS),'--headless','--num_envs',str(len(initial['tensors']['data_id'])),
                '--seed',str(seed),'--sim_device','cuda:0','--rl_device','cuda:0','--graphics_device_id','0','--pipeline','cpu',
                '--output',str(folder/'native_eval.json'),'--output_path',str(folder/'native')]
            with (args.run_dir/(name+'.log')).open('x') as log:
                process=subprocess.Popen(cmd,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT)
                try:code=process.wait(timeout=min(240,max(1,manifest['budget_seconds']-(time.monotonic()-start))))
                except BaseException:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise
            if code:raise RuntimeError(name+' failed; see log')
            if sum(p.stat().st_size for p in args.run_dir.rglob('*') if p.is_file())>(.25 if args.smoke else 2)*2**30:
                raise ValueError('storage cap')
            return dict(name=name,gpu=gpu,elapsed_seconds=time.monotonic()-start)
        finally:slots.put(gpu)
    jobs=[(base,seed,g,k) for base,seed in [('oracle-y-utility-s263',263),('oracle-y-utility-extra-s264',264)] for g in range(2) for k in range(7)]
    if args.smoke:jobs=[('oracle-y-utility-s263',263,0,args.smoke_candidate)]
    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures=[pool.submit(worker,*job) for job in jobs]
            for f in as_completed(futures):
                try:event=f.result()
                except BaseException:
                    for future in futures:future.cancel()
                    raise
                events.append(event);manifest['completed']=events
                (args.run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                print(json.dumps(event),flush=True)
        if not args.smoke:
            result=assemble(args.run_dir)
            print(json.dumps(dict(status=result['status'],arms={a:r['successes'] for a,r in result['arms'].items()})),flush=True)
        for p,h in inputs.items():
            if sha(Path(p))!=h:raise ValueError('frozen campaign input drift')
        manifest.update(status='SMOKE_PASS' if args.smoke else 'COMPLETED')
    except BaseException as exc:manifest.update(status='FAILED',error=repr(exc));raise
    finally:
        manifest['elapsed_seconds']=time.monotonic()-start
        (args.run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
