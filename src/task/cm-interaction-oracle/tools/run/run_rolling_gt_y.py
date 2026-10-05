#!/usr/bin/env python3
"""Cold replay forks and actual mixed execution for fixed rolling GT-Y Probe."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import queue
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from rolling_control import OFFSETS,candidate_scores,mixed_plan,execution_z,control_gate
from oracle_y_utility import stable_grasp_z,CANDIDATES,utility
from rolling_y_audit import short_y

PYTHON='/home2/wyy/miniconda3/envs/graspenv/bin/python'
CHECKPOINT=Path('/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth')
MOTIONS=Path('/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/CmResidual/agent_contact_option_airplane_motions')
OUTPUT=ROOT/'outputs/cm-interaction-oracle'


def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


class Campaign:
    def __init__(self,args):
        self.args=args;self.begin=time.monotonic();self.root=args.run_dir.resolve()
        self.root.mkdir(parents=True,exist_ok=False)
        self.events=[];self.processes={};self.summaries=[]
        self.gpu_slots=queue.Queue()
        for gpu in args.gpus or [args.gpu]:
            self.gpu_slots.put(gpu)
            self.gpu_slots.put(gpu)
        self.commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        paths=[Path(__file__),ROOT/'src/task/cm-interaction-oracle/tools/run/collect_oracle_y_candidates.py',
               ROOT/'src/task/cm-interaction-oracle/src/rolling_control.py',
               ROOT/'src/task/cm-interaction-oracle/src/rolling_y_audit.py',
               ROOT/'src/task/cm-interaction-oracle/src/oracle_y_utility.py',args.protocol]
        self.inputs={str(p.resolve()):sha(p) for p in paths}
        (self.root/'frozen_protocol.md').write_text(args.protocol.read_text())
        self.progress('STARTED')

    def progress(self,stage,**kwargs):
        event=dict(stage=stage,elapsed_seconds=time.monotonic()-self.begin,**kwargs)
        self.events.append(event)
        value=dict(status=stage,run_id=self.root.name,git_commit=self.commit,work_version=self.commit,
                   pid=os.getpid(),physical_gpus=self.args.gpus or [self.args.gpu],inputs=self.inputs,
                   elapsed_seconds=event['elapsed_seconds'],events=self.events)
        (self.root/'progress.json').write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(event),flush=True)

    def check(self):
        if time.monotonic()-self.begin>self.args.wall_seconds:raise TimeoutError('campaign deadline')
        if any(sha(Path(p))!=h for p,h in self.inputs.items()):raise ValueError('campaign input drift')
        if sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())>self.args.storage_gib*2**30:
            raise ValueError('campaign storage cap')

    def worker(self,*args,**kwargs):
        gpu=self.gpu_slots.get()
        try:return self._worker(*args,physical_gpu=gpu,**kwargs)
        finally:self.gpu_slots.put(gpu)

    def _worker(self,name,reference,schedule,group,seed,offset,candidate=0,window=32,record=False,plan=None,physical_gpu=6):
        self.check(); folder=self.root/name
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(physical_gpu),TMPDIR=str(ROOT/'tmp'),
            TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/torch_extensions'),MAX_JOBS='2',OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',LD_LIBRARY_PATH='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH',''))
        initial=load(reference/'initial_state.pt'); n=len(initial['tensors']['data_id'])
        config=ROOT/'src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7'
        cmd=[PYTHON,str(ROOT/'src/task/cm-interaction-oracle/tools/run/collect_oracle_y_candidates.py'),
             '--run-dir',str(folder),'--reference',str(reference),'--anchor-schedule',str(schedule),
             '--group-id',str(group),'--rolling-offset',str(offset),'--candidate',str(candidate),
             '--post-window',str(window),'--wall-seconds','180','--num_threads','1',
             '--task','Dexplore_Inspire','--cfg_env',str(config/'environment.yaml'),
             '--cfg_train',str(config/'training.yaml'),'--checkpoint',str(CHECKPOINT),
             '--motion_file',str(MOTIONS),'--headless','--num_envs',str(n),'--seed',str(seed),
             '--sim_device','cuda:0','--rl_device','cuda:0','--graphics_device_id','0','--pipeline','cpu',
             '--output',str(folder/'native_eval.json'),'--output_path',str(folder/'native')]
        if record:cmd.append('--record-rolling-trace')
        if plan is not None:cmd.extend(['--rolling-plan',str(plan)])
        log=self.root/(name+'.log')
        with log.open('x') as f:
            process=subprocess.Popen(cmd,cwd=ROOT/'third_party/DExplore',env=env,stdout=f,stderr=subprocess.STDOUT)
            self.processes[name]=process
            try:code=process.wait(timeout=min(240,max(1,self.args.wall_seconds-(time.monotonic()-self.begin))))
            except BaseException:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
            finally:self.processes.pop(name,None)
        if code:raise RuntimeError(f'{name} failed ({code}); see {log}')
        report=json.loads((folder/'result.json').read_text())
        return folder,load(folder/'panel.pt'),report

    def repeated_control(self,base,seed,group):
        reference=OUTPUT/(base+'-sync-reference');schedule=OUTPUT/(base+'-sync-schedule.json')
        old=load(reference/'panel.pt');info=json.loads(schedule.read_text())
        rows=torch.tensor([i for i,g in enumerate(info['groups']) if g==group],dtype=torch.long)
        origin=info['clocks'][group]
        initial=load(reference/'initial_state.pt')
        remaining=initial['tensors']['max_episode_length'][old['motion_id'][rows]]-old['start_frame'][rows]-origin
        if not (remaining>121).all() or initial['scalars']['rollout_length']-origin<=121:
            raise ValueError('original cohort lacks full final32step lookahead; do not filter')
        name=f's{seed}-g{group}'
        a,pa,_=self.worker(name+'-baseline',reference,schedule,group,seed,0,window=90,record=True)
        b,pb,_=self.worker(name+'-repeat',reference,schedule,group,seed,0,window=90)
        for key in ('before','history','actor_obs','hand_root','height','pair','valid_steps','actions','pd_targets'):
            if not torch.equal(pa[key][rows],pb[key][rows]) or not torch.equal(pa[key][rows],old[key][rows]):
                raise ValueError('fresh baseline/repeat/old drift: '+key)
        zb,detail=stable_grasp_z(pa['height'][rows],pa['pair'][rows],pa['rest_height'][rows])
        control=dict(base=base,seed=seed,group=group,rows=rows.tolist(),origin=origin,
                     baseline=zb.tolist(),baseline_qualification=detail['first_stable_step'].tolist(),
                     baseline_dir=str(a),schedule=str(schedule),motion=pa['motion_id'][rows].tolist(),
                     rest=pa['rest_height'][rows].tolist(),all_baseline_repeat_fields_exact=True,
                     full_future_support_min=int(remaining.min()))
        write(self.root/(name+'-control.json'),control)
        self.progress('CONTROL_PASSED',seed=seed,group=group,anchors=len(rows))
        return control

    def run_group(self,control):
        seed=control['seed'];group=control['group'];name=f's{seed}-g{group}'
        schedule=Path(control['schedule']);reference=Path(control['baseline_dir'])
        rows=torch.tensor(control['rows']);decisions=[]
        for offset in OFFSETS:
            self.check();prefix=f'{name}-t{offset:02d}'
            results=[None]*7
            results[0]=self.worker(prefix+'-k0',reference,schedule,group,seed,offset,0,record=True)
            p0=results[0][1]
            baseline_y,_=short_y(p0['before'][rows,2],p0['before'][rows,71]>.5,
                                p0['height'][rows],p0['pair'][rows],p0['rest_height'][rows])
            baseline_score=utility(baseline_y)
            certified=bool((baseline_score==1.25).all())
            if certified:
                # U<=1+.25 for EVERY candidate; exact ties prefer baseline.
                # Store only the observed baseline Y, never invent other labels.
                choices=torch.zeros(len(p0['triggers']),dtype=torch.long)
                y=baseline_y[:,None];scores=baseline_score[:,None]
                results=results[:1]
            else:
                with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                    futures={pool.submit(self.worker,prefix+f'-k{k}',reference,schedule,group,seed,offset,k):k for k in range(1,7)}
                    for future in as_completed(futures):results[futures[future]]=future.result()
                panels=[r[1] for r in results]; y,scores=candidate_scores(panels,rows)
                choices=mixed_plan(scores,rows,len(panels[0]['triggers']))
            # Only independent physical forks run concurrently. Choice/execution stays sequential.
            plan_path=self.root/(prefix+'-plan.json')
            write(plan_path,dict(offset=offset,choices=choices.tolist(),rows=rows.tolist(),
                source_prefix_sha256=sha(reference/'trace.pt'),y=y.tolist(),utility=scores.tolist(),
                candidate_panel_sha256={str(r[0]/'panel.pt'):sha(r[0]/'panel.pt') for r in results},
                baseline_upper_bound_certificate=certified,
                tie_rule='baseline first then fixed candidate order'))
            actual,pa,_=results[0] if certified else self.worker(prefix+'-execute',reference,schedule,group,seed,offset,record=True,plan=plan_path)
            for key in ('before','history','actor_obs','hand_root'):
                if not torch.allclose(pa[key][rows],p0[key][rows],atol=1e-4,rtol=0):
                    raise ValueError('actual execution current H mismatch')
            if not certified and not torch.equal(pa['rolling_choices'],choices):raise ValueError('choice did not enter execution')
            expected_base=pa['base_actions'][rows]
            expected=(expected_base+pa['delta'][choices[rows],None]).clamp(-1,1)
            if not torch.equal(pa['actions'][rows],expected):raise ValueError('native executed residual mismatch')
            gap=(scores.max(-1).values-scores[:,0])
            chosen_first8_height=pa['height'][rows,:8] if certified else torch.stack([panels[int(choices[r])]['height'][r,:8] for r in rows])
            difference=(pa['height'][rows,:8]-chosen_first8_height).abs()
            decision=dict(offset=offset,choices=choices[rows].tolist(),utility_gain_over_zero=gap.tolist(),
                          clipped_steps=pa['clipped_steps'][rows].tolist(),actual_dir=str(actual),
                          candidate_vs_actual_first8_height_max=float(difference.max()),
                          full_world_prefix_max=pa['full_world_prefix_errors'].amax(0).tolist())
            decisions.append(decision);reference=actual
            self.progress('ROUND_EXECUTED',seed=seed,group=group,offset=offset,
                          selections=np.bincount(choices[rows],minlength=7).tolist(),
                          first8_height_max=float(difference.max()),baseline_max_certificate=certified)
        trace=load(reference/'trace.pt')
        rolling,detail=execution_z(trace,rows,control['origin'],torch.tensor(control['rest']))
        report=dict(**control,rolling=rolling.tolist(),decisions=decisions,
                    rolling_qualification=detail['first_stable_step'].tolist(),
                    rolling_dropped=detail['dropped_after_qualification'].tolist(),final_trace=str(reference/'trace.pt'))
        write(self.root/(name+'-result.json'),report);self.summaries.append(report)
        self.progress('GROUP_COMPLETED',seed=seed,group=group,baseline=sum(control['baseline']),rolling=int(rolling.sum()))

    def finish(self):
        baseline=np.concatenate([s['baseline'] for s in self.summaries])
        rolling=np.concatenate([s['rolling'] for s in self.summaries])
        motion=np.concatenate([s['motion'] for s in self.summaries])
        report=control_gate(baseline,rolling,motion)
        report.update(groups=self.summaries,code_commit=self.commit,
            old_one_shot_reference=dict(baseline=23,selected_y=24,finite_candidate_z=25,anchors=32,
                interpretation='Historical one-shot panel;25/32 is NOT a rolling upper bound; no gain-retention ratio.'),
            candidate_names=list(CANDIDATES),offsets=list(OFFSETS),no_training=True,
            scope='Executed mixed-arm receding horizon on four cold-replay groups; exposed cohort, aggregate force proxy and boundedZ; Probe only.')
        write(self.root/'result.json',report)
        np.savez_compressed(self.root/'result.npz',baseline=baseline,rolling=rolling,motion=motion)
        self.progress('COMPLETED',baseline=report['baseline_count'],rolling=report['rolling_count'],verdict=report['status'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--gpu',type=int,default=6)
    p.add_argument('--gpus',type=int,nargs='+',help='Idle physical GPUs, at most2 workers/device and4 total')
    p.add_argument('--workers',type=int,default=4,choices=(1,2,3,4));p.add_argument('--wall-seconds',type=int,default=7200)
    p.add_argument('--storage-gib',type=float,default=4);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.gpus and (len(args.gpus)>4 or len(set(args.gpus))!=len(args.gpus)):
        raise ValueError('one to four distinct physical GPUs required')
    torch.set_num_threads(2);campaign=Campaign(args)
    try:
        specifications=[('oracle-y-utility-s263',263,0),('oracle-y-utility-s263',263,1),
                        ('oracle-y-utility-extra-s264',264,0),('oracle-y-utility-extra-s264',264,1)]
        if args.smoke:specifications=specifications[:1]
        controls=[campaign.repeated_control(*spec) for spec in specifications]
        if args.smoke:
            c=controls[0];reference=Path(c['baseline_dir']);schedule=Path(c['schedule']);rows=torch.tensor(c['rows'])
            # Engineering smoke: nonzero mixed plan then fresh next-state repeat forks.
            n=len(load(reference/'panel.pt')['triggers']);choices=torch.zeros(n,dtype=torch.long)
            choices[rows[::2]]=2
            path=campaign.root/'smoke-plan.json';write(path,dict(choices=choices.tolist()))
            actual,_,_=campaign.worker('smoke-mixed-execute',reference,schedule,0,263,0,record=True,plan=path)
            a,pa,_=campaign.worker('smoke-next-zero-a',actual,schedule,0,263,8)
            b,pb,_=campaign.worker('smoke-next-zero-b',actual,schedule,0,263,8)
            for key in ('before','history','actor_obs','height','pair','actions'):
                if not torch.equal(pa[key][rows],pb[key][rows]):raise ValueError('next-state repeat drift: '+key)
            campaign.progress('SMOKE_PASSED',mixed_nonzero=int((choices>0).sum()),all_next_state_repeat_exact=True)
        else:
            for control in controls:campaign.run_group(control)
            campaign.finish()
    except BaseException as e:
        for process in list(campaign.processes.values()):
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        campaign.progress('FAILED',error=repr(e));raise
if __name__=='__main__':main()
