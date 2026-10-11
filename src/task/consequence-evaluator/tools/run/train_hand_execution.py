"""Matched action-conditioned/H-only bridge fit on disjoint ordinary episodes."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.data import sha
from consequence_evaluator.contracts import is_within
from consequence_evaluator.hand_execution import HandExecution,SCHEMA,current_frame,transform_future,compose_motion,metrics


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_source(path):
    m=json.loads((path/'manifest.json').read_text())
    if m['status']!='COMPLETED' or m['schema']!=SCHEMA or m['mode']!='random' or m['smoke']:
        raise ValueError('complete ordinary random episodes required')
    if m['decisions_sha256']!=sha(path/'decisions.npz'):raise ValueError('decision data drift')
    with np.load(path/'decisions.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    states=[];hands=[];target=[];nominal=[];objects=[]
    for i in range(len(d['tick'])):
        obj,hand=current_frame(d['object_history'][i],d['hand_history'][i])
        states.append(np.r_[d['history'][i],hand.ravel()]);hands.append(hand[-1]);objects.append(obj)
        target.append(transform_future(d['object_history'][i,-1],d['hand_future'][i]))
        nominal.append(transform_future(d['object_history'][i,-1],d['nominal'][i]))
    return dict(state=np.array(states,np.float32),current=np.array(hands,np.float32),target=np.array(target,np.float32),
                nominal=np.array(nominal,np.float32),action=d['action'].astype('float32'),episode=d['env'],
                candidate=d['candidate'],tick=d['tick'],object_history=np.array(objects),
                hand_history=np.array([current_frame(o,h)[1] for o,h in zip(d['object_history'],d['hand_history'])])),m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for split in ('train','val','test'):p.add_argument('--'+split,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--plan-units',choices=('uniform','physical'),default='uniform')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator') or not 1<=a.steps<=2000 or not 1<=a.seconds<=600:
        raise ValueError('fresh bounded fit required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);torch.set_num_threads(2);torch.manual_seed(291)
    torch.backends.cuda.matmul.allow_tf32=False
    data={};meta={};hashes={};start=time.monotonic()
    for split in ('train','val','test'):
        path=getattr(a,split).resolve();data[split],meta[split]=load_source(path)
        for name in ('manifest.json','decisions.npz'):hashes[str(path/name)]=sha(path/name)
    if len({m['seed'] for m in meta.values()})!=3 or len({m['actor_sha256'] for m in meta.values()})!=1:
        raise ValueError('whole-source split/actor mismatch')
    hashes[str(Path(__file__).resolve())]=sha(__file__)
    source=TASK/'src/consequence_evaluator/hand_execution.py';hashes[str(source)]=sha(source)
    tr=torch.from_numpy(data['train']['state']);mean=tr.mean(0);std=tr.std(0,unbiased=False).clamp_min(1e-4)
    plan_unit=torch.tensor([.01]*3+[.1]*15) if a.plan_units=='physical' else torch.full((18,),.5)
    tensors={split:{'state':((torch.from_numpy(d['state'])-mean)/std).cuda(),
        'action':(torch.from_numpy(d['action'])/plan_unit).cuda(),'current':torch.from_numpy(d['current']).cuda(),
        'target':torch.from_numpy(d['target']).cuda()} for split,d in data.items()}
    initial=HandExecution();models={arm:copy.deepcopy(initial).cuda() for arm in ('HA','H')}
    opts={arm:torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4) for arm,m in models.items()}
    rng=np.random.default_rng(291);groups=[np.flatnonzero(data['train']['episode']==e) for e in np.unique(data['train']['episode'])]
    out.mkdir();torch.save(initial.state_dict(),out/'initial.pt')
    manifest=dict(schema=SCHEMA,status='RUNNING',seed=291,physical_gpu=a.gpu,input_sha256=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=meta,
        steps_cap=a.steps,seconds_cap=a.seconds,source_actor_sha256=meta['train']['actor_sha256'],
        test_used_for_selection=False,architecture=dict(width=256),plan_units=a.plan_units,common_motion='explicit wrist translation plus relative finger changes',
        input_whitelist=['raw_current_H1442','four_past_current_hand_points_in_fixed_current_object_frame','known_residual_plan24x18'])
    write(out/'manifest.json',manifest);best={arm:float('inf') for arm in models};selected={}
    @torch.inference_mode()
    def predict(model,split,zero_plan=False):
        model.eval();t=tensors[split];values=[]
        for begin in range(0,len(t['state']),256):
            s=slice(begin,begin+256);plan=t['action'][s]
            if zero_plan:plan=torch.zeros_like(plan)
            motion=model(t['state'][s],plan)*.01
            values.append(compose_motion(t['current'][s],motion).cpu().numpy())
        return np.concatenate(values)
    for step in range(1,a.steps+1):
        if time.monotonic()-start>a.seconds:raise TimeoutError('fit budget')
        batch=np.array([rng.choice(groups[rng.integers(len(groups))]) for _ in range(128)])
        t=tensors['train'];losses={}
        for arm,model in models.items():
            model.train();opts[arm].zero_grad(set_to_none=True)
            plan=t['action'][batch] if arm=='HA' else torch.zeros_like(t['action'][batch])
            pred=compose_motion(t['current'][batch],model(t['state'][batch],plan)*.01)
            loss=((pred-t['target'][batch])/.01).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite fit')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opts[arm].step();losses[arm]=float(loss)
        if step==1 or step%100==0 or step==a.steps:
            vals={}
            for arm,model in models.items():
                q=predict(model,'val',arm=='H');d=data['val'];error=((q-d['target'])**2).sum(-1).mean((1,2))
                val=float(np.mean([error[d['episode']==e].mean() for e in np.unique(d['episode'])]));vals[arm]=val
                if val<best[arm]:
                    best[arm]=val;selected[arm]=step
                    torch.save(dict(schema=SCHEMA,model=model.state_dict(),statistics=(mean,std),architecture=dict(width=256),
                        output_unit_m=.01,plan_unit=plan_unit,step=step,arm=arm,actor_sha256=meta['train']['actor_sha256']),out/(arm+'.pt'))
            elapsed=time.monotonic()-start;gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
            row=dict(step=step,elapsed_s=elapsed,eta_s=elapsed/step*(a.steps-step),loss=losses,val_episode_mse=vals,gpu=gpu)
            print(json.dumps(row),flush=True)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    # Checkpoints are frozen before any test predictions are made.
    freeze={arm:sha(out/(arm+'.pt')) for arm in models};results={};saved={}
    for arm,model in models.items():
        model.load_state_dict(torch.load(out/(arm+'.pt'),map_location='cuda',weights_only=False)['model'])
        q=predict(model,'test',arm=='H');saved[arm]=q;results[arm]=metrics(q,data['test']['target'])
    d=data['test'];persistent=np.repeat(d['current'][:,None],24,1)
    results['persistence']=metrics(persistent,d['target']);results['nominal_FK']=metrics(d['nominal'],d['target'])
    zero=predict(models['HA'],'test',True);results['HA_zero_plan']=metrics(zero,d['target']);active=d['candidate']!=0
    zero_mse=float(np.mean(np.sum((zero[active]-d['target'][active])**2,-1)))
    normal_mse=float(np.mean(np.sum((saved['HA'][active]-d['target'][active])**2,-1)))
    sensitivity=zero_mse/normal_mse-1
    per_candidate={str(k):{arm:metrics(q[d['candidate']==k],d['target'][d['candidate']==k])
        for arm,q in dict(saved,persistence=persistent,nominal_FK=d['nominal']).items()} for k in np.unique(d['candidate'])}
    gate=(results['HA']['point_rmse_m']<=.9*results['persistence']['point_rmse_m'] and
          results['HA']['point_rmse_m']<=.9*results['nominal_FK']['point_rmse_m'] and
          results['HA']['h24_rmse_m']<results['persistence']['h24_rmse_m'] and sensitivity>=.01)
    report=dict(status='PROMISING' if gate else 'UNCLEAR',bridge_gate=bool(gate),metrics=results,per_candidate=per_candidate,
        active_zero_plan_relative_mse_increase=sensitivity,test_windows=len(d['state']),test_episodes=int(len(np.unique(d['episode']))),
        selected_steps=selected,checkpoint_sha256=freeze,scope='ordinary held single realized plans; no same-H candidate ranking',
        zero_motion_equals_current_persistence=True,elapsed_s=time.monotonic()-start)
    np.savez_compressed(out/'test-predictions.npz',**saved,zero_plan=zero,target=d['target'],candidate=d['candidate'],episode=d['episode'],tick=d['tick'])
    write(out/'result.json',report)
    if any(sha(p)!=h for p,h in hashes.items()) or any(sha(out/(arm+'.pt'))!=h for arm,h in freeze.items()):raise ValueError('fit input/weight drift')
    manifest.update(status='COMPLETED',selected_steps=selected,checkpoint_sha256=freeze,completed_steps=a.steps,elapsed_s=time.monotonic()-start)
    write(out/'manifest.json',manifest);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
