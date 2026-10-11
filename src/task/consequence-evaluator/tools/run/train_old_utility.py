"""Matched C0/C1/C2 frozen-U32 regression; test on untouched ref13 candidates."""
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
from scipy.stats import spearmanr

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.data import sha
from consequence_evaluator.contracts import is_within
from consequence_evaluator.old_utility import SCHEMA,OldUtility,panel_metrics


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def regression(y,pred,episode):
    mse=[float(np.mean((pred[episode==e]-y[episode==e])**2)) for e in np.unique(episode)]
    corr=float(spearmanr(y,pred).statistic) if np.std(y)>0 and np.std(pred)>0 else None
    return dict(mae=float(np.abs(y-pred).mean()),rmse=float(np.sqrt(np.mean((y-pred)**2))),
        episode_mse=float(np.mean(mse)),spearman=corr,episodes=len(mse),windows=len(y))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--pw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--seconds',type=int,default=600)
    a=p.parse_args();data=a.data.resolve();pw=a.pw.resolve();out=a.output.resolve()
    if out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator') or not 1<=a.steps<=1200 or not 1<=a.seconds<=600:
        raise ValueError('fresh bounded matched fit required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);torch.set_num_threads(2);torch.manual_seed(288)
    torch.backends.cuda.matmul.allow_tf32=False
    m=json.loads((data/'manifest.json').read_text());pm=json.loads((pw/'manifest.json').read_text())
    if (m['schema']!=SCHEMA or pm['status']!='COMPLETED' or not pm['oracle_observed_hand']
            or m['windows_sha256']!=sha(data/'windows.npz') or pm['windows_sha256']!=m['windows_sha256']
            or pm['future_sha256']!=sha(pw/'future.npz')):raise ValueError('fixed same-row GT/PW data required')
    with np.load(data/'windows.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    with np.load(pw/'future.npz',allow_pickle=False) as f:predicted=f['future']
    if predicted.shape!=d['future'].shape:raise ValueError('C1/C2 future schema mismatch')
    for group in np.unique(d['split_group']):
        if len(np.unique(d['split'][d['split_group']==group]))!=1:raise ValueError('split leakage')
    ids={split:np.flatnonzero(d['split']==split) for split in ('train','val','test','panel')}
    stats={};inputs={};tr=ids['train']
    for key in ('history','action','future'):
        x=torch.from_numpy(d[key]).float();train=x[tr];axes=tuple(range(train.ndim-1))
        if key=='action':mean=torch.zeros(18);std=torch.full((18,),.5)
        else:mean=train.mean(axes);std=train.std(axes,unbiased=False).clamp_min(1e-3 if key=='future' else 1e-4)
        stats[key]=(mean,std);inputs[key]=((x-mean)/std).cuda()
    mean,std=stats['future'];inputs['pw']=((torch.from_numpy(predicted)-mean)/std).cuda()
    labels=torch.from_numpy(d['label']).float().cuda();initial=OldUtility()
    models={arm:copy.deepcopy(initial).cuda() for arm in ('C0','C1','C2')}
    opt={arm:torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4) for arm,model in models.items()}
    active=np.abs(d['action']).max((1,2))>1e-7;groups={}
    for flag in (False,True):
        rows=tr[active[tr]==flag];groups[flag]=[rows[d['episode'][rows]==e] for e in np.unique(d['episode'][rows])]
        if not groups[flag]:raise ValueError('active/inactive plan coverage required')
    rng=np.random.default_rng(288);batches=[]
    for _ in range(a.steps):
        batches.append(np.array([rng.choice(groups[j%2==0][rng.integers(len(groups[j%2==0]))]) for j in range(128)]))
    hashes={str(path):sha(path) for path in (data/'manifest.json',data/'windows.npz',pw/'manifest.json',pw/'future.npz',
        Path(__file__).resolve(),TASK/'src/consequence_evaluator/old_utility.py',TASK/'src/consequence_evaluator/model.py')}
    out.mkdir(parents=True);torch.save(initial.state_dict(),out/'initial.pt')
    manifest=dict(schema=SCHEMA,status='RUNNING',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        input_sha256=hashes,initial_sha256=sha(out/'initial.pt'),physical_gpu=a.gpu,seed=288,
        architecture=dict(history_dim=1442,width=128,layers=2),steps_cap=a.steps,seconds_cap=a.seconds,
        objective='MSE frozen U32',sampler='50%active /50%inactive, episode-uniform then window-uniform',
        action_normalization='fixed native residual unit0.5; no tiny data-std amplification',
        C2='observed-hand-conditioned PointWorld oracle, not deployable',test_used_for_selection=False)
    write(out/'manifest.json',manifest);best={arm:float('inf') for arm in models};selected={};start=time.monotonic();completed=0
    @torch.inference_mode()
    def predict(model,rows,arm,donor=None):
        model.eval();values=[]
        # Multiple of seven avoids numerical pseudo-ties across panel boundaries.
        for begin in range(0,len(rows),252):
            chosen=rows[begin:begin+252];future_rows=chosen if donor is None else donor[begin:begin+252]
            future=inputs['pw' if arm=='C2' else 'future'][future_rows]
            values.append(model(inputs['history'][chosen],inputs['action'][chosen],future,arm!='C0').cpu().numpy())
        return np.concatenate(values)
    for step,batch in enumerate(batches,1):
        if time.monotonic()-start>a.seconds:break
        losses={}
        for arm,model in models.items():
            model.train();opt[arm].zero_grad(set_to_none=True)
            q=model(inputs['history'][batch],inputs['action'][batch],inputs['pw' if arm=='C2' else 'future'][batch],arm!='C0')
            loss=((q-labels[batch])**2).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite teacher fit')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt[arm].step();losses[arm]=float(loss)
        completed=step
        if step==1 or step%100==0 or step==a.steps:
            val={}
            for arm,model in models.items():
                rows=ids['val'];pred=predict(model,rows,arm);metric=regression(d['label'][rows],pred,d['episode'][rows]);val[arm]=metric
                if metric['episode_mse']<best[arm]:
                    best[arm]=metric['episode_mse'];selected[arm]=step
                    torch.save(dict(schema=SCHEMA,model=model.state_dict(),statistics=stats,architecture=manifest['architecture'],
                        arm=arm,step=step,label='frozen U32',data_sha256=m['windows_sha256'],PW_sha256=pm['checkpoint_sha256']),out/(arm+'.pt'))
            elapsed=time.monotonic()-start;gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
            row=dict(step=step,elapsed_s=elapsed,losses=losses,val=val,gpu=gpu)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(step=step,elapsed_s=round(elapsed,1),eta_s=round(elapsed/step*(a.steps-step),1),
                gpu=gpu,val_rmse={arm:v['rmse'] for arm,v in val.items()})),flush=True)
    # Freeze all val-selected weights before opening ordinary and same-H tests.
    panels=np.unique(d['panel'][ids['panel']]);panel_rows=np.array([sorted(ids['panel'][d['panel'][ids['panel']]==i],key=lambda r:d['candidate'][r]) for i in panels])
    if panel_rows.shape!=(25,7):raise ValueError('complete frozen25x7 panel required')
    for rows in panel_rows:
        if not np.array_equal(d['candidate'][rows],np.arange(7)) or not np.all(d['history'][rows]==d['history'][rows[0]]):
            raise ValueError('held panel identity failure')
    flat=panel_rows.ravel();rng=np.random.default_rng(290)
    donor=np.stack([np.roll(rows,int(rng.integers(1,7))) for rows in panel_rows]).ravel()
    target=d['label'][flat].reshape(25,7);metrics={};ordinary={};predictions={};shuffled={}
    for arm,model in models.items():
        saved=torch.load(out/(arm+'.pt'),map_location='cuda',weights_only=False);model.load_state_dict(saved['model'])
        predictions[arm]=predict(model,flat,arm).reshape(25,7);metrics[arm]=panel_metrics(target,predictions[arm])
        rows=ids['test'];test=predict(model,rows,arm);ordinary[arm]=regression(d['label'][rows],test,d['episode'][rows])
        if arm!='C0':shuffled[arm]=predict(model,flat,arm,donor).reshape(25,7)
    shuffle_metrics={arm:panel_metrics(target,q) for arm,q in shuffled.items()}
    gain=metrics['C1']['pairwise_accuracy']-metrics['C0']['pairwise_accuracy']
    drop=metrics['C1']['pairwise_accuracy']-shuffle_metrics['C1']['pairwise_accuracy']
    gt_gate=bool(metrics['C1']['pairwise_accuracy']>=.70 and gain>=.03 and drop>=.03
                 and metrics['C1']['mean_regret']<=metrics['C0']['mean_regret'])
    pw_gap=metrics['C1']['pairwise_accuracy']-metrics['C2']['pairwise_accuracy']
    pw_gate=bool(gt_gate and metrics['C2']['pairwise_accuracy']>=.70 and pw_gap<=.05
                 and metrics['C2']['mean_regret']<=metrics['C1']['mean_regret']+.05)
    result=dict(status='PROMISING' if gt_gate else 'UNCLEAR',gt_information_gate=gt_gate,PW_retention_gate=pw_gate,
        metrics=metrics,shuffle_metrics=shuffle_metrics,ordinary_test=ordinary,teacher=panel_metrics(target,target),
        C1_gain_vs_C0=gain,C1_shuffle_drop=drop,C1_minus_C2=pw_gap,selected_steps=selected,completed_steps=completed,
        evidence_scope='single recovered cohort,5informative anchors; old U teacher ranking, not full-task success',
        C2_oracle_observed_hand=True,deployable_planner=False,actual_Z90_executed=False,
        checkpoint_sha256={arm:sha(out/(arm+'.pt')) for arm in models},input_sha256=hashes)
    if any(sha(path)!=digest for path,digest in hashes.items()):raise ValueError('fit inputs drift')
    np.savez_compressed(out/'panel-predictions.npz',target=target,panel_rows=panel_rows,donor=donor,**predictions,
                        C1_shuffle=shuffled['C1'],C2_shuffle=shuffled['C2'])
    write(out/'result.json',result);manifest.update(status='COMPLETED',completed_steps=completed,selected_steps=selected,
        elapsed_s=time.monotonic()-start,checkpoint_sha256=result['checkpoint_sha256']);write(out/'manifest.json',manifest)
    print(json.dumps({key:result[key] for key in ('status','gt_information_gate','PW_retention_gate','metrics','shuffle_metrics','C1_gain_vs_C0','C1_shuffle_drop','C1_minus_C2')},indent=2),flush=True)


if __name__=='__main__':main()
