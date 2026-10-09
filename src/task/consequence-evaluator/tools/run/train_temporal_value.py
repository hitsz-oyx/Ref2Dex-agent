"""Matched HA/HAZ ref4_3 weak-label Probe; test is evaluated after val selection."""
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
from consequence_evaluator.temporal_value import SCHEMA,LABEL_RULE,TemporalValue,temporal_loss


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def metrics(pred,label,episode,tick):
    positive=label>0
    per_episode=[float(np.mean((pred[episode==e]-label[episode==e])**2)) for e in np.unique(episode)]
    balanced=.5*(float((pred[positive]>0).mean())+float((pred[~positive]<0).mean()))
    rng=np.random.default_rng(286);correct=[];groups=set()
    for bucket in np.unique(tick//16):
        rows=np.flatnonzero(tick//16==bucket);p=rows[positive[rows]];n=rows[~positive[rows]]
        if not len(p) or not len(n):
            continue
        for _ in range(min(128,len(p)*len(n))):
            i,j=int(rng.choice(p)),int(rng.choice(n))
            correct.append(float(pred[i]>pred[j])+.5*float(pred[i]==pred[j]))
            groups.add((str(episode[i]),str(episode[j])))
    return dict(episode_mse=float(np.mean(per_episode)),mae=float(np.abs(pred-label).mean()),
        balanced_sign_accuracy=balanced,same_time_cross_episode_rank=float(np.mean(correct)) if correct else None,
        diagnostic_pairs=len(correct),unique_episode_pairs=len(groups),episodes=len(per_episode),windows=len(label))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,default=285)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args();data=args.data.resolve();out=args.output.resolve()
    if out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator'):
        raise ValueError('fresh task-owned training output required')
    if not (100<=args.seed<=299 and 1<=args.steps<=2000 and 1<=args.seconds<=600 and args.batch_size<=256):
        raise ValueError('bounded Probe required')
    if subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid',
            '--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    m=json.loads((data/'manifest.json').read_text())
    if m['schema']!=SCHEMA or m['rule']!=LABEL_RULE or not m['training_allowed'] or sha(data/'windows.npz')!=m['windows_sha256']:
        raise ValueError('fixed weak-label data contract required')
    with np.load(data/'windows.npz',allow_pickle=False) as f:
        arrays={key:f[key] for key in f.files}
    for episode in np.unique(arrays['episode']):
        if len(np.unique(arrays['split'][arrays['episode']==episode]))!=1:
            raise ValueError('episode split leakage')
    for group in np.unique(arrays['split_group']):
        if len(np.unique(arrays['split'][arrays['split_group']==group]))!=1:
            raise ValueError('seed-group split leakage')
    indices={split:np.flatnonzero((arrays['split']==split)&arrays['supervised']) for split in ('train','val','test')}
    stats={};inputs={}
    for key in ('history','action','future'):
        value=torch.from_numpy(arrays[key]).float()
        training=value[indices['train']];axes=tuple(range(training.ndim-1))
        mean=training.mean(axes);std=training.std(axes,unbiased=False).clamp_min(1e-4)
        stats[key]=(mean,std);inputs[key]=((value-mean)/std).to('cuda:0')
    labels=torch.from_numpy(arrays['label']).float().to('cuda:0')
    initial=TemporalValue(inputs['history'].shape[-1]);models={arm:copy.deepcopy(initial).to('cuda:0') for arm in ('HA','HAZ')}
    optimizers={arm:torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4) for arm,model in models.items()}
    rng=np.random.default_rng(args.seed);train_ids=indices['train'];groups={}
    for label in (0,1):
        groups[label]=[train_ids[(arrays['episode'][train_ids]==e)] for e in np.unique(arrays['episode'][train_ids])
                       if arrays['episode_success'][train_ids[arrays['episode'][train_ids]==e]][0]==label]
        if not groups[label]:
            raise ValueError('balanced successful and failed train episodes required')
    batches=[]
    for _ in range(args.steps):
        batches.append(np.array([rng.choice(groups[j%2][rng.integers(len(groups[j%2]))])
                                 for j in range(args.batch_size)]))
    hashes={str(path):sha(path) for path in (data/'manifest.json',data/'windows.npz',Path(__file__).resolve(),
        TASK/'src/consequence_evaluator/temporal_value.py',TASK/'src/consequence_evaluator/model.py')}
    out.mkdir(parents=True);torch.save(dict(model=initial.state_dict(),statistics=stats),out/'initial.pt')
    manifest=dict(schema=SCHEMA,status='RUNNING',rule=LABEL_RULE,args=vars(args).copy(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=hashes,initial_sha256=sha(out/'initial.pt'),torch=torch.__version__,
        model_input_whitelist=['history','action','future'],balanced_episode_sampler=True,
        model_architecture=dict(history_dim=1442,width=128,layers=2,bins=41),test_used_for_selection=False)
    manifest['args']={k:str(v) if isinstance(v,Path) else v for k,v in manifest['args'].items()}
    write(out/'manifest.json',manifest);best={arm:float('inf') for arm in models};selected={};start=time.monotonic()
    @torch.inference_mode()
    def predict(model,ids,arm,donor=None):
        model.eval();pred=[]
        for offset in range(0,len(ids),256):
            rows=ids[offset:offset+256];future_rows=rows if donor is None else donor[offset:offset+256]
            pred.append(model(inputs['history'][rows],inputs['action'][rows],inputs['future'][future_rows],arm=='HAZ')['score'].cpu().numpy())
        return np.concatenate(pred)
    completed_steps=0
    for step,batch in enumerate(batches,1):
        if time.monotonic()-start>args.seconds:
            break
        losses={}
        for arm,model in models.items():
            model.train();optimizers[arm].zero_grad(set_to_none=True)
            pred=model(inputs['history'][batch],inputs['action'][batch],inputs['future'][batch],arm=='HAZ')
            loss=temporal_loss(pred['logits'],labels[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite weak-label fit')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizers[arm].step();losses[arm]=float(loss)
        completed_steps=step
        if step%100==0 or step==1 or step==args.steps:
            validation={}
            for arm,model in models.items():
                ids=indices['val'];pred=predict(model,ids,arm)
                value=metrics(pred,arrays['label'][ids],arrays['episode'][ids],arrays['tick'][ids]);validation[arm]=value
                if value['episode_mse']<best[arm]:
                    best[arm]=value['episode_mse'];selected[arm]=step
                    torch.save(dict(schema=SCHEMA,model=model.state_dict(),statistics=stats,
                        architecture=manifest['model_architecture'],arm=arm,step=step,
                        label_rule=LABEL_RULE,data_sha256=m['windows_sha256'],input_sha256=hashes),out/(arm+'.pt'))
            gpu=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=utilization.gpu,memory.used',
                '--format=csv,noheader'],text=True).strip()
            row=dict(step=step,elapsed_s=time.monotonic()-start,loss=losses,validation=validation,gpu=gpu)
            with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps(dict(step=step,elapsed_s=round(row['elapsed_s'],2),gpu=gpu,
                val_rank={arm:value['same_time_cross_episode_rank'] for arm,value in validation.items()})),flush=True)
    # Both frozen val-selected checkpoints are now evaluated on test exactly once.
    test_ids=indices['test'];donor=[];shuffle_rng=np.random.default_rng(287)
    for row in test_ids:
        candidates=test_ids[(arrays['tick'][test_ids]//16==arrays['tick'][row]//16)&(arrays['episode'][test_ids]!=arrays['episode'][row])]
        if not len(candidates):raise ValueError('no same-time other-episode future donor')
        donor.append(int(shuffle_rng.choice(candidates)))
    results={};predictions={}
    for arm,model in models.items():
        saved=torch.load(out/(arm+'.pt'),map_location='cuda:0',weights_only=False);model.load_state_dict(saved['model'])
        predictions[arm]=predict(model,test_ids,arm)
        results[arm]=metrics(predictions[arm],arrays['label'][test_ids],arrays['episode'][test_ids],arrays['tick'][test_ids])
    shuffled=predict(models['HAZ'],test_ids,'HAZ',np.asarray(donor))
    results['HAZ_future_shuffle']=metrics(shuffled,arrays['label'][test_ids],arrays['episode'][test_ids],arrays['tick'][test_ids])
    rank=results['HAZ']['same_time_cross_episode_rank'];base=results['HA']['same_time_cross_episode_rank']
    degradation=rank-results['HAZ_future_shuffle']['same_time_cross_episode_rank']
    gate=bool(rank>=.65 and rank-base>=.03 and degradation>=.03)
    result=dict(status='PROMISING' if gate else 'UNCLEAR',gt_future_information_gate=gate,test=results,
        completed_steps=completed_steps,
        selected_steps=selected,rank_gain_vs_HA=rank-base,future_shuffle_rank_drop=degradation,
        interpretation='cross-episode held-out weak-label prognosis; not same-state control or Gate1',
        input_sha256=hashes,checkpoint_sha256={arm:sha(out/(arm+'.pt')) for arm in models})
    if any(sha(path)!=digest for path,digest in hashes.items()):raise ValueError('training input drift')
    np.savez_compressed(out/'test-predictions.npz',indices=test_ids,HA=predictions['HA'],HAZ=predictions['HAZ'],shuffled=shuffled,donor=np.asarray(donor))
    write(out/'result.json',result);manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-start,
        completed_steps=completed_steps,
        selected_steps=selected,checkpoint_sha256=result['checkpoint_sha256']);write(out/'manifest.json',manifest)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
