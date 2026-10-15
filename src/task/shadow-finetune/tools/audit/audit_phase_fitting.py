"""Compare existing paired checkpoints on fixed TRAIN phases; never train or admit tail queries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--code-root',type=Path,required=True,help='Frozen runtime used by both checkpoints')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=6)
    a=p.parse_args();root=Path(__file__).resolve().parents[5]
    out=a.output.resolve()
    if out.exists() or root/'outputs/shadow-finetune' not in out.parents:
        raise ValueError('fresh Task output required')
    controls=json.loads((a.run/'controls.json').read_text())
    code=a.code_root.resolve()
    runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    if runtime_commit!=controls['git_commit']:raise ValueError('original runtime required')
    def pids(gpu):
        s=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        return {int(x) for x in s.split() if x.isdigit()}
    def guard():
        if time.monotonic()-started>600:raise TimeoutError('ten-minute inference audit')
        if pids(a.gpu)-{os.getpid()}:raise RuntimeError('foreign selected-GPU process')
        if a.gpu==6 and pids(7):raise RuntimeError('temporary test requires GPU7 reserve')
    if pids(a.gpu):raise RuntimeError('selected GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    scratch=root/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(scratch),TORCH_EXTENSIONS_DIR=str(root/'tmp/shared-wam-v2-extensions'))
    sys.path.insert(0,str(code/'src/task/shadow-finetune/src'))
    import torch
    from shadow_finetune.data import ShadowWindows,collate,sample_sequence
    from shadow_finetune.model import ShadowPolicy,loss
    from ptv3_shared_wam.loading import to_device
    torch.set_num_threads(2);torch.cuda.set_device(0)
    # Match the saved offline evaluation, including TF32 and batch16 FP32 inference.
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    started=time.monotonic();guard()
    readers={arm:ShadowWindows(path,'train') for arm,path in controls['training_caches'].items()}
    dev=ShadowWindows(controls['validation_cache'],'dev')
    rng=np.random.default_rng(217)
    panels={'common_dev':(dev,[dev.keys[int(i)] for i in controls['validation_ids']]),
            'small_train':(readers['small'],[readers['small'].keys[i] for i in readers['small'].draw(rng,64)]),
            'full_train':(readers['full'],[readers['full'].keys[i] for i in readers['full'].draw(rng,64)])}
    full=readers['full'];phase={'startup':[],'near_goal':[],'tail':[]}
    for i in rng.permutation(len(full.records)):
        i=int(i);row=full.records[i];seq=full.sequence(i)
        ticks=np.asarray(row['ticks'],dtype=np.int64)
        if 0 not in ticks:continue
        near=ticks[np.linalg.norm(seq['poses'][ticks,:3,3]-seq['goal'][ticks],axis=-1)<=.05]
        only=np.arange(max(0,min(len(seq['action'])-7,row['unsafe_onset']-8)))
        local=np.einsum('tpi,tij->tpj',seq['hand']-seq['poses'][:,None,:3,3],seq['poses'][:,:3,:3])
        only=only[(local[only]>=-2).all((1,2))]
        tail=np.setdiff1d(only,ticks)
        if not len(near) or not len(tail):continue
        phase['startup'].append((i,0));phase['near_goal'].append((i,int(rng.choice(near))))
        phase['tail'].append((i,int(rng.choice(tail))))
        if len(phase['startup'])==64:break
    if len(phase['startup'])!=64:raise ValueError('insufficient matched-phase sequences')
    panels.update({'full_'+name:(full,keys) for name,keys in phase.items()})
    samples={};identities={}
    for name,(reader,keys) in panels.items():
        samples[name]=[];identities[name]=[]
        for i,tick in keys:
            seq=reader.sequence(i);row=reader.records[i];admitted=tick in row['ticks']
            if not admitted and name!='full_tail':raise ValueError('unadmitted non-tail query')
            sample=sample_sequence(seq,reader.geometry(i),tick,supervised=admitted)
            if not admitted:sample['action_target']=seq['action'][tick:tick+8].astype(np.float32).copy()
            if sample['action_target'].shape!=(8,24):raise ValueError('action support')
            samples[name].append(sample)
            identities[name].append(dict(record=i,tick=tick,raw_episode=row['raw_episode'],env=row['env'],
                object_code=row['object_code'],admitted=admitted,
                goal_distance_m=float(np.linalg.norm(seq['poses'][tick,:3,3]-seq['goal'][tick]))))
    out.mkdir(parents=True)
    result=dict(status='RUNNING',run=str(a.run.resolve()),runtime_code_root=str(code),
        runtime_commit=runtime_commit,audit_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        seed=217,gpu=a.gpu,panels=identities,arms={},
        scope='fixed small TRAIN/full TRAIN plus matched64 full TRAIN sequences at startup/near-goal/safe omitted tail; tail hypothetical, no training admission; no causal or success claim')
    def write():
        (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write()
    try:
        for arm in ['small','full']:
            guard();packet=torch.load(a.run/('train-'+arm)/'step1000.pt',map_location='cpu',weights_only=False)
            if packet['git_commit']!=runtime_commit:raise ValueError('checkpoint code mismatch')
            model=ShadowPolicy(packet['stats'],packet['config']).cuda().eval()
            model.load_state_dict(packet['model'],strict=True);config=packet['config'];del packet
            result['arms'][arm]={}
            for name,panel in samples.items():
                guard();predictions=[];targets=[];terms_sum={}
                with torch.inference_mode():
                    for start in range(0,len(panel),16):
                        b=to_device(collate(panel[start:start+16]),'cuda');pred=model(b)
                        if not torch.isfinite(pred['action']).all():raise FloatingPointError('nonfinite action')
                        predictions.append(pred['action'].cpu().numpy());targets.append(b['action_target'].cpu().numpy())
                        if name!='full_tail':
                            value,terms=loss(pred,b,config,model)
                            for k,v in dict(total=value,**terms).items():
                                terms_sum[k]=terms_sum.get(k,0.)+float(v)*len(b['action_target'])/len(panel)
                predicted=np.concatenate(predictions);target=np.concatenate(targets)
                if not np.isfinite(target).all():raise ValueError('nonfinite target')
                measures={}
                constant=model.actor.action_mean.detach().cpu().numpy()[None]
                for mode,command in [('raw',predicted),('clipped',np.clip(predicted,-1,1)),
                                     ('constant_mean',np.broadcast_to(constant,predicted.shape))]:
                    err=np.abs(command[:,0]-target[:,0])
                    measures[mode]={k:float(err[:,s].mean()) for k,s in
                        [('all',slice(None)),('force',slice(0,3)),('torque',slice(3,6)),('fingers',slice(6,24))]}
                    measures[mode]['per_example_mae_p95']=float(np.quantile(err.mean(1),.95))
                measures.update(normalized_losses=terms_sum,
                    predicted_first_clip_fraction=float((np.abs(predicted[:,0])>1).mean()),
                    label_first_mean_abs={k:float(np.abs(target[:,0,s]).mean()) for k,s in
                        [('all',slice(None)),('force',slice(0,3)),('torque',slice(3,6)),('fingers',slice(6,24))]})
                if name=='common_dev':
                    expected=json.loads((a.run/('train-'+arm)/'curve.json').read_text())[-1]
                    checks={k:abs(terms_sum[k]-expected[k]) for k in terms_sum}
                    checks['command_first_mae']=abs(measures['raw']['all']-expected['command_first_mae'])
                    measures['saved_dev_max_difference']=max(checks.values())
                    if max(checks.values())>1e-5:raise ValueError('saved DEV reproduction failed: '+str(checks))
                result['arms'][arm][name]=measures
                print(json.dumps(dict(arm=arm,panel=name,raw=measures['raw'],clipped=measures['clipped'])),flush=True)
                if torch.cuda.max_memory_allocated()>4*2**30:raise RuntimeError('4GiB allocation budget')
                write()
            del model;torch.cuda.empty_cache()
        guard();result.update(status='COMPLETED',elapsed_s=time.monotonic()-started,
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        write();print(json.dumps(dict(status=result['status'],elapsed_s=result['elapsed_s'])),flush=True)
    except BaseException as error:
        result.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started);write();raise


if __name__=='__main__':main()
