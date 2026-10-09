"""Common frozen C1 score waterfall on single realized held-out plans."""
import argparse
import importlib.util
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
from consequence_evaluator.hand_planner import HandPlanner
from consequence_evaluator.old_utility import teacher


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--test',type=Path,required=True)
    p.add_argument('--bridge',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TRITON_CACHE_DIR=str(ROOT/'tmp/hand-execution/triton'))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic()
    cfg=json.loads(a.inputs.read_text());model=HandPlanner(a.bridge,cfg['C1'],cfg['PW'],cfg['canonical'])
    spec=importlib.util.spec_from_file_location('bridge_training',TASK/'tools/run/train_hand_execution.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    d,meta=module.load_source(a.test)
    with np.load(a.test/'decisions.npz',allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    with np.load(a.test/'trajectory.npz',allow_pickle=False) as f:trajectory={k:f[k] for k in f.files}
    hashes={str(path.resolve()):sha(path) for path in (a.inputs,a.test/'manifest.json',a.test/'decisions.npz',a.test/'trajectory.npz',Path(__file__))}
    gt=[];labels=[]
    for i,(t,e) in enumerate(zip(raw['tick'],raw['env'])):
        obj=raw['object_history'][i,-1];future=np.linalg.inv(obj)[None]@raw['object_future'][i]
        hand=(raw['hand_future'][i]-raw['object_future'][i,:,:3,3,None].transpose(0,2,1))@np.eye(3)
        hand=np.einsum('tki,tij->tkj',hand,raw['object_future'][i,:,:3,:3])
        gt.append(np.concatenate((future[:,:3].reshape(24,12),hand.reshape(24,33)),-1))
        y,u=teacher(trajectory['object_pose'][t,e,2,3][None],trajectory['pair'][t,e][None],
            trajectory['object_pose'][t+1:t+33,e,2,3][None],trajectory['pair'][t+1:t+33,e][None],
            trajectory['reference_object_pose'][0,e,2,3][None])
        labels.append(u[0])
    gt=np.array(gt,np.float32);label=np.array(labels)
    future_hand=model.predict_hand(raw['history'],d['object_history'],d['hand_history'],d['action'])
    futures={'C1_GT':gt,'C2a_GT_hand_PW_object':model.predict_object(d['object_history'],d['hand_history'],d['target']),
        'C2b_pred_hand_PW_object':model.predict_object(d['object_history'],d['hand_history'],future_hand)}
    scores={k:model.score(raw['history'],d['action'],v) for k,v in futures.items()}
    report={}
    for k,q in scores.items():
        report[k]=dict(teacher_rmse=float(np.sqrt(np.mean((q-label)**2))),teacher_mae=float(np.abs(q-label).mean()),
            common_C1_GT_score_rmse=float(np.sqrt(np.mean((q-scores['C1_GT'])**2))))
    model.verify()
    if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('waterfall input drift')
    elapsed=time.monotonic()-start
    if elapsed>180:raise TimeoutError('waterfall budget')
    a.output.mkdir();np.savez_compressed(a.output/'scores.npz',**scores,target=label,episode=d['episode'],tick=d['tick'])
    result=dict(status='UNCLEAR',scope='held ordinary single realized plans, not candidate ranking or control outcomes',
        common_frozen_C1=True,C2a_oracle=True,C2b_predicted_hand=True,windows=len(label),metrics=report,
        elapsed_s=elapsed,input_sha256=dict(hashes,**model.hashes),source_seed=meta['seed'])
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
