#!/usr/bin/env python3
"""Fixed compute-matched forecast fits and independent-cohort scoring."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.support_response import SCHEMA
from src.task.CmResidual.paired_evaluation import fingerprint

def model():return torch.nn.Sequential(torch.nn.Linear(72,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))

def logloss(prob,label):
    prob=np.clip(prob.astype(np.float64),1e-7,1-1e-7);label=label.astype(np.float64)
    return float(np.mean(-label*np.log(prob)-(1-label)*np.log1p(-prob)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory;begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root/'run_manifest.json').read_text());audit=json.loads((root/'collection_audit.json').read_text())
    if m['run_status']!='COLLECTION_COMPLETED' or audit['run_status']!='COMPLETED' or sha(root/'dataset.pt')!=audit['dataset_sha256']:raise ValueError('audited cohort provenance')
    if not torch.cuda.is_available():raise ValueError('GPU fit required')
    data=torch.load(root/'dataset.pt',map_location='cpu',weights_only=False);seed=data['seed'].numpy();fit=seed<525;test=~fit
    if fit.sum()!=4608 or test.sum()!=1536 or not np.array_equal(np.unique(seed[test]),[525,526]):raise ValueError('independent cohorts')
    device=torch.device('cuda:0');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw=data['state'][fit].to(device);y=data['label'][fit].to(device);mean=raw.mean(0);std=raw.std(0,unbiased=False).clamp_min(.001);z=((raw-mean)/std).clamp(-10,10)
    action=data['primitive'][fit].to(device)/torch.tensor([.01,.01,.30],device=device);out=root/'fit';out.mkdir(exist_ok=False);fits={};predictions={}
    for variant in ('cm','state_only'):
        torch.manual_seed(741);torch.cuda.manual_seed_all(741);network=model().to(device);initial=fingerprint(network.state_dict());optimizer=torch.optim.Adam(network.parameters(),lr=.001)
        x=torch.cat((z,action if variant=='cm' else torch.zeros_like(action)),-1);losses=[]
        for update in range(1000):
            if time.monotonic()-begin>300:raise TimeoutError('fixed fit budget')
            ids=torch.randint(len(x),(256,),device=device);logits=network(x[ids]).squeeze(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[ids])
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(network.parameters(),10);optimizer.step()
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('nonfinite fit')
            if (update+1)%200==0:losses.append(dict(update=update+1,loss=float(loss)))
        network.eval();path=out/(variant+'.pt');payload=dict(model={k:v.cpu() for k,v in network.state_dict().items()},mean=mean.cpu(),std=std.cpu(),schema=SCHEMA,
            variant=variant,seed=741,updates=1000,fit_rows=4608,fit_seeds=list(range(519,525)),dataset_sha256=sha(root/'dataset.pt'),initial_model_fingerprint=initial,final_model_fingerprint=fingerprint(network.state_dict()),no_test_selection=True)
        torch.save(payload,path);fits[variant]=dict(checkpoint_sha256=sha(path),initial_fingerprint=initial,final_fingerprint=payload['final_model_fingerprint'],loss_trace=losses)
        # Only fixed endpoints are scored. Neither test labels nor test predictions choose a fit.
        with torch.no_grad():
            test_z=((data['state'][test].to(device)-mean)/std).clamp(-10,10);test_action=data['primitive'][test].to(device)/torch.tensor([.01,.01,.30],device=device)
            xx=torch.cat((test_z,test_action if variant=='cm' else torch.zeros_like(test_action)),-1)
            predictions[variant]=torch.sigmoid(network(xx).squeeze(-1)).cpu().numpy()
    if fits['cm']['initial_fingerprint']!=fits['state_only']['initial_fingerprint']:raise ValueError('matched initialization')
    motion=data['motion'].numpy();arm=data['arm'].numpy();labels=data['label'].numpy();global_fit=np.empty((3,8))
    for mo in range(3):
        for aa in range(8):
            ids=fit&(motion==mo)&(arm==aa)
            if ids.sum()!=192:raise ValueError('fit arm coverage')
            global_fit[mo,aa]=(labels[ids].sum()+1)/(ids.sum()+2)
    predictions['global_motion_arm']=global_fit[motion[test],arm[test]]
    test_labels=labels[test];test_motion=motion[test];test_seed=seed[test];scores={}
    def score(ids):return dict(n=int(ids.sum()),positive=int(test_labels[ids].sum()),rate=float(test_labels[ids].mean()),
        logloss={v:logloss(p[ids],test_labels[ids]) for v,p in predictions.items()})
    scores['pooled']=score(np.ones(test.sum(),dtype=bool));scores['motions']={str(mo):score(test_motion==mo) for mo in range(3)}
    scores['seeds_motion1']={str(s):score((test_seed==s)&(test_motion==1)) for s in (525,526)}
    primary=scores['motions']['1']['logloss'];gains={v:1-primary['cm']/primary[v] for v in ('state_only','global_motion_arm')}
    gates=dict(each_test_seed_class10=all(.1<=v['rate']<=.9 for v in scores['seeds_motion1'].values()),
        pooled_cm_gain10_each_control=all(v>=.1 for v in gains.values()),each_seed_lower_than_each_control=all(v['logloss']['cm']<v['logloss'][c] for v in scores['seeds_motion1'].values() for c in ('state_only','global_motion_arm')))
    torch.save(dict(test_mask=torch.from_numpy(test),test_label=torch.from_numpy(test_labels),predictions={k:torch.from_numpy(v) for k,v in predictions.items()},global_fit_probabilities=torch.from_numpy(global_fit)),out/'predictions.pt')
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',primary_motion=1,scores=scores,relative_gain=gains,gates=gates,fits=fits,
        fit_rows=4608,test_rows=1536,updates_each=1000,matched_initialization=True,no_test_selection=True,dataset_sha256=sha(root/'dataset.pt'),predictions_sha256=sha(out/'predictions.pt'),wall_seconds=time.monotonic()-begin,
        boundary='short-horizon physical information Probe only; no policy training utility, new method, generalization or formal Validation')
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
