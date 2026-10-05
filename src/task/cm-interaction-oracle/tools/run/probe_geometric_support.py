#!/usr/bin/env python3
"""Post-ref10 Decision: isolate bilinear amplification from point-flow scaling."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(TASK/'src'));sys.path.insert(0,str(TASK/'tools/run'))
sys.path.insert(0,str(TASK/'tools/audit'))
from geometric_consequence import normalize,pca_apply,pad,action_design,ridge_fit,ridge_predict
from probe_conditional_consequence import preprocess
from probe_interventions import sha
from consequence_sufficiency import cluster_gain
from conditional_consequence import contrast_scores
from audit_geometric_consequence import gpu_tree

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    def save(name,value): (args.run_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    manifest=dict(run_status='STARTED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_diagnostic_sha256=sha(args.source_run/'diagnostic.pt'),source_manifest_sha256=sha(args.source_run/'manifest.json'),
        dataset_sha256=sha(args.dataset),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),ridge=32.,physical_flow_scale_m=.02,
        seeds=[241,242],code_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            (Path(__file__).resolve(),TASK/'src/geometric_consequence.py',TASK/'tools/audit/audit_geometric_consequence.py')},
        created_at=datetime.now(timezone.utc).isoformat(),protocol='Two-by-two flow scaling × state-action products; additive matched Arm/Joint controls; no OOF or task refits')
    save('manifest.json',manifest)
    try:
        assert json.loads((args.source_run/'manifest.json').read_text())['run_status']=='COMPLETED'
        d=torch.load(args.source_run/'diagnostic.pt',map_location='cpu',weights_only=False)
        p=torch.load(args.dataset,map_location='cpu',weights_only=False)
        assert manifest['dataset_sha256']=='138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149'
        device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        n=gpu_tree(d['preprocess']['full'],device);geo=gpu_tree(d['geometry'],device)
        train,test=d['train'],d['test'];tr=torch.as_tensor(train,device=device)
        h,_,_=preprocess(p,tr,device,n['h']);h=torch.cat((h,pca_apply(geo['geometry'],n['geometry'])),-1)
        rows=torch.arange(len(p['arm']),device=device);arms=p['arm'].to(device)
        z=d['z'].to(device);base=d['base'].to(device);scale=d['target_scale'].to(device)
        onehot=torch.nn.functional.one_hot(arms,15)[:,1:].float()
        projected=(geo['flow'][rows,arms]-n['flow']['mean'])@n['flow']['projection']
        acts=dict(StateAdditive=torch.zeros(len(rows),32,device=device),
            ArmAdditive=pad(normalize(onehot,n['arm'])),JointAdditive=pad(normalize(geo['joint'][rows,arms],n['joint'])),
            FlowAdditive=normalize(projected,n['flow']['score']),
            FlowPhysicalAdditive=projected/.02,FlowPhysicalProducts=projected/.02)
        predictions,states,metrics,contrasts={}, {}, {}, {}
        target=(z-base)/scale
        for name,a in acts.items():
            x=action_design(h,a)
            if name!='FlowPhysicalProducts':x[:,h.shape[1]+32:]=0
            model=ridge_fit(x,target,tr)
            pred=ridge_predict(x,model)*scale+base
            error=((pred-z)/scale).square().cpu().numpy()[test]
            metrics[name]=dict(E_mse=float(error[:,:12].mean()),I_mse=float(error[:,12:].mean()),joint_mse=float(error.mean()),
                train_joint_mse=float(((pred[tr]-z[tr])/scale).square().mean()),
                feature_abs_max_train=float(x[tr].abs().max()),feature_abs_max_test=float(x[test].abs().max()))
            predictions[name]=pred.cpu().numpy();states[name]=gpu_tree(model,'cpu')
            values=[]
            for arm in range(15):
                if name=='StateAdditive':ac=torch.zeros(len(test),32,device=device)
                elif name=='ArmAdditive':
                    oh=torch.zeros(len(test),14,device=device)
                    if arm:oh[:,arm-1]=1
                    ac=pad(normalize(oh,n['arm']))
                elif name=='JointAdditive':ac=pad(normalize(geo['joint'][test,arm],n['joint']))
                else:
                    proj=(geo['flow'][test,arm]-n['flow']['mean'])@n['flow']['projection']
                    ac=normalize(proj,n['flow']['score']) if name=='FlowAdditive' else proj/.02
                xc=action_design(h[test],ac)
                if name!='FlowPhysicalProducts':xc[:,h.shape[1]+32:]=0
                values.append((ridge_predict(xc,model)*scale+base[test]).cpu().numpy())
            effect=(np.stack(values,1)[:,1:]-np.stack(values,1)[:,:1]).mean(0)
            beta=d['GT_contrasts']
            contrasts[name]=dict(raw_I=contrast_scores(beta,effect,scale.cpu().numpy(),np.arange(12,26)),
                centered_I=contrast_scores(beta-beta.mean(0),effect-effect.mean(0),scale.cpu().numpy(),np.arange(12,26)))
        predictions['FlowProducts']=d['full_predictions']['Flow']
        predictions['TrainMean']=d['full_predictions']['TrainMean']
        errors={name:((pred[test]-d['z'].numpy()[test])/d['target_scale'].numpy())**2 for name,pred in predictions.items()}
        comparisons={}
        for new,control in (('FlowAdditive','FlowProducts'),('FlowPhysicalProducts','FlowProducts'),
            ('FlowPhysicalAdditive','FlowAdditive'),('FlowPhysicalAdditive','StateAdditive'),
            ('FlowPhysicalAdditive','JointAdditive'),('FlowPhysicalAdditive','ArmAdditive'),('FlowPhysicalAdditive','TrainMean')):
            comparisons[new+'_vs_'+control]={key:cluster_gain(errors[control][:,axes].mean(1),errors[new][:,axes].mean(1),d['clusters'][test],242)
                for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))}
        sufficient=all(comparisons['FlowPhysicalAdditive_vs_'+control]['I']['gain']>=.03
            for control in ('StateAdditive','ArmAdditive','JointAdditive','TrainMean')) and contrasts['FlowPhysicalAdditive']['centered_I']['gain_vs_zero']>0
        result=dict(status='PROMISING' if sufficient else 'UNPROMISING',metrics=metrics,comparisons=comparisons,contrasts=contrasts,
            elapsed_seconds=time.monotonic()-started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),policy_executed=False,
            limitations='Post-result fixed support diagnosis, previously exposed test set; no tuning sweep, OOF task transfer or independent Validation.')
        save('result.json',result)
        torch.save(dict(predictions=predictions,states=states,train=train,test=test,source_diagnostic_sha256=manifest['source_diagnostic_sha256']),args.run_dir/'diagnostic.pt')
        assert sha(args.source_run/'diagnostic.pt')==manifest['source_diagnostic_sha256']
        assert all(sha(ROOT/key)==value for key,value in manifest['code_sha256'].items())
        manifest['run_status']='COMPLETED';print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}');raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic()-started,completed_at=datetime.now(timezone.utc).isoformat());save('manifest.json',manifest)

if __name__=='__main__':main()
