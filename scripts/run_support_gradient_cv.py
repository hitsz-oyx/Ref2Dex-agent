#!/usr/bin/env python3
"""Fixed reused-data full-gradient control-variate decision screen."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import admission,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1);a=p.parse_args()
    source=a.source.resolve();out=a.output.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('new isolated output')
    metadata=json.loads((source/'results.json').read_text());m=json.loads((source/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or metadata['label']!='UNPROMISING':raise ValueError('closed failed forecast design')
    files=[source/'run_manifest.json',source/'results.json',source/'collection_audit.json',source/'dataset.pt',source/'fit/predictions.pt',source/'fit/cm.pt',source/'fit/state_only.pt',
        Path(__file__),ROOT/'src/task/CmResidual/support_gradient_cv.py',ROOT/'docs/decisions/D-20261002-support-gradient-control-variate.md']
    hashes={str(f):sha(f) for f in files};out.mkdir();begin=time.monotonic();manifest=dict(run_status='RUNNING',pid=os.getpid(),input_sha256=hashes,reused_test_data=True,no_training=True,no_new_physics=True,wall_limit_seconds=300,storage_limit_bytes=50<<20)
    try:
        gpu=admission(a.gpu);manifest['admission']=gpu;os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid'];os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
        import numpy as np
        import torch
        from scripts.fit_support_response_information import model
        from src.task.CmResidual.support_response import PRIMITIVES
        from src.task.CmResidual.support_gradient_cv import corrected_logit_gradient
        from src.task.CmResidual.paired_evaluation import fingerprint
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda:0')
        data=torch.load(source/'dataset.pt',map_location='cpu',weights_only=False);saved=torch.load(source/'fit/predictions.pt',map_location='cpu',weights_only=False)
        if sha(source/'dataset.pt')!=metadata['dataset_sha256'] or sha(source/'fit/predictions.pt')!=metadata['predictions_sha256']:raise ValueError('dataset/prediction provenance')
        test=data['seed']>=525
        if not torch.equal(test,saved['test_mask']) or int(test.sum())!=1536:raise ValueError('all fixed test rows')
        state=data['state'][test].to(device);arms=data['arm'][test].to(device);reward=data['physical105'][test].to(device,dtype=torch.float64);N=len(state);q={};maximum_replay_error=0.
        with torch.no_grad():
            for variant in ('cm','state_only'):
                path=source/'fit'/(variant+'.pt')
                if sha(path)!=metadata['fits'][variant]['checkpoint_sha256']:raise ValueError('fixed checkpoint drift')
                payload=torch.load(path,map_location='cpu',weights_only=False);network=model().to(device);network.load_state_dict(payload['model']);network.eval()
                if fingerprint(network.state_dict())!=payload['final_model_fingerprint']:raise ValueError('predictor weight fingerprint')
                mean=payload['mean'].to(device);std=payload['std'].to(device);z=((state-mean)/std).clamp(-10,10)
                macro=torch.tensor(PRIMITIVES,device=device)/torch.tensor([.01,.01,.30],device=device)
                actions=macro[None].expand(N,-1,-1) if variant=='cm' else torch.zeros(N,8,3,device=device)
                x=torch.cat((z[:,None].expand(-1,8,-1),actions),-1).reshape(N*8,72)
                pred=torch.cat([torch.sigmoid(network(x[i:i+1024])).squeeze(-1) for i in range(0,len(x),1024)]).reshape(N,8)
                assigned=pred[torch.arange(N,device=device),arms].cpu();error=float((assigned-saved['predictions'][variant]).abs().max());maximum_replay_error=max(maximum_replay_error,error)
                if error>1e-5:raise ValueError('assigned probability replay drift')
                q[variant]=pred.double()
            q['global_motion_arm']=saved['global_fit_probabilities'].to(device)[data['motion'][test].to(device)]
            torch.manual_seed(751);torch.cuda.manual_seed_all(751)
            trunk=torch.nn.Sequential(torch.nn.Linear(69,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU()).to(device)
            hidden=torch.cat((trunk(z),torch.ones(N,1,device=device)),-1).double();bias=torch.zeros(8,device=device,dtype=torch.float64);bias[0]=2
            pi=torch.softmax(bias,-1)[None].expand(N,-1);behavior=torch.full_like(pi,1/8);gradients={};moments={}
            for variant,prediction in q.items():
                g=corrected_logit_gradient(pi,behavior,arms,reward,prediction)
                gradients[variant]=(g[:,:,None]*hidden[:,None,:]).cpu()
                moments[variant]=gradients[variant].square().sum((1,2)).numpy()
        seeds=data['seed'][test].numpy();motion=data['motion'][test].numpy()
        def summarize(ids):
            second={k:float(v[ids].mean()) for k,v in moments.items()}
            mean={k:g[ids].mean(0).numpy() for k,g in gradients.items()}
            return dict(n=int(ids.sum()),reward105_count=int(reward.cpu().numpy()[ids].sum()),second_moment=second,
                cm_minus_control={k:second['cm']-second[k] for k in ('state_only','global_motion_arm')},
                relative_second_moment_gain={k:1-second['cm']/second[k] if second[k]>0 else None for k in ('state_only','global_motion_arm')},
                finite_sample_mean_gradient_difference_norm={k:float(np.linalg.norm(mean['cm']-mean[k])) for k in ('state_only','global_motion_arm')})
        pooled=summarize(np.ones(N,dtype=bool));panels={str(s):summarize(seeds==s) for s in (525,526)};motions={str(mm):summarize(motion==mm) for mm in range(3)}
        gates=dict(pooled_second_moment_gain20_both=all(v is not None and v>=.2 for v in pooled['relative_second_moment_gain'].values()),
            each_seed_lower_both=all(p['cm_minus_control'][k]<0 for p in panels.values() for k in ('state_only','global_motion_arm')))
        torch.save(dict(gradients=gradients,counterfactual_physical30={k:v.cpu() for k,v in q.items()},policy_probability=pi[0].cpu(),
            initial_actor_trunk=trunk.cpu().state_dict(),actor_mean=mean.cpu(),actor_std=std.cpu(),test_seed=data['seed'][test],test_motion=data['motion'][test],physical105=reward.cpu()),out/'analysis.pt')
        result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',reused_data=True,posthoc_to_forecasting=True,
            forecasting_primary_label_unchanged='UNPROMISING',no_training=True,no_new_physics=True,coefficient=1,actor_seed=751,
            actor_final_weights_zero=True,actor_bias=[2,0,0,0,0,0,0,0],all_full_last_layer_gradient_coordinates=520,
            pooled=pooled,seeds=panels,motions=motions,gates=gates,maximum_assigned_probability_replay_error=maximum_replay_error,
            boundary='gradient second-moment candidate only; no trained-policy benefit, novel theorem or formal Validation')
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        if any(sha(Path(f))!=h for f,h in hashes.items()):raise ValueError('protected input drift')
        if time.monotonic()-begin>300 or sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>50<<20:raise RuntimeError('budget')
        manifest.update(run_status='COMPLETED',label=result['label'],input_sha256=hashes,inputs_unchanged=True)
        print(json.dumps(result))
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:
        manifest['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
