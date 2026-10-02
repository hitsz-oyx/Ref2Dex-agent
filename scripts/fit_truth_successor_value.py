"""Three matched fixed-policy task-value fits; actual successor is an oracle only."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha
import numpy as np
import torch
from src.task.CmResidual.truth_successor_value import SCHEMA,VARIANTS,read_panel,features,initialized_model,predictions,phase_template,analyze

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();out=a.output.resolve();assert torch.cuda.is_available() and not out.exists() and ROOT in out.parents
    start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;device=torch.device('cuda:0');m=json.loads((root/'run_manifest.json').read_text());assert m['experiment_id']=='P-20261002-truth-successor-task-value'
    cp=torch.load(Path(m['panel_checkpoints']['547']['path']),map_location='cpu',weights_only=False);assert cp['updates']==0
    states=[cp['variants'][k]['actor'] for k in ('cm','state_only','none')]
    assert all(all(torch.equal(states[0][key],st[key]) for key in states[0]) for st in states[1:])
    fit=read_panel(root/'s547');test=read_panel(root/'s595');assert fit['eligible_rows']>=10000 and test['eligible_rows']>=1024 and test['episodes']>=32
    xs={k:features(fit,k).to(device) for k in VARIANTS};models={k:initialized_model().to(device).train() for k in VARIANTS};optim={k:torch.optim.Adam(models[k].parameters(),lr=3e-4,foreach=False) for k in VARIANTS}
    gpu_fit={k:v.to(device) if torch.is_tensor(v) else v for k,v in fit.items()};generator=torch.Generator(device='cpu').manual_seed(3532);last={}
    for step in range(1500):
        ix=torch.randint(fit['eligible_rows'],(4096,),generator=generator).to(device)
        mini={k:v[ix] if torch.is_tensor(v) and v.shape[:1]==(fit['eligible_rows'],) else v for k,v in gpu_fit.items()}
        for variant in VARIANTS:
            model=models[variant];opt=optim[variant];prob=predictions(model,xs[variant][ix],mini,variant)
            loss=torch.nn.functional.binary_cross_entropy(prob,mini['target']);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            if not torch.isfinite(norm):raise ValueError('finite valuefitgradient')
            opt.step();last[variant]=float(loss)
    preds={};errors={};parameters={};gpu_test={k:v.to(device) if torch.is_tensor(v) else v for k,v in test.items()}
    with torch.no_grad():
        for variant in VARIANTS:
            model=models[variant].eval();x=features(test,variant);preds[variant]=predictions(model,x.to(device),gpu_test,variant).cpu();state={k:v.detach().cpu() for k,v in model.state_dict().items()};parameters[variant]=state
            z=x.numpy().astype(np.float64)
            for index in (0,2):z=np.maximum(z@state[str(index)+'.weight'].numpy().astype(np.float64).T+state[str(index)+'.bias'].numpy().astype(np.float64),0)
            z=z@state['4.weight'].numpy().astype(np.float64).T+state['4.bias'].numpy().astype(np.float64);z=1/(1+np.exp(-z.clip(-700,700)));z=z.ravel()
            if variant=='oracle_successor':z=np.where(test['next_known'].numpy(),test['next_value'].numpy(),z)
            errors[variant]=float(np.abs(z-preds[variant].numpy()).max())
            if errors[variant]>2e-5:raise ValueError(('fullvalueNumPyreplay',variant,errors[variant]))
    table=phase_template(fit);preds['motion_time']=table[test['motion'],test['timebin']];report=analyze(test,preds);out.mkdir()
    torch.save(dict(schema=SCHEMA,parameters=parameters,phase_template=table,updates=1500,init_seed=3531,batch_seed=3532,device='cuda',actor_not_trained=True),out/'models.pt')
    torch.save(dict(predictions=preds,target=test['target'],cluster=test['cluster'],tick=test['tick'],environment=test['environment'],next_known=test['next_known'],schema=SCHEMA),out/'predictions.pt')
    report.update(run_status='COMPLETED',fit_rows=fit['eligible_rows'],test_rows=test['eligible_rows'],test_episodes=test['episodes'],fit_next_known_fraction=fit['next_known_fraction'],test_next_known_fraction=test['next_known_fraction'],full_numpy_forward_maximum=errors,last_loss=last,actual_optimizer_steps=4500,model_sha256=sha(out/'models.pt'),prediction_sha256=sha(out/'predictions.pt'),privileged_oracle_only=True,no_policy_or_cm_training=True,wall_seconds=time.monotonic()-start)
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
