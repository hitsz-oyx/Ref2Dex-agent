"""Engineering-only saved-minibatch replay; no optimizer or physics updates."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='GPU-0606f00a-d9d0-3a00-5b49-c9e747b77307'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
from scripts.resume_continuous_critic_after_resource_failure import admission,verify_gpu
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.continuous_critic_cm import initialized_models,actor_loss,critic_loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--update',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or ROOT not in a.output.resolve().parents:raise ValueError('new own diagnosis')
    admitted=admission(4);verify_gpu(admitted);begin=time.monotonic();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pack=torch.load(a.update/'update_packet.pt',map_location='cpu',weights_only=False)
    results={};payload={}
    for variant,rec in pack['records'].items():
        actor,critic=initialized_models();actor.load_state_dict(rec['actor_before']);critic.load_state_dict(rec['critic_before']);actor=actor.cuda().train();critic=critic.cuda().train()
        outputs={};hooks=[]
        for prefix,model in [('actor',actor),('critic',critic)]:
            for key,module in model.named_modules():
                if isinstance(module,torch.nn.Linear):
                    def hook(module,inputs,value,name=prefix+'.'+key):outputs[name]=value.detach().cpu()
                    hooks.append(module.register_forward_hook(hook))
        x=rec['input'].cuda();mean,logstd=actor(x);value,pred=critic(x,rec['action'].cuda(),variant)
        al=actor_loss(mean,logstd,rec['request'].cuda(),rec['old_logp'].cuda(),rec['advantage'].cuda());cl=critic_loss(value,pred,rec['return_target'].cuda(),rec['transition'].cuda(),variant)
        (al+cl).backward()
        graderror={name:float((param.grad.cpu()-rec['gradient'][name]).abs().max()) for name,param in list(actor.named_parameters())+[('critic.'+k,v) for k,v in critic.named_parameters()]}
        diff=[];activations=rec['input'].numpy().astype(np.float64)
        for key in ['network.0','network.2','network.4']:
            weights=rec['actor_before'];pre=activations@weights[key+'.weight'].numpy().astype(np.float64).T+weights[key+'.bias'].numpy().astype(np.float64)
            actual=outputs['actor.'+key].numpy().astype(np.float64)
            indices=np.argwhere((pre>0)!=(actual>0)) if key!='network.4' else []
            flips=[dict(sample=int(i),unit=int(j),numpy64=float(pre[i,j]),gpu32=float(actual[i,j])) for i,j in indices]
            diff.append(dict(layer=key,maximum_preactivation_error=float(np.abs(pre-actual).max()),sign_flips=flips))
            activations=np.maximum(pre,0) if key!='network.4' else pre
        results[variant]=dict(saved_gradient_replay_error=graderror,actor_relu_comparison=diff,
            saved_forward_maximum=max(float((t.cpu()-rec[k]).abs().max()) for t,k in [(mean,'mean'),(value,'value'),(pred,'prediction')]),
            actor_loss_error=abs(float(al)-rec['actor_loss']),critic_loss_error=abs(float(cl)-rec['critic_loss']))
        payload[variant]=outputs
        for h in hooks:h.remove()
    a.output.mkdir();torch.save(payload,a.output/'replayed_preactivations.pt')
    record=dict(run_status='COMPLETED',engineering_only=True,no_optimizer_or_physics_updates=True,source_sha256={str(a.update/f):sha(a.update/f) for f in ['update_packet.pt','policy_heads.pt','results.json']},admission=admitted,results=results,wall_seconds=time.monotonic()-begin,script_sha256=sha(Path(__file__)))
    (a.output/'results.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
