"""Matched no-Cm task-Q fits and short-oracle ranking, never terminal-oracle selection."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    import numpy as np,torch
    from src.task.CmResidual.oracle_features import descriptor,OracleQ,masked,ARMS
    from scripts.run_contact_response_probe import sha
    out=a.source;fit=out/'fit';fit.mkdir();begin=time.monotonic();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    options=np.load(out/'options.npy');packets={};hashes={}
    for scene in ('train','eval'):
        data=[]
        for index,option in enumerate(options):
            path=out/scene/('c%02d'%index)
            if index==0:path=out/scene/'baseline'
            data.append(descriptor(path,out/scene/'baseline',option))
            for name in ('trace.pt','initial.pt','contacts.npy','physics_states.pt','physical_metadata.json','contact_frames.json'):
                hashes[str(path/name)]=sha(path/name)
        packets[scene]=dict(features=np.stack([r['features'] for r in data]),labels=np.stack([r['labels'] for r in data]),motion=data[0]['motion'],common_dim=data[0]['common_dim'],effect_dim=data[0]['effect_dim'],interaction_dim=data[0]['interaction_dim'],raw_contacts=sum(r['raw_contacts'] for r in data),prefix_exact=True)
    torch.save(packets,fit/'feature_packet.pt')
    train=packets['train'];x=torch.from_numpy(train['features'].reshape(-1,train['features'].shape[-1])).to('cuda');y=torch.from_numpy(train['labels'].reshape(-1)).to('cuda')
    mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(.01)
    z=((x-mean)/std).clamp(-10,10)
    ev=torch.from_numpy(packets['eval']['features'].reshape(-1,x.shape[-1])).to('cuda');ev=((ev-mean)/std).clamp(-10,10)
    common,effect=train['common_dim'],train['effect_dim']
    torch.manual_seed(806);model=OracleQ(x.shape[1]);initial=copy.deepcopy(model.state_dict());models={};losses={};scores={};chosen={};schedules=[]
    rng=np.random.RandomState(807);schedule=rng.randint(0,len(x),size=(1500,128));np.save(fit/'minibatches.npy',schedule)
    for arm in ARMS:
        network=OracleQ(x.shape[1]);network.load_state_dict(initial);network=network.to('cuda')
        optimizer=torch.optim.AdamW(network.parameters(),lr=3e-4,weight_decay=1e-4)
        inputs=masked(z,arm,common,effect);loss_history=[]
        for step,indices in enumerate(schedule):
            ids=torch.from_numpy(indices).to('cuda');optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(network(inputs[ids]),y[ids])
            if not torch.isfinite(loss):raise ValueError('finite Q loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.);optimizer.step();loss_history.append(float(loss))
        network.eval()
        with torch.no_grad():pred=network(masked(ev,arm,common,effect)).sigmoid().cpu().numpy().reshape(8,-1)
        choice=pred.argmax(0);np.save(fit/(arm+'_choices.npy'),choice);np.save(fit/(arm+'_options.npy'),options[choice])
        scores[arm]=pred;chosen[arm]=choice;models[arm]=dict(model={k:v.cpu() for k,v in network.state_dict().items()},optimizer=optimizer.state_dict(),updates=1500)
        losses[arm]=loss_history
        print(json.dumps(dict(arm=arm,updates=1500,train_last_loss=loss_history[-1],choices=np.bincount(choice,minlength=8).tolist())),flush=True)
        del network,optimizer,inputs
    torch.save(dict(models=models,initial=initial,mean=mean.cpu(),std=std.cpu(),common_dim=common,effect_dim=effect,input_dim=x.shape[1],scores=scores,choices=chosen,losses=losses,updates=6000,seed=806),fit/'selector.pt')
    result=dict(run_status='COMPLETED',actual_optimizer_steps=6000,cm_updates=0,actor_updates=0,input_dim=x.shape[1],parameters=sum(v.numel() for v in initial.values()),train_examples=len(x),eval_examples=len(ev),arms=list(ARMS),shared_initialization=True,shared_minibatches=True,training_normalization_only=True,selection_has_no_terminal_label_input=True,input_sha256=hashes,wall_seconds=time.monotonic()-begin)
    (fit/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
