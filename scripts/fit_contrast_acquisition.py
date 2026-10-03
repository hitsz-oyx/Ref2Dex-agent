"""GPU fixed-label-budget physical acquisition; no simulator/actor updates."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.task.CmResidual.contrast_acquisition import NAMES,QUOTAS,blocks,initial_blocks,acquire,scores,initialized,metrics
BASE=ROOT/'src/task/CmResidual/research/contact_response/output'


def arm_input(x,arm):return torch.cat((x,torch.nn.functional.one_hot(arm,7)[:,1:].to(x.dtype)),-1)


def fit(x,y,mean,std,ym,ys,seed,schedule_seed,updates):
    net=initialized(seed).cuda();initial={k:v.cpu().clone() for k,v in net.state_dict().items()}
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001,foreach=False)
    generator=torch.Generator(device='cpu').manual_seed(schedule_seed)
    schedule=torch.randint(len(x),(updates,128),generator=generator)
    xx=(x-mean)/std;yy=(y-ym)/ys;early=[];losses=[]
    for step,ids in enumerate(schedule):
        ids=ids.cuda();loss=(net(xx[ids])-yy[ids]).square().mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step<3:early.append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()});losses.append(float(loss.detach()))
    cp=dict(initial=initial,state={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),schedule=schedule,early_states=early,losses=losses,
            mean=mean.cpu(),std=std.cpu(),target_mean=ym.cpu(),target_std=ys.cpu(),seed=seed,schedule_seed=schedule_seed,updates=updates)
    return net,cp


@torch.no_grad()
def candidates(net,x,mean,std,ym,ys):
    values=[]
    for start in range(0,len(x),256):
        packet=x[start:start+256];p=[]
        for arm in range(7):
            labels=torch.full((len(packet),),arm,device=x.device,dtype=torch.long)
            p.append((net((arm_input(packet,labels)-mean)/std)*ys+ym).cpu().numpy())
        values.append(np.stack(p,1))
    return np.concatenate(values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'fit';assert torch.cuda.is_available() and not out.exists();out.mkdir()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=BASE/'P-20261001-direct-randomized-response-fit-r1';raw=torch.load(old/'fit_data.pt',map_location='cpu',weights_only=False)
    assert raw['x'].shape==(3072,81);block,strata=blocks(raw['rows'],'acquisition_seed','actor');initial=initial_blocks(strata);initial_rows=block[initial].ravel()
    x=raw['x'].cuda();arm=raw['arm'].cuda();initial_x=arm_input(x[initial_rows],arm[initial_rows]);initial_y=raw['y'][initial_rows].cuda()
    mean=initial_x.mean(0);std=initial_x.std(0,unbiased=False).clamp_min(1e-5);ym=initial_y.mean(0);ys=initial_y.std(0,unbiased=False).clamp_min(1e-6)
    ensemble=[];bootstrap_indices=[]
    for k in range(3):
        rng=np.random.default_rng(4510+k);ids=[]
        for s,q in zip(sorted(set(strata)),QUOTAS):
            b=np.array([i for i in initial if strata[i]==s]);assert len(b)==q;ids.extend(rng.choice(b,q,replace=True))
        row_ids=block[np.array(ids)].ravel();bootstrap_indices.append(row_ids)
        net,cp=fit(arm_input(x[row_ids],arm[row_ids]),raw['y'][row_ids].cuda(),mean,std,ym,ys,4520,4521+k,600)
        cp['row_ids']=row_ids;torch.save(cp,out/('ensemble'+str(k)+'.pt'))
        ensemble.append(candidates(net,x,mean,std,ym,ys));print(json.dumps(dict(ensemble=k,updates=600)),flush=True)
    predictions=np.stack(ensemble).astype(np.float64);row_scores=scores(predictions)
    block_scores={k:v[block].mean(1) for k,v in row_scores.items()};selected=acquire(strata,initial,block_scores)
    record=dict(initial_blocks=initial.tolist(),acquired_blocks={k:v.tolist() for k,v in selected.items()},block_rows=block.tolist(),strata=strata,
                source_labels_not_used_for_scores=True,initial_rows=initial_rows.tolist())
    # This artifact is committed before additional outcome labels are indexed.
    (out/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savez(out/'selection_arrays.npz',ensemble_predictions=predictions,contrast_scores=block_scores['contrast'],absolute_scores=block_scores['absolute'],bootstrap_rows=np.stack(bootstrap_indices))
    test=torch.load(BASE/'P-20261001-direct-randomized-response-test-r1/predictions.pt',map_location='cpu',weights_only=False)
    xt=test['context'].cuda();held={};final_rows={}
    for name in NAMES:
        rows=np.r_[initial_rows,block[selected[name]].ravel()];assert len(rows)==1024 and len(np.unique(rows))==1024;final_rows[name]=rows
        net,cp=fit(arm_input(x[rows],arm[rows]),raw['y'][rows].cuda(),mean,std,ym,ys,4540,4541,1000)
        cp['row_ids']=rows;torch.save(cp,out/(name+'.pt'))
        v=candidates(net,xt,mean,std,ym,ys).astype(np.float64);held[name]=(v[:,1::2]-v[:,2::2]).transpose(0,2,1)
        print(json.dumps(dict(final=name,labels=1024,updates=1000)),flush=True)
    held['zero']=np.zeros_like(held['uniform']);pseudo=test['pseudo_contrast'].numpy()
    result,risk,boot=metrics(held,pseudo,test['rows'])
    result.update(new_optimizer_updates=4800,new_native_ticks=0,initial_labels=512,additional_labels_each=512,test_rows=3072,
                  overlap_blocks={a+':'+b:int(len(np.intersect1d(selected[a],selected[b]))) for a,b in (('contrast','absolute'),('contrast','uniform'),('absolute','uniform'))},
                  previously_viewed_test=True,legacy_native_physics=True,not_online_interaction_savings=True)
    np.savez(out/'held.npz',**held,pseudo=pseudo,risk=risk,bootstrap=boot)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
