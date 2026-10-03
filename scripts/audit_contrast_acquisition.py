"""Independent NumPy inference, ranks, raw targets, risk and early AdamW."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
BASE=ROOT/'src/task/CmResidual/research/contact_response/output'


def forward(state,x):
    activations=[x]
    for i in (0,2,4):
        x=x@state[str(i)+'.weight'].T+state[str(i)+'.bias']
        if i<4:x=np.maximum(x,0);activations.append(x)
    return x,activations


def all_candidates(cp,x):
    s={k:v.numpy().astype(np.float64) for k,v in cp['state'].items()};values=[]
    for arm in range(7):
        onehot=np.zeros((len(x),6));
        if arm:onehot[:,arm-1]=1
        xx=(np.c_[x,onehot]-cp['mean'].numpy())/cp['std'].numpy()
        v,_=forward(s,xx);values.append(v*cp['target_std'].numpy()+cp['target_mean'].numpy())
    return np.stack(values,1)


def replay(cp,x,y):
    state={k:v.numpy().astype(np.float64).copy() for k,v in cp['initial'].items()};m={k:np.zeros_like(v) for k,v in state.items()};v={k:np.zeros_like(v) for k,v in state.items()};maximum=loss_max=0.
    for step in range(3):
        ids=cp['schedule'][step].numpy();xx=(x[ids]-cp['mean'].numpy())/cp['std'].numpy();yy=(y[ids]-cp['target_mean'].numpy())/cp['target_std'].numpy()
        pred,acts=forward(state,xx);loss=float(((pred-yy)**2).mean());loss_max=max(loss_max,abs(loss-cp['losses'][step]));delta=2*(pred-yy)/pred.size;grad={}
        for i,activation in ((4,acts[2]),(2,acts[1]),(0,acts[0])):
            key=str(i);grad[key+'.weight']=delta.T@activation;grad[key+'.bias']=delta.sum(0)
            if i:delta=(delta@state[key+'.weight'])*(activation>0)
        for key in state:
            m[key]=.9*m[key]+.1*grad[key];v[key]=.999*v[key]+.001*grad[key]**2
            state[key]=state[key]*(1-.001*.0001)-.001*(m[key]/(1-.9**(step+1)))/(np.sqrt(v[key]/(1-.999**(step+1)))+1e-8)
            maximum=max(maximum,float(np.abs(state[key]-cp['early_states'][step][key].numpy()).max()))
    assert maximum<5e-5 and loss_max<2e-5,(maximum,loss_max)
    return maximum,loss_max


def regroup(rows,sk,ak):
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r[sk],r['environment']),[]).append(i)
    keys=sorted(groups);ids=np.array([sorted(groups[k],key=lambda i:rows[i][ak]) for k in keys]);strata=[(k[0],rows[groups[k][0]]['motion']) for k in keys]
    assert ids.shape==(1536,2)
    return ids,strata


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();fit=a.root/'fit';torch.set_num_threads(2)
    source=BASE/'P-20261001-direct-randomized-response-fit-r1';test_source=BASE/'P-20261001-direct-randomized-response-test-r1'
    raw=torch.load(source/'fit_data.pt',map_location='cpu',weights_only=False);test=torch.load(test_source/'predictions.pt',map_location='cpu',weights_only=False)
    # Rebuild all raw response and pseudo labels. Contexts inherit prior full
    # geometry audit by protected byte identity, not a new FK reconstruction.
    rawmap={};testmap={}
    for target,folder,seeds in ((rawmap,BASE/'P-20261001-randomized-effect-risk-r2',(492,493)),(testmap,test_source,(494,495))):
        for actor in (286,287):
            for seed in seeds:
                packet=torch.load(folder/f't{actor}_s{seed}/windows.pt',map_location='cpu',weights_only=False)
                assert packet['complete'].all()
                y=packet['next_state'][:,36:39].numpy()-packet['state'][:,36:39].numpy()-packet['state'][:,43:46].numpy()/np.float32(30)
                arm=packet['arm'].numpy();prop=packet['propensity'].numpy();z=np.zeros((768,3,3))
                for j in range(3):z[:,:,j]=y.astype(np.float64)*((arm==2*j+1)/prop[:,2*j+1].astype(np.float64)-(arm==2*j+2)/prop[:,2*j+2].astype(np.float64))[:,None]
                for i,e in enumerate(packet['environment'].numpy()):target[(actor,seed,int(e))]=(y[i],arm[i],prop[i],z[i])
    for i,r in enumerate(raw['rows']):
        y,arm,prop,_=rawmap[(r['actor'],r['acquisition_seed'],r['environment'])]
        assert np.array_equal(y,raw['y'][i].numpy()) and arm==raw['arm'][i] and np.array_equal(prop,raw['propensity'][i].numpy())
    pseudo=[]
    for r in test['rows']:pseudo.append(testmap[(r['training_seed'],r['evaluation_seed'],r['environment'])][3])
    pseudo=np.array(pseudo);np.testing.assert_allclose(pseudo,test['pseudo_contrast'].numpy(),rtol=0,atol=1e-12)
    blocks,strata=regroup(raw['rows'],'acquisition_seed','actor');quotas=(43,43,43,43,42,42);rng=np.random.default_rng(4500);initial=[]
    for s,q in zip(sorted(set(strata)),quotas):initial.extend(rng.choice([i for i,t in enumerate(strata) if t==s],q,replace=False))
    initial=np.sort(initial);initial_rows=blocks[initial].ravel();selection=json.loads((fit/'selection.json').read_text())
    assert np.array_equal(initial,selection['initial_blocks']) and np.array_equal(blocks,selection['block_rows']) and np.array_equal(initial_rows,selection['initial_rows'])
    x=raw['x'].numpy().astype(np.float64);arms=raw['arm'].numpy();onehot=np.eye(7)[arms][:,1:];xx=np.c_[x,onehot];y=raw['y'].numpy().astype(np.float64)
    saved=np.load(fit/'selection_arrays.npz');ensemble=[];forward_max=update_max=loss_max=0.;cps={}
    for k in range(3):
        cp=torch.load(fit/f'ensemble{k}.pt',map_location='cpu',weights_only=False);cps['ensemble'+str(k)]=cp;rng=np.random.default_rng(4510+k);b=[]
        for s,q in zip(sorted(set(strata)),quotas):b.extend(rng.choice([i for i in initial if strata[i]==s],q,replace=True))
        ids=blocks[np.array(b)].ravel();assert np.array_equal(ids,cp['row_ids']) and np.array_equal(ids,saved['bootstrap_rows'][k])
        schedule=torch.randint(512,(600,128),generator=torch.Generator().manual_seed(4521+k));assert torch.equal(schedule,cp['schedule'])
        for key in cp['initial']:assert torch.equal(cp['initial'][key],cps['ensemble0']['initial'][key])
        v=all_candidates(cp,x);forward_max=max(forward_max,float(np.abs(v-saved['ensemble_predictions'][k]).max()));ensemble.append(v)
        e,l=replay(cp,xx[ids],y[ids]);update_max=max(update_max,e);loss_max=max(loss_max,l)
        for name,expected in [('mean',xx[initial_rows].mean(0)),('std',xx[initial_rows].std(0).clip(1e-5)),('target_mean',y[initial_rows].mean(0)),('target_std',y[initial_rows].std(0).clip(1e-6))]:np.testing.assert_allclose(cp[name].numpy(),expected,atol=3e-6,rtol=2e-5)
    e=np.array(ensemble);diff=e[:,:,1::2]-e[:,:,2::2];rs={'contrast':diff.var(0).sum((1,2)),'absolute':e.var(0).sum((1,2))};bs={k:v[blocks].mean(1) for k,v in rs.items()}
    choices={};rng=np.random.default_rng(4530)
    for name in ('contrast','absolute','uniform'):
        b=[]
        for s,q in zip(sorted(set(strata)),quotas):
            pool=np.array([i for i,t in enumerate(strata) if t==s and i not in initial])
            chosen=rng.choice(pool,q,replace=False) if name=='uniform' else pool[np.lexsort((pool,-bs[name][pool]))[:q]]
            b.extend(chosen)
        choices[name]=np.sort(b);assert np.array_equal(choices[name],selection['acquired_blocks'][name]),'independent acquisition ranks differ'
    held=np.load(fit/'held.npz');d={}
    for name in ('contrast','absolute','uniform'):
        cp=torch.load(fit/(name+'.pt'),map_location='cpu',weights_only=False);cps[name]=cp;ids=np.r_[initial_rows,blocks[choices[name]].ravel()]
        assert len(np.unique(ids))==1024 and np.array_equal(ids,cp['row_ids'])
        schedule=torch.randint(1024,(1000,128),generator=torch.Generator().manual_seed(4541));assert torch.equal(schedule,cp['schedule'])
        v=all_candidates(cp,test['context'].numpy());d[name]=(v[:,1::2]-v[:,2::2]).transpose(0,2,1);forward_max=max(forward_max,float(np.abs(d[name]-held[name]).max()))
        e,l=replay(cp,xx[ids],y[ids]);update_max=max(update_max,e);loss_max=max(loss_max,l)
        for key in cp['initial']:assert torch.equal(cp['initial'][key],cps['contrast']['initial'][key])
        for key in ('mean','std','target_mean','target_std'):assert torch.equal(cp[key],cps['ensemble0'][key])
    assert forward_max<2e-6,forward_max
    # Verify saved predictions and exact statistics separately: independent
    # forward roundoff must not feed a relaxed scientific classification.
    d={k:held[k] for k in ('contrast','absolute','uniform','zero')};risk=np.stack([((d['contrast']**2-d[c]**2)-2*(d['contrast']-d[c])*pseudo).sum((1,2))*1e6 for c in ('uniform','absolute','zero')],-1)
    np.testing.assert_allclose(risk,held['risk'],atol=1e-10,rtol=1e-12)
    ids,ss=regroup(test['rows'],'evaluation_seed','training_seed');rng=np.random.default_rng(4550);boot=np.zeros((2000,3))
    for s in sorted(set(ss)):
        packet=ids[np.array([t==s for t in ss])];draw=rng.integers(len(packet),size=(2000,len(packet)));boot+=risk[packet][draw].sum((1,2))/3072
    np.testing.assert_allclose(boot,held['bootstrap'],atol=1e-10,rtol=1e-12);point=risk.mean(0);upper=np.quantile(boot,.95,axis=0)
    both=all(risk[[i for i,r in enumerate(test['rows']) if r['evaluation_seed']==s],0].mean()<0 for s in (494,495))
    gates=dict(uniform_margin=bool(point[0]<=-1),uniform_upper=bool(upper[0]<0),absolute_margin=bool(point[1]<=-.5),absolute_upper=bool(upper[1]<0),both_test_seeds=bool(both),better_zero=bool(point[2]<0))
    r=json.loads((fit/'results.json').read_text());assert r['gates']==gates and r['label']==('PROMISING' if all(gates.values()) else 'UNPROMISING')
    for j,c in enumerate(('uniform','absolute','zero')):
        np.testing.assert_allclose(r['relative_risk_mm2'][c]['point'],point[j],atol=1e-10)
        np.testing.assert_allclose(r['relative_risk_mm2'][c]['upper95'],upper[j],atol=1e-10)
    report=dict(run_status='COMPLETED',label=r['label'],raw_fit_rows=3072,raw_test_rows=3072,inherited_context_geometry_audit=True,
                forward_max_m=forward_max,early_adamw_parameter_max=update_max,early_loss_max=loss_max,independent_updates_replayed=18,remaining_updates_not_replayed=4782,
                all_ranks_selections_and_paired_risk_verified=True)
    (a.root/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
