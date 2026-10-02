"""Frozen Gaussian moment field; fresh-test paired risk, never policy selection."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha

def network(outputs,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=torch.nn.Sequential(torch.nn.Linear(70,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,outputs))
        torch.nn.init.zeros_(model[-1].weight);torch.nn.init.zeros_(model[-1].bias)
    return model

def cell_ids(progress,motion,lift,stop):
    phase=np.where(progress<lift,0,np.where(progress<stop-74,1,np.where(progress<=stop,2,3)))
    return motion*4+phase

def paired_risk(state,global_response,moment):
    return ((state*state-global_response*global_response)-2*(state-global_response)*moment).mean(-1)

def load_panel(path,sources):
    result=json.loads((path/'results.json').read_text());audit=json.loads((path/'panel_audit.json').read_text())
    assert result['run_status']==audit['run_status']=='COMPLETED' and result['continuous_updates']==0 and not result['deterministic']
    for f in ('initial.pt','trace.pt','results.json','panel_audit.json'):sources[str((path/f).resolve())]=sha(path/f)
    assert sha(path/'initial.pt')==result['initial_sha256'] and sha(path/'trace.pt')==result['trace_sha256']
    initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(path/'trace.pt',map_location='cpu',weights_only=False)
    group=initial['policy_group'].numpy();motion=initial['motion'].numpy();progress=trace['progress'].numpy()
    stop=initial['phase_stop'].numpy()[motion];lift=initial['lift_start'].numpy()[motion]
    eligible=(progress>=stop[None]-74)&(progress<=stop[None]+30)
    assert np.all(eligible.sum(0)==105) and np.array_equal(trace['request_mean'].numpy(),np.zeros((202,768,12),np.float32))
    eps=trace['request_noise'].numpy();request=trace['request'].numpy()
    assert np.max(np.abs(request[:,group>0]-eps[:,group>0]*.05))<2e-6
    pre=torch.cat((initial['object_root'][None],trace['object_root'][:-1]),0).numpy();post=trace['object_root'].numpy()
    y=np.concatenate((post[...,:3]-pre[...,:3],post[...,7:10]-pre[...,7:10]),-1)/np.array([.005]*3+[.05]*3,np.float32)
    assert np.max(np.abs(y-trace['physical_transition'].numpy()))<1e-5
    cells=cell_ids(progress,motion[None],lift[None],stop[None]);x=trace['normalized_context'].numpy()
    packets=[]
    for g in range(4):
        ids=np.flatnonzero(group==g);assert len(ids)==192
        # Episode first; preserve matching private stream indices across replicas.
        keep=eligible[:,ids].T
        def select(a):return a[:,ids].swapaxes(0,1)[keep]
        packets.append(dict(x=select(x),y=select(y),eps=select(eps),cells=select(cells),episode=np.repeat(np.arange(192),105)))
    assert np.array_equal(packets[1]['eps'],packets[2]['eps']) and np.array_equal(packets[1]['eps'],packets[3]['eps'])
    return packets

def np_forward(x,state):
    x=x.astype(np.float64)
    for layer in (0,2,4):
        x=x@state[str(layer)+'.weight'].numpy().astype(np.float64).T+state[str(layer)+'.bias'].numpy().astype(np.float64)
        if layer!=4:x=np.maximum(x,0)
    return x

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert ROOT in out.parents and not out.exists();out.mkdir();torch.set_num_threads(2);assert torch.cuda.is_available()
    begin=time.monotonic();sources={str(Path(__file__)):sha(Path(__file__))};fit=load_panel(a.source/'s547',sources);test=load_panel(a.test/'s570',sources)
    for pth in [ROOT/'docs/experiments/probes/P-20261002-state-response-field.md',a.source/'u00/policy_heads.pt']:
        sources[str(pth.resolve())]=sha(pth)
    def tensor(x):return torch.from_numpy(x).to('cuda')
    def predict(model,x):
        with torch.no_grad():return np.concatenate([model(tensor(block)).cpu().numpy() for block in np.array_split(x,max(1,(len(x)+4095)//4096))])
    def train(model,x,y,seed,name):
        model=model.cuda();tx=tensor(x);ty=tensor(y);rng=torch.Generator(device='cuda').manual_seed(seed);opt=torch.optim.Adam(model.parameters(),lr=3e-4)
        for step in range(2000):
            if time.monotonic()-begin>3300:raise TimeoutError('fixed field-fit budget')
            ids=torch.randint(len(tx),(4096,),device='cuda',generator=rng)
            loss=(model(tx[ids])-ty[ids]).square().mean();assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%500==499:print(json.dumps(dict(model=name,step=step+1,loss=float(loss.detach()))),flush=True)
        return model
    b=train(network(6,1081),fit[0]['x'],fit[0]['y'],1083,'baseline')
    x=np.concatenate([d['x'] for d in fit[1:]]);y=np.concatenate([d['y'] for d in fit[1:]]);eps=np.concatenate([d['eps'] for d in fit[1:]]);cells=np.concatenate([d['cells'] for d in fit[1:]])
    residual=y-predict(b,x);moments=(eps[:,:,None]*residual[:,None,:]).reshape(-1,72)
    global_field=np.zeros((12,72),np.float32);cell_counts=np.bincount(cells,minlength=12)
    for c in range(12):
        if cell_counts[c]:global_field[c]=moments[cells==c].mean(0)
    energy=float(np.mean(global_field[cells].astype(np.float64)**2));g=train(network(72,1082),x,moments,1084,'response_field')
    state={name:{k:v.detach().cpu() for k,v in model.state_dict().items()} for name,model in [('baseline',b),('field',g)]}
    torch.save(dict(schema='gaussian-request-response-field70x12x6-v1',models=state,global_field=torch.from_numpy(global_field),source_sha256=sources,baseline_steps=2000,field_steps=2000,behavior_mean=0.,behavior_std=.05),out/'response_field.pt')
    errors={};packet={};joint_episode=np.zeros((192,2),np.float64)
    for replica,d in enumerate(test[1:]):
        baseline=predict(b,d['x']);response=predict(g,d['x']);target=(d['eps'][:,:,None]*(d['y']-baseline)[:,None,:]).reshape(-1,72);control=global_field[d['cells']]
        risk=paired_risk(response.astype(np.float64),control.astype(np.float64),target.astype(np.float64));assert np.isfinite(risk).all()
        joint_episode[:,0]+=risk.reshape(192,105).sum(1);joint_episode[:,1]+=105
        sample=np.arange(0,len(d['x']),max(1,len(d['x'])//256))[:256]
        for name,value in [('baseline',baseline),('field',response)]:
            error=float(np.abs(np_forward(d['x'][sample],state[name])-value[sample]).max());errors[name]=max(errors.get(name,0.),error);assert error<=2e-5
        packet[str(replica)]=dict(x=torch.from_numpy(d['x']),y=torch.from_numpy(d['y']),eps=torch.from_numpy(d['eps']),cells=torch.from_numpy(d['cells']),baseline=torch.from_numpy(baseline),field=torch.from_numpy(response))
    torch.save(packet,out/'test_predictions.pt');point=float(joint_episode[:,0].sum()/joint_episode[:,1].sum())
    rng=np.random.RandomState(1085);boot=joint_episode[rng.randint(0,192,size=(2000,192))].sum(1);ci=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist()
    gates=dict(nonzero_fit_global_energy=energy>0.,risk_gain_at_least_ten_percent=point<=-.1*energy,paired_upper95_negative=ci[1]<0.)
    for path,h in sources.items():assert sha(Path(path))==h,path
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',paired_risk_difference=point,interval95=ci,fit_global_response_energy=energy,gain_over_fit_global_energy=-point/energy if energy else None,gates=gates,fit_drift_transitions=20160,fit_response_transitions=60480,test_response_transitions=60480,test_common_stream_episode_clusters=192,fit_cell_counts=cell_counts.tolist(),independent_numpy_forward_max_error=errors,source_sha256=sources,checkpoint_sha256=sha(out/'response_field.pt'),predictions_sha256=sha(out/'test_predictions.pt'),wall_seconds=time.monotonic()-begin,no_policy_updates=True,cm_policy_utility_unproved=True,formal_validation=False)
    (out/'episode_risks.json').write_text(json.dumps(joint_episode.tolist())+'\n');(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)

if __name__=='__main__':main()
