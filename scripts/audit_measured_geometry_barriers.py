"""Held-out geometry and clock audit plus matched feature/statistical replay.

Pre-fit limits: independent transforms after FP32 cast,3e-6 absolute;
barrier labels,3u*abs(label)+1e-9 (u=2^-24). No fitted frame correction.
This is an exploratory input/label/result audit; native and fit's fixed sampled
NumPy forward checks are retained, not replaced by a claim of full optimizer replay.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.fit_measured_geometry_barriers import read_panel,features,VARIANTS

def matrices(q):
    q=torch.as_tensor(q,dtype=torch.float64);q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True);x,y,z,w=q.unbind(-1);m=torch.empty((*q.shape[:-1],3,3),dtype=torch.float64)
    m[...,0,0]=1-2*(y*y+z*z);m[...,1,1]=1-2*(x*x+z*z);m[...,2,2]=1-2*(x*x+y*y)
    m[...,0,1]=2*(x*y-z*w);m[...,1,0]=2*(x*y+z*w);m[...,0,2]=2*(x*z+y*w);m[...,2,0]=2*(x*z-y*w);m[...,1,2]=2*(y*z-x*w);m[...,2,1]=2*(y*z+x*w)
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.directory.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists();out.mkdir();begin=time.monotonic();torch.set_num_threads(2);u=2**-24
    m=json.loads((run/'run_manifest.json').read_text());r=json.loads((run/'fit/results.json').read_text());assert m['run_status']==r['run_status']=='COMPLETED'
    for f,h in {**m['input_sha256'],**r['source_sha256']}.items():assert sha(Path(f))==h,f
    if r['label']=='UNCLEAR':
        (out/'results.json').write_text(json.dumps(dict(run_status='COMPLETED',label='UNCLEAR',original_stop_preserved=True))+'\n');return
    cp=torch.load(run/'fit/barrier_models.pt',map_location='cpu',weights_only=False);packet=torch.load(run/'fit/test_predictions.pt',map_location='cpu',weights_only=False);assert cp['updates_each']==1500 and sha(run/'fit/barrier_models.pt')==r['model_sha256'] and sha(run/'fit/test_predictions.pt')==r['predictions_sha256']
    sources={};parts=[read_panel(run/f's{k}',sources) for k in (578,579)];fit={key:np.concatenate([x[key] for x in parts]) for key in ['state','extra','prior','action','y','cells']};del parts
    em=fit['extra'].mean(0);es=fit['extra'].std(0).clip(.001);residual=fit['y']-fit['prior'];ym=residual.mean(0);ys=residual.std(0).clip(.001)
    for key,value in [('extra_mean',em),('extra_std',es),('residual_mean',ym),('residual_std',ys)]:assert np.array_equal(cp[key].numpy(),value),key
    st=np.zeros((12,70),np.float32);et=np.zeros((12,80),np.float32);counts=np.bincount(fit['cells'],minlength=12)
    for cell in range(12):
        if counts[cell]:st[cell]=fit['state'][fit['cells']==cell].mean(0);et[cell]=fit['extra'][fit['cells']==cell].mean(0)
    assert np.array_equal(st,cp['global_templates'].numpy()) and np.array_equal(et,cp['extra_templates'].numpy()) and counts.tolist()==cp['fit_cell_counts'];del fit,residual
    data={k:v.numpy() for k,v in packet['data'].items()};raw=run/'s580';i=torch.load(raw/'initial.pt',map_location='cpu',weights_only=False);t=torch.load(raw/'trace.pt',map_location='cpu',weights_only=False);meta=json.loads((raw/'physical_metadata.json').read_text())
    groups=i['policy_group'].numpy();motion=i['motion'].numpy();stop=i['phase_stop'].numpy()[motion];progress=t['progress'].numpy();mask=(progress>=stop[None]-74)&(progress<=stop[None]+30)&(groups[None]>0);mask[:2]=False;tick,env=np.where(mask)
    assert np.array_equal(tick,data['tick']) and np.array_equal(env,data['environment']);post=t['object_root'];current=post[tick-1,env].double();previous=post[tick-2,env].double();n=len(tick);assert torch.equal(current.float(),t['context'][tick,env,36:49])
    inv=matrices(current[:,3:7]).transpose(-1,-2);previous_inv=matrices(previous[:,3:7]).transpose(-1,-2)
    body=t['hand_body_position'][tick-1,env].double();prev_body=t['hand_body_position'][tick-2,env].double();pos=(inv[:,None]@(body-current[:,None,:3])[...,None]).squeeze(-1);oldpos=(previous_inv[:,None]@(prev_body-previous[:,None,:3])[...,None]).squeeze(-1);rel=inv[:,None]@matrices(t['hand_body_quaternion'][tick-1,env])
    force=torch.cat([t['object_force'][tick-1,env,None],t['hand_force'][tick-1,env]],1).double();weight=torch.tensor([v['mass'] for v in meta['object_body_properties']],dtype=torch.float64)*np.linalg.norm(meta['gravity']);normalized=(inv[:,None]@force[...,None]).squeeze(-1)/weight[env,None,None]
    clear=t['clearance'].double();prior=torch.stack([(current[:,2]-previous[:,2]),(clear[tick-1,env]-clear[tick-2,env])],-1)/.005;y=torch.stack([(post[tick,env,2].double()-current[:,2]),(clear[tick,env]-clear[tick-1,env])],-1)/.005
    extra=torch.cat([pos.reshape(n,15),rel[...,:2].reshape(n,30),normalized.reshape(n,18),prior,(pos-oldpos).reshape(n,15)],-1).float().numpy();err=np.abs(extra-data['extra']);assert np.all(err<=3e-6+3*u*np.abs(extra)),'actual observed geometry input reconstruction'
    for name,value in [('y',y),('prior',prior)]:assert np.all(np.abs(data[name]-value.numpy())<=3*u*np.abs(value.numpy())+1e-9),name
    for variant in VARIANTS:assert np.array_equal(features(data,em,es,st,et,variant),packet['features'][variant].numpy())
    stream=np.full(768,-1,int)
    for group in (1,2,3):stream[np.flatnonzero(groups==group)]=np.arange(192)
    assert np.array_equal(data['cluster'],stream[env]);pred={k:v.numpy().astype(np.float64) for k,v in packet['predictions'].items()};assert np.array_equal(pred['physical_persistence'],data['prior'].astype(np.float64))
    names=VARIANTS+['physical_persistence'];ep=np.zeros((192,6));np.add.at(ep[:,0],data['cluster'],1)
    for column,name in enumerate(names,1):np.add.at(ep[:,column],data['cluster'],np.sum((pred[name]-data['y'].astype(np.float64))**2,axis=1)/2)
    assert np.max(np.abs(ep-np.array(json.loads((run/'fit/episode_errors.json').read_text()))))<1e-9
    rates=ep[:,1:].sum(0)/ep[:,0].sum();rng=np.random.RandomState(3213);boot=ep[rng.randint(0,192,(2000,192))].sum(1);values=boot[:,1:]/boot[:,0,None];ci={name:np.quantile(values[:,0]-values[:,j],[.025,.975]).tolist() for j,name in enumerate(names[1:],1)};gates={name:dict(gain1percent=bool(rates[0]<=.99*rates[j]),paired_upper95_negative=bool(ci[name][1]<0)) for j,name in enumerate(names[1:],1)};label='PROMISING' if all(all(v.values()) for v in gates.values()) else 'UNPROMISING'
    assert r['label']==label and r['gates']==gates and all(abs(r['mse'][name]-rates[j])<1e-12 for j,name in enumerate(names)) and all(np.max(np.abs(np.array(r['paired_difference_interval95'][name])-ci[name]))<1e-12 for name in ci)
    result=dict(run_status='COMPLETED',label=label,all_test_geometry_independent_torch_reconstruction=True,maximum_geometry_rounding_error=float(err.max()),all_observed_previous_tick_forces_and_flow=True,all_barrier_labels_independently_reconstructed=True,all_fit_statistics_templates_exact=True,fit_decoding_reuses_primary_reader=True,all_control_features_exact=True,all_paired_risk_cluster_gates_rebuilt=True,optimizer_not_replayed=True,model_forward_scope='primary fixed512sample NumPy checks, no additional model replay',source_manifest_sha256=sha(run/'run_manifest.json'),source_result_sha256=sha(run/'fit/results.json'),script_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-begin,cpu_reason='independent observed geometry/label transforms, file/statistical checks; no fitting',policy_utility_unproved=True,formal_validation=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
