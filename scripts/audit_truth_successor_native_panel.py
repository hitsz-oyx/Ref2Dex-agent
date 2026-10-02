#!/usr/bin/env python3
"""Independent native continuous request, causal critic, PD and geometry replay."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scripts.analyze_static_hold_feasibility import rotation
from scripts.run_contact_response_probe import sha

PRIMITIVES=np.array([[0,0,0],[0,0,-.30],[0,0,-.15],[0,0,.15],[-.01,0,0],[.01,0,0],[0,-.01,0],[0,.01,0]],dtype=np.float32)
COUPLING={6:((7,1.05),),8:((9,1.05),),10:((11,1.05),),12:((13,1.05),),15:((16,.6),(17,.8))}

def couple(goal,lo,hi):
    for parent,children in COUPLING.items():
        bottom=max(lo[parent],*(lo[j]/ratio for j,ratio in children));top=min(hi[parent],*(hi[j]/ratio for j,ratio in children))
        goal[...,parent]=np.clip(goal[...,parent],bottom,top)
        for child,ratio in children:goal[...,child]=goal[...,parent]*ratio
    return goal

def numpy_features(context,initial_height,clearance):
    q=context[:,:18];dq=context[:,18:36];obj=context[:,36:49]
    return np.concatenate((q[:,3:],dq,obj[:,:3]-q[:,:3],obj[:,3:7],obj[:,7:10]-dq[:,:3],obj[:,10:13],context[:,49:51],
        context[:,51:69]-q,context[:,69:70],(obj[:,2]-initial_height)[:,None],clearance[:,None]),-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--panel',type=int,required=True);a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root/'run_manifest.json').read_text())
    if a.panel!=595 or m['experiment_id']!='P-20261002-truth-successor-task-value':raise ValueError('fixed collection cohorts')
    panel_record=m['panel_checkpoints'][str(a.panel)];checkpoint_path=Path(panel_record['path'])
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if checkpoint['updates']!=panel_record['update'] or sha(checkpoint_path)!=panel_record['sha256']:raise ValueError('initial heads only')
    base_checkpoint=torch.load(Path(m['base_checkpoint']),map_location='cpu',weights_only=False)
    maximum=dict(target=0.,action=0.,mean=0.,logprob=0.,value=0.,prediction=0.,physical_target=0.)
    independent=np.array([0,1,2,3,4,5,6,8,10,12,14,15]);scale=np.array([.02]*3+[.10]*3+[.15]*6,dtype=np.float32)
    def layer(x,state,key):return x@state[key+'.weight'].numpy().astype(np.float64).T+state[key+'.bias'].numpy().astype(np.float64)
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects'
    def vertices(f):return np.array([list(map(float,l.split()[1:4])) for l in f.read_text().splitlines() if l.startswith('v ')])
    full=vertices(asset/'airplane/airplane.obj');hull=ConvexHull(full)
    # Linear support extrema lie on the convex hull; validate the hull against ALL mesh vertices.
    containment=float((full@hull.equations[:,:3].T+hull.equations[:,3]).max())
    if containment>1e-7:raise ValueError('full mesh outside computed convex hull')
    ov=full[hull.vertices];tv=vertices(asset/'table/table.obj');max_clear_error=0.;rows=[];source_hashes={}
    for seed in (a.panel,):
        directory=root/f's{seed}';r=json.loads((directory/'results.json').read_text())
        if r['run_status']!='COMPLETED' or r['mechanical_trajectories']!=768 or r['learned_policy_calls']!=202 or not r['no_source_actor_calls'] or not r['no_object_writes_after_first_tick'] or r['policy_checkpoint_sha256']!=m['policy_sha256']:raise ValueError('panel provenance')
        for name in ('initial','trace'):
            f=directory/(name+'.pt')
            if sha(f)!=r[name+'_sha256']:raise ValueError('trace drift')
            source_hashes[str(f)]=sha(f)
        initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False);data=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        n=lambda name:initial[name].numpy();v=lambda name:data[name].numpy()
        if initial['initial_state_semantics']!='committed native reset packet before firstsimulate' or not np.array_equal(n('reset_root_actor_ids'),np.sort(np.concatenate((np.arange(768)*3,np.arange(768)*3+2)))) or not np.array_equal(n('reset_dof_actor_ids'),np.arange(768)*3):raise ValueError('combined native reset transaction')
        if initial['refreshes_suppressed']<6:raise ValueError('native refresh ordering')
        motion=n('motion');arm=n('arm_assignment');steps=n('decision_steps');stops=n('phase_stop')[motion];parameters=n('primitive_parameters')
        if v('progress').shape!=(202,768) or not np.array_equal(v('progress'),np.broadcast_to(np.arange(1,203)[:,None],(202,768))):raise ValueError('all actual ticks')
        if not np.array_equal(steps,n('lift_start')[motion]-8) or np.any(parameters):raise ValueError('query timing/primitive contract')
        if any(not np.array_equal(np.bincount(arm[motion==mo],minlength=4),np.ones(4,dtype=int)*64) for mo in range(3)):raise ValueError('balanced assignment')
        private=torch.Generator(device='cpu').manual_seed(seed+16000);assigned=np.full(768,-1,dtype=np.int64)
        for mo in range(3):
            group=np.flatnonzero(motion==mo);assigned[group]=torch.arange(4).repeat_interleave(64)[torch.randperm(256,generator=private)].numpy()
        if not np.array_equal(assigned,arm):raise ValueError('prephysics private random assignment replay')
        g=torch.Generator(device='cpu').manual_seed(seed+11000);offsets=((torch.rand((768,2),generator=g)*2-1)*.01).numpy()
        if not np.array_equal(offsets,n('placement_offsets')) or initial['placement_seed']!=seed+11000 or initial['assignment_seed']!=seed+16000:raise ValueError('private placement seed')
        expected=n('before_placement').copy();expected[:,:2]+=offsets
        if not np.allclose(expected,n('object_root'),atol=1e-7,rtol=0):raise ValueError('only XY placement')
        if not np.array_equal(n('before_placement')[:,2:],n('object_root')[:,2:]):raise ValueError('initial Z/orientation/velocity changed')
        physical=json.loads((directory/'physical_metadata.json').read_text())
        if physical['actors_per_env']!=3 or physical['gravity']!=[0.,0.,-9.8100004196167] or sha(directory/'physical_metadata.json')!=r['physical_metadata_sha256'] or any(x['flags'] or abs(x['mass']-.0025936129968613386)>1e-9 for x in physical['object_body_properties']):raise ValueError('native physics drift')
        q=np.concatenate((n('base_q')[None],v('native_q')[:-1]),0);dq=np.concatenate((n('initial_dof_vel')[None],v('native_dq')[:-1]),0)
        obj=np.concatenate((n('object_root')[None],v('object_root')[:-1]),0);pair=np.concatenate((n('initial_contact')[None],v('contact')[:-1]),0)
        planned_progress=np.minimum(np.arange(1,203)[:,None],stops[None]);ref=n('native_reference_q')[motion[None],planned_progress]
        context=np.concatenate((q,dq,obj,pair,ref,(planned_progress/stops[None])[...,None]),-1)
        if not np.allclose(context,v('context'),atol=1e-6,rtol=0):raise ValueError('causal context reconstruction')
        lo=n('native_lower');hi=n('native_upper');residual=v('model_residual');scales=np.array([.02]*3+[.10]*3+[.40]*12,dtype=np.float32)
        base=ref+residual*scales;base[...,14]=np.clip(base[...,14],lo[14],hi[14]);base=couple(base,lo,hi)
        if not np.allclose(base,v('base_target'),atol=1e-5,rtol=0):raise ValueError('frozen actor target decoder')
        if initial['independent_coordinates']!=independent.tolist() or not np.array_equal(arm,n('policy_group')):raise ValueError('independent policy dimensions/groups')
        goal=base.copy();requested=v('request')
        active=arm>0
        delta=np.tanh(requested)*scale
        goal[:,active,:6]=np.maximum(np.minimum(base[:,active,:6]+delta[:,active,:6],q[:,active,:6]+np.abs(n('pd_scale')[:6])),q[:,active,:6]-np.abs(n('pd_scale')[:6]))
        for column,parent in enumerate(independent[6:],6):goal[:,active,parent]+=delta[:,active,column]
        goal[...,14]=np.clip(goal[...,14],lo[14],hi[14]);goal=couple(goal,lo,hi)
        maximum['target']=float(np.abs(goal-v('target')).max())
        if maximum['target']>1e-5:raise ValueError('bounded continuous target decoding')
        expected_action=(goal[:,:,independent]-q[:,:,independent])/np.abs(n('pd_scale')[independent])
        expected_action[:,~active]=0
        maximum['action']=float(np.abs(expected_action-v('executed_action12')).max())
        if maximum['action']>1e-5:raise ValueError('causal executed target minus currentq input')
        if not np.array_equal(n('action_normalization12'),np.abs(n('pd_scale')[independent])):raise ValueError('native PD normalization')
        # Raw-state reconstruction is checked separately. Replay ACTUAL saved inputs.
        expected_z=np.clip((v('context')-base_checkpoint['observation_mean'].numpy())/base_checkpoint['observation_std'].numpy(),-10,10)
        if not np.allclose(expected_z,v('normalized_context'),atol=2e-6,rtol=0):raise ValueError('fixed normalized current state')
        z=v('normalized_context').astype(np.float64)
        for group,variant in enumerate(('cm','state_only','none'),1):
            selected=arm==group;x=z[:,selected].reshape(-1,70)
            packet=checkpoint['variants'][variant];actor=packet['actor'];critic=packet['critic']
            h=np.maximum(layer(x,actor,'network.0'),0);h=np.maximum(layer(h,actor,'network.2'),0);mu=layer(h,actor,'network.4')
            stored_mean=v('request_mean')[:,selected].reshape(-1,12);maximum['mean']=max(maximum['mean'],float(np.abs(mu-stored_mean).max()))
            logstd=np.clip(actor['log_std'].numpy().astype(np.float64),-5,0)
            raw=requested[:,selected].reshape(-1,12).astype(np.float64)
            if not np.allclose(raw,stored_mean+np.exp(logstd)*v('request_noise')[:,selected].reshape(-1,12),atol=2e-6,rtol=0):raise ValueError('actual Gaussian request construction')
            lp=(-.5*((raw-mu)*np.exp(-logstd))**2-logstd-.5*np.log(2*np.pi)).sum(-1)
            maximum['logprob']=max(maximum['logprob'],float(np.abs(lp-v('request_logprob')[:,selected].ravel()).max()))
            latent=np.maximum(layer(x,critic,'encoder.0'),0);latent=np.maximum(layer(latent,critic,'encoder.2'),0)
            value=layer(latent,critic,'value').ravel();maximum['value']=max(maximum['value'],float(np.abs(value-v('critic_value')[:,selected].ravel()).max()))
            action_input=v('executed_action12')[:,selected].reshape(-1,12).astype(np.float64) if variant=='cm' else np.zeros((len(x),12))
            hidden=np.maximum(layer(np.concatenate((latent,action_input),-1),critic,'dynamics.0'),0)
            prediction=layer(hidden,critic,'dynamics.2');maximum['prediction']=max(maximum['prediction'],float(np.abs(prediction-v('aux_prediction')[:,selected].reshape(-1,6)).max()))
        if r['continuous_updates']!=panel_record['update'] or r['continuous_checkpoint_sha256']!=panel_record['sha256'] or r['deterministic']!=(a.panel in (568,569)):raise ValueError('frozen final-only deployment')
        if maximum['mean']>2e-5 or maximum['logprob']>3e-5 or maximum['value']>2e-5 or maximum['prediction']>3e-5:raise ValueError(('independent model/probability replay',maximum))
        groups=[np.flatnonzero(arm==group) for group in (1,2,3)]
        if any(not np.array_equal(v('request_noise')[:,groups[0]],v('request_noise')[:,ids]) for ids in groups[1:]):raise ValueError('common initial private request streams')
        physical_target=np.concatenate((v('object_root')[...,:3]-obj[...,:3],v('object_root')[...,7:10]-obj[...,7:10]),-1)/np.array([.005]*3+[.05]*3,dtype=np.float32)
        maximum['physical_target']=float(np.abs(physical_target-v('physical_transition')).max())
        if maximum['physical_target']>1e-5:raise ValueError('true one-step target, no future model input')
        action=v('action').copy()
        if np.any(np.abs(action)>1+1e-6) or np.any(action[...,[7,9,11,13,16,17]]!=0):raise ValueError('native action range/null')
        action[...,6:]=(1+action[...,6:])/2;pd=n('pd_offset')+n('pd_scale')*action;pd[...,:6]+=q[...,:6]
        for parent,children in COUPLING.items():
            for child,ratio in children:pd[...,child]=pd[...,parent]*ratio
        if not np.allclose(pd,goal,atol=1e-5,rtol=0):raise ValueError('native PD inverse')
        tab=n('table_root');tabrot=rotation(tab[:,3:7])
        if np.any(np.abs(tabrot[:,2,1])<.999):raise ValueError('table must be horizontal/thinY')
        support=(tabrot[:,2,:]@tv.T+tab[:,2,None]).max(-1)
        flat=v('object_root').reshape(-1,13);clear=np.empty(len(flat))
        for start in range(0,len(flat),1024):
            block=flat[start:start+1024];normal=rotation(block[:,3:7])[:,2,:]
            clear[start:start+len(block)]=(normal@ov.T+block[:,2,None]).min(-1)-support[np.arange(start,start+len(block))%768]
        clear=clear.reshape(202,768);error=float(np.abs(clear-v('clearance')).max());max_clear_error=max(max_clear_error,error)
        if error>1e-5:raise ValueError('independent fullmesh support')
        ids=np.arange(768);decision_context=context[steps,ids];decision_clearance=clear[steps-1,ids].astype(np.float32)
        features=numpy_features(decision_context.astype(np.float32),n('initial_height'),decision_clearance)
        if not np.allclose(decision_context,v('decision_context'),atol=1e-6,rtol=0) or not np.allclose(features,v('decision_features'),atol=1e-6,rtol=0):raise ValueError('pre-intervention physical features')
        rise=v('object_root')[...,2]-n('initial_height')[None];valid=(rise>=np.float32(.03))&(clear.astype(np.float32)>=np.float32(.02))
        for env in range(768):
            window=(v('progress')[:,env]>=steps[env]+21)&(v('progress')[:,env]<=steps[env]+30)
            retained=(v('progress')[:,env]>=stops[env]-74)&(v('progress')[:,env]<=stops[env]+30)
            if window.sum()!=10 or retained.sum()!=105:raise ValueError('complete primary/secondary windows')
            rows.append(dict(seed=seed,environment=env,motion=int(motion[env]),arm=int(arm[env]),physical30=bool(valid[window,env].all()),physical105=bool(valid[retained,env].all())))
    if len(rows)!=768:raise ValueError('complete integration cohort')
    (directory/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    report=dict(run_status='COMPLETED',rows=len(rows),checkpoint_update=checkpoint['updates'],no_policy_update_inside_rollout=True,no_policy_utility_claim=True,
        full_mesh_vertices=len(full),convex_hull_vertices=len(ov),full_mesh_hull_containment_max_m=containment,
        independent_clearance_max_error_m=max_clear_error,independent_numeric_maximum=maximum,
        causal_inputs_request_likelihood_target_projection_and_PD_verified=True,all_cohorts_included=True,
        source_sha256=source_hashes,wall_seconds=time.monotonic()-begin)
    (directory/'panel_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
if __name__=='__main__':main()
