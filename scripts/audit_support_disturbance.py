#!/usr/bin/env python3
"""Independent disturbance tensor, native PD and recovery geometry audit."""
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
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root/'run_manifest.json').read_text())
    if m['run_status']!='COLLECTION_COMPLETED' or m['evaluation_seeds']!=[527,528]:raise ValueError('fixed collection cohorts')
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects'
    def vertices(f):return np.array([list(map(float,l.split()[1:4])) for l in f.read_text().splitlines() if l.startswith('v ')])
    full=vertices(asset/'airplane/airplane.obj');hull=ConvexHull(full)
    # Linear support extrema lie on the convex hull; validate the hull against ALL mesh vertices.
    containment=float((full@hull.equations[:,:3].T+hull.equations[:,3]).max())
    if containment>1e-7:raise ValueError('full mesh outside computed convex hull')
    ov=full[hull.vertices];tv=vertices(asset/'table/table.obj');max_clear_error=0.;rows=[];source_hashes={}
    for seed in (527,528):
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
        anchor=n('anchors')[motion];cell=n('cell_assignment');direction=((cell//4)%2)*2-1;load=cell//8
        expected_parameters=np.zeros((768,3),dtype=np.float32);expected_parameters[:,0]=direction*.02*((arm==1).astype(float)-(arm==2).astype(float));expected_parameters[:,2]=(arm==3)*.15
        if not np.array_equal(steps,anchor+11) or not np.array_equal(n('impulse_steps'),anchor+10) or not np.allclose(parameters,expected_parameters,atol=1e-7,rtol=0):raise ValueError('recovery timing/primitive contract')
        if not np.array_equal(arm,cell%4) or not np.array_equal(load,n('load_index')) or not np.array_equal(direction,n('force_direction')):raise ValueError('factorial cells')
        b=int(initial['bodies_per_env']);object_slot=b-1
        if not np.array_equal(n('object_body_indices'),np.arange(768)*b+object_slot):raise ValueError('object SDK body index')
        force=np.zeros((202,768,b,3),dtype=np.float32)
        levels=np.array([0.,.05,.15,.45],dtype=np.float32)
        force[anchor+10,np.arange(768),object_slot,0]=levels[load]*direction
        if not np.array_equal(v('applied_force'),force) or not np.array_equal(n('load_levels'),levels):raise ValueError('one step object-only force, all other bodies/ticks zero')
        if any(not np.array_equal(np.bincount(cell[motion==mo],minlength=32),np.ones(32,dtype=int)*8) for mo in range(3)):raise ValueError('balanced factorial cells')
        private=torch.Generator(device='cpu').manual_seed(seed+13000);assigned=np.full(768,-1,dtype=np.int64)
        for mo in range(3):
            group=np.flatnonzero(motion==mo);assigned[group]=torch.arange(32).repeat_interleave(8)[torch.randperm(256,generator=private)].numpy()
        if not np.array_equal(assigned,cell):raise ValueError('prephysics private random assignment replay')
        offsets=np.zeros((768,2),dtype=np.float32)
        if not np.array_equal(offsets,n('placement_offsets')) or initial['placement_seed'] is not None or initial['assignment_seed']!=seed+13000:raise ValueError('new task has no XY placement perturbation')
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
        active=np.arange(202)[:,None]>=steps[None];applied=parameters[None]*active[...,None];goal=base.copy();goal[...,:2]+=applied[...,:2]
        for parent in COUPLING:goal[...,parent]+=applied[...,2]
        goal=couple(goal,lo,hi)
        if not np.allclose(goal,v('target'),atol=1e-5,rtol=0):raise ValueError('executable primitive decoding')
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
            window=(v('progress')[:,env]>=anchor[env]+1)&(v('progress')[:,env]<=anchor[env]+10)
            retained=(v('progress')[:,env]>=stops[env]-59)&(v('progress')[:,env]<=stops[env]+30)
            if window.sum()!=10 or retained.sum()!=90:raise ValueError('complete pre-disturbance/recovery windows')
            rows.append(dict(seed=seed,environment=env,motion=int(motion[env]),arm=int(arm[env]),load_index=int(load[env]),direction=int(direction[env]),pre_geometry10=bool(valid[window,env].all()),recovery90=bool(valid[retained,env].all())))
    if len(rows)!=1536:raise ValueError('complete cohorts')
    (root/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    def summarize(group):
        return dict(n=len(group),pre_geometry10_count=sum(r['pre_geometry10'] for r in group),pre_geometry10_rate=sum(r['pre_geometry10'] for r in group)/len(group),
            arms={str(a):dict(n=sum(r['arm']==a for r in group),recovery90_count=sum(r['recovery90'] for r in group if r['arm']==a),
                recovery90_rate=sum(r['recovery90'] for r in group if r['arm']==a)/sum(r['arm']==a for r in group)) for a in range(4)})
    summaries={str(mo):{str(li):summarize([r for r in rows if r['motion']==mo and r['load_index']==li]) for li in range(4)} for mo in range(3)}
    per_seed={str(seed):{str(li):summarize([r for r in rows if r['motion']==1 and r['load_index']==li and r['seed']==seed]) for li in range(4)} for seed in (527,528)}
    eligible=[];gates={}
    for li in (1,2,3):
        p=summaries['1'][str(li)];rates={k:v['recovery90_rate'] for k,v in p['arms'].items()}
        good=[]
        for a in (1,3):
            seed_pass=all(per_seed[str(seed)][str(li)]['arms'][str(a)]['recovery90_rate']>=per_seed[str(seed)][str(li)]['arms'][str(c)]['recovery90_rate'] for seed in (527,528) for c in (0,2))
            if rates[str(a)]-rates['0']>=.15 and rates[str(a)]-rates['2']>=.10 and seed_pass:good.append(a)
        valid=p['pre_geometry10_rate']>=.8 and .1<=rates['0']<=.8 and bool(good)
        gates[str(li)]=dict(pre_geometry80=p['pre_geometry10_rate']>=.8,null_rate10_to80=.1<=rates['0']<=.8,eligible_recovery_arms=good,eligible=valid)
        if valid:eligible.append(li)
    result=dict(run_status='COMPLETED',label='PROMISING' if eligible else 'UNPROMISING',primary_motion=1,rows=1536,
        summaries=summaries,primary_seeds=per_seed,gates=gates,eligible_load_indices=eligible,chosen_lowest_load_index=min(eligible) if eligible else None,
        load_levels_n=[0,.05,.15,.45],boundary='finite task calibration with privileged recovery arms; no Cm, learned policy improvement or Validation')
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    report=dict(run_status='COMPLETED',rows=len(rows),full_mesh_vertices=len(full),convex_hull_vertices=len(ov),full_mesh_hull_containment_max_m=containment,
        independent_clearance_max_error_m=max_clear_error,causal_context_and_primitive_reconstruction=True,one_tick_object_only_forces_verified=True,
        all_cohorts_included=True,source_sha256=source_hashes,wall_seconds=time.monotonic()-begin)
    (root/'collection_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
if __name__=='__main__':main()
