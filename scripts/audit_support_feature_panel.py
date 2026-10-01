#!/usr/bin/env python3
"""Independent per-panel geometry/control audit for fresh policy learning."""
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
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--evaluation',action='store_true');a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root.parent/'run_manifest.json').read_text())
    if m['run_status']!='RUNNING' or a.seed not in (m['evaluation_seeds'] if a.evaluation else m['training_seeds']):raise ValueError('fixed fresh cohorts')
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects'
    def vertices(f):return np.array([list(map(float,l.split()[1:4])) for l in f.read_text().splitlines() if l.startswith('v ')])
    full=vertices(asset/'airplane/airplane.obj');hull=ConvexHull(full)
    # Linear support extrema lie on the convex hull; validate the hull against ALL mesh vertices.
    containment=float((full@hull.equations[:,:3].T+hull.equations[:,3]).max())
    if containment>1e-7:raise ValueError('full mesh outside computed convex hull')
    ov=full[hull.vertices];tv=vertices(asset/'table/table.obj');max_clear_error=0.;rows=[];source_hashes={}
    for seed in (a.seed,):
        directory=root;r=json.loads((directory/'results.json').read_text())
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
        if a.evaluation:
            choice=torch.load(directory/'decisions.pt',map_location='cpu',weights_only=False)
            if sha(directory/'decisions.pt')!=r['decisions_sha256']:raise ValueError('decision packet drift')
            arm=choice['macro_action'].numpy();parameters=choice['parameters'].numpy();groups=n('policy_group')
            if np.any(n('arm_assignment')!=-1) or np.any(n('primitive_parameters')!=0):raise ValueError('macro chosen only at current-state decision')
        if v('progress').shape!=(202,768) or not np.array_equal(v('progress'),np.broadcast_to(np.arange(1,203)[:,None],(202,768))):raise ValueError('all actual ticks')
        if not np.array_equal(steps,n('lift_start')[motion]-8) or not np.array_equal(parameters,PRIMITIVES[arm]):raise ValueError('query timing/primitive contract')
        category=groups if a.evaluation else arm
        count=4 if a.evaluation else 8;per_cell=64 if a.evaluation else 32
        if any(not np.array_equal(np.bincount(category[motion==mo],minlength=count),np.ones(count,dtype=int)*per_cell) for mo in range(3)):raise ValueError('balanced group / behavior assignment')
        private=torch.Generator(device='cpu').manual_seed(seed+(14000 if a.evaluation else 12000));assigned=np.full(768,-1,dtype=np.int64)
        for mo in range(3):
            group=np.flatnonzero(motion==mo);assigned[group]=torch.arange(count).repeat_interleave(per_cell)[torch.randperm(256,generator=private)].numpy()
        if not np.array_equal(assigned,category):raise ValueError('prephysics private random assignment replay')
        g=torch.Generator(device='cpu').manual_seed(seed+11000);offsets=((torch.rand((768,2),generator=g)*2-1)*.01).numpy()
        if not np.array_equal(offsets,n('placement_offsets')) or initial['placement_seed']!=seed+11000 or initial['assignment_seed']!=seed+(14000 if a.evaluation else 12000):raise ValueError('private placement seed')
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
            window=(v('progress')[:,env]>=steps[env]+21)&(v('progress')[:,env]<=steps[env]+30)
            retained=(v('progress')[:,env]>=stops[env]-74)&(v('progress')[:,env]<=stops[env]+30)
            if window.sum()!=10 or retained.sum()!=105:raise ValueError('complete primary/secondary windows')
            rows.append(dict(seed=seed,environment=env,motion=int(motion[env]),arm=int(arm[env]),policy_group=int(groups[env]) if a.evaluation else -1,physical30=bool(valid[window,env].all()),physical105=bool(valid[retained,env].all())))
        panel_features=features
        panel_parameters=parameters
    if len(rows)!=768:raise ValueError('complete panel')
    dataset=root/'audited.pt';torch.save(dict(state=torch.from_numpy(panel_features),primitive=torch.from_numpy(panel_parameters),
        label=torch.tensor([r['physical30'] for r in rows],dtype=torch.float32),motion=torch.tensor([r['motion'] for r in rows]),
        arm=torch.tensor([r['arm'] for r in rows]),policy_group=torch.tensor([r['policy_group'] for r in rows]),seed=torch.tensor([r['seed'] for r in rows]),physical105=torch.tensor([r['physical105'] for r in rows]),rows=rows),dataset)
    report=dict(run_status='COMPLETED',seed=a.seed,evaluation=a.evaluation,rows=len(rows),full_mesh_vertices=len(full),convex_hull_vertices=len(ov),full_mesh_hull_containment_max_m=containment,
        independent_clearance_max_error_m=max_clear_error,causal_context_and_primitive_reconstruction=True,all_cohorts_included=True,dataset_sha256=sha(dataset),source_sha256=source_hashes,wall_seconds=time.monotonic()-begin)
    (root/'panel_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
if __name__=='__main__':main()
