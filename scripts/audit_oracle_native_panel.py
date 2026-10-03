"""Independent NumPy/SciPy mesh, native PD, actual105 labels and contact force checks."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--envs',type=int,default=96);p.add_argument('--ticks',type=int,default=202);a=p.parse_args()
    import numpy as np,torch
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from scripts.run_contact_response_probe import sha
    directory=a.panel;load=lambda path:torch.load(path,map_location='cpu',weights_only=False)
    initial=load(directory/'initial.pt');trace=load(directory/'trace.pt');meta=json.loads((directory/'physical_metadata.json').read_text());n=len(initial['motion']);ticks=len(trace['object_root'])
    if ticks!=a.ticks or n!=a.envs or meta['physics']!='gpu' or not meta['cpu_data_pipeline']:raise ValueError('fixed native scene')
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects'
    def vertices(path):return np.array([list(map(float,line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith('v ')])
    full=vertices(asset/'airplane/airplane.obj');hull=ConvexHull(full);support=full[hull.vertices];table=vertices(asset/'table/table.obj')
    containment=float((full@hull.equations[:,:3].T+hull.equations[:,3]).max())
    if containment>1e-7:raise ValueError('convex support certification')
    tab=initial['table_root'].numpy();R=Rotation.from_quat(tab[:,3:7]).as_matrix()
    if np.max(np.abs(np.abs(R[:,2,1])-1))>1e-5:raise ValueError('horizontal table plane')
    top=(R[:,2,:]@table.T+tab[:,2,None]).max(-1)
    roots=trace['object_root'].numpy().reshape(-1,13);clear=np.empty(len(roots))
    for start in range(0,len(roots),512):
        block=roots[start:start+512];normal=Rotation.from_quat(block[:,3:7]).as_matrix()[:,2,:]
        clear[start:start+len(block)]=(normal@support.T+block[:,2,None]).min(-1)-top[np.arange(start,start+len(block))%n]
    clear=clear.reshape(ticks,n);error=float(np.max(np.abs(clear-trace['clearance'].numpy())))
    if error>1e-5:raise ValueError('fullmesh clearance')
    stops=initial['phase_stop'][initial['motion']].numpy();mask=(np.arange(ticks)[:,None]>=stops[None,:]-74)&(np.arange(ticks)[:,None]<=stops[None,:]+30)
    if ticks==202 and not np.all(mask.sum(0)==105):raise ValueError('full105')
    good=(roots.reshape(ticks,n,13)[:,:,2]-initial['initial_height'].numpy()[None,:]>=.03)&(clear>=.02)
    labels=(good|~mask).all(0) if ticks==202 else None
    action=trace['action'].numpy().copy();q=trace['context'].numpy()[...,:18]
    if np.max(np.abs(action))>1+1e-6 or np.any(action[...,[7,9,11,13,16,17]]!=0):raise ValueError('bounded/null native actions')
    action[...,6:]=(1+action[...,6:])/2
    target=initial['pd_offset'].numpy()+initial['pd_scale'].numpy()*action;target[...,:6]+=q[...,:6]
    for parent,child,ratio in ((6,7,1.05),(8,9,1.05),(10,11,1.05),(12,13,1.05),(15,16,.6),(15,17,.8)):target[...,child]=target[...,parent]*ratio
    pd_error=float(np.max(np.abs(target-trace['target'].numpy())))
    if pd_error>1e-5:raise ValueError('actual native PD decoding')
    # Verify that object contact lambda * signed normal matches recorded tensor.
    contacts=np.load(directory/'contacts.npy',mmap_mode='r');frames=json.loads((directory/'contact_frames.json').read_text());physics=load(directory/'physics_states.pt')
    if len(frames)!=32*2*n:raise ValueError('all short contact frames')
    force_error=0.;position=0
    for frame in frames:
        if frame['offset']!=position or not 36<=frame['tick']<68:raise ValueError('raw contact temporal coverage')
        r=contacts[position:position+frame['count']];position+=frame['count'];select=(r['body0']==26)|(r['body1']==26);r=r[select]
        signed=np.where(r['body0']==26,1.,-1.)*r['lambda'];normal=np.column_stack([r['normal'][c] for c in ('x','y','z')]);total=(normal*signed[:,None]).sum(0,dtype=np.float64)
        actual=physics['net_force'][frame['tick']*2+frame['subtick'],frame['env'],26].numpy();force_error=max(force_error,float(np.max(np.abs(total-actual))))
    if position!=len(contacts) or force_error>1e-4:raise ValueError('attributed contact/normal-force contract')
    for actors in meta['actor_body_properties']:
        if len(actors[0]['names'])!=25 or len(actors[1]['names'])!=1 or len(actors[2]['names'])!=1 or abs(actors[2]['properties'][0]['mass']-.0025936129968613386)>1e-9:raise ValueError('actual nominal native actor identity')
    result=dict(run_status='COMPLETED',engineering_only=True,episodes=n,ticks=ticks,full_mesh_vertices=len(full),support_vertices=len(support),hull_containment_max=containment,clearance_max_error=error,pd_decode_max_error=pd_error,normal_force_reconstruction_max_error_N=force_error,successes=int(labels.sum()) if labels is not None else None,by_motion=[int(labels[initial['motion'].numpy()==m].sum()) for m in range(3)] if labels is not None else None,episode_success=labels.tolist() if labels is not None else None,all105_verified=ticks==202,no_terminal_query_labels=ticks==68,all_raw_contact_frames_verified=True,no_new_neural_or_physics=True,input_sha256={str(directory/f):sha(directory/f) for f in ('initial.pt','trace.pt','contacts.npy','physics_states.pt','physical_metadata.json','contact_frames.json')})
    with (directory/'native_audit.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','episode_success')}),flush=True)

if __name__=='__main__':main()
