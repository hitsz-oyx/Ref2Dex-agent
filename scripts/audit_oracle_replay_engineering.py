"""Independent saved-array prefix and attributed contact audit; no new physics."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    import torch, numpy as np
    from scripts.run_contact_response_probe import sha
    out=a.source;baseline=out/'baseline';replay=out/'replay'
    load=lambda path:torch.load(path,map_location='cpu',weights_only=False)
    x,y=load(baseline/'trace.pt'),load(replay/'trace.pt')
    initial=load(baseline/'initial.pt');n=len(initial['motion']);t=len(x['native_q'])
    decisions=initial['lift_start'][initial['motion']]-8
    # Fixed absolute tolerances, set before inspecting replay arrays.
    limits=dict(native_q=1e-6,native_dq=1e-5,object_position=1e-6,
                object_quaternion=1e-6,object_velocity=1e-5,rigid_position=1e-6,
                rigid_quaternion=1e-6,rigid_velocity=1e-5)
    metrics={};passed=True
    for key,slice_,name in (('native_q',slice(None),'native_q'),('native_dq',slice(None),'native_dq'),
        ('object_root',slice(0,3),'object_position'),('object_root',slice(3,7),'object_quaternion'),
        ('object_root',slice(7,13),'object_velocity'),('rigid_state',slice(0,3),'rigid_position'),
        ('rigid_state',slice(3,7),'rigid_quaternion'),('rigid_state',slice(7,13),'rigid_velocity')):
        delta=(x[key][...,slice_]-y[key][...,slice_]).abs()
        mask=torch.arange(t)[:,None] < decisions[None,:]
        value=float(delta[mask].max());metrics[name]=dict(max_prefix_abs_error=value,limit=limits[name])
        passed &= value<=limits[name]
    command_exact=torch.equal(x['target'],y['target']) and torch.equal(x['action'],y['action'])
    all_state_equal={key:torch.equal(x[key],y[key]) for key in ('object_root','native_q','native_dq','rigid_state','net_force')}
    contacts=np.load(baseline/'contacts.npy');frames=json.loads((baseline/'contact_frames.json').read_text())
    physics=load(baseline/'physics_states.pt')['net_force'].numpy()
    normal_sum=np.zeros_like(physics,dtype=np.float64);pair_counts={};nonzero=0
    for f in frames:
        records=contacts[f['offset']:f['offset']+f['count']];index=f['tick']*2+f['subtick'];env=f['env']
        for record in records:
            b0,b1=int(record['body0']),int(record['body1']);normal=np.array([record['normal'][v] for v in ('x','y','z')]);force=float(record['lambda'])*normal
            pair_counts[str(tuple(sorted((b0,b1))))]=pair_counts.get(str(tuple(sorted((b0,b1)))),0)+1
            if 0<=b0<27: normal_sum[index,env,b0]+=force
            if 0<=b1<27: normal_sum[index,env,b1]-=force
            nonzero+=abs(float(record['lambda']))>1e-8
    target_force=physics[:,:,26];pred=normal_sum[:,:,26]
    difference=np.linalg.norm(target_force-pred,axis=-1)
    sign_correct=float(np.mean(np.linalg.norm(target_force+pred,axis=-1)))>float(np.mean(difference))
    result=dict(run_status='COMPLETED',engineering_only=True,prefix_limits=limits,prefix_metrics=metrics,
        prefix_contract_pass=bool(passed and command_exact),identical_all_commands=command_exact,
        complete_replay_state_equal=all_state_equal,contact_records=len(contacts),nonzero_normal_records=int(nonzero),
        pair_counts=pair_counts,target_normal_sum_force_residual_mean=float(difference.mean()),
        target_normal_sum_force_residual_max=float(difference.max()),normal_on_body0_sign_supported=sign_correct,
        input_sha256={str(path):sha(path) for path in (baseline/'trace.pt',replay/'trace.pt',baseline/'contacts.npy',baseline/'physics_states.pt')},
        friction_force_not_certified=True,substep_aggregate_points_not_final_pose_exact=True)
    dest=out/'replay_audit.json'
    with dest.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
