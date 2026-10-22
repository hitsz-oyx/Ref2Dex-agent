"""Quantify real joint/phase/configuration failures and held-object servo errors."""
import argparse,json,collections
from pathlib import Path
import numpy as np


def profile(manifests,names):
    groups={};records=[]
    for path in manifests:
        m=json.loads(path.read_text())
        for r in m['runs']:
            if r['arm']!='A1':continue
            folder=path.parent/r['id'];infos=json.loads((folder/'consumer.json').read_text());audit=json.loads((folder/'native-audit.json').read_text())
            with np.load(folder/'trajectory.npz') as z:
                q=z['q'][:,0];dq=z['dq'][:,0];pd=z['pd_targets_actual'];obj=z['objects'][:,0];action=z['action'];forces=z['contact_forces'][:,0]
            phase=infos[-1]['phase'] if infos else 'reset';indices=r['physical_safety_final']['joint_violation_indices'];key=r['subset'];g=groups.setdefault(key,dict(n=0,joint_abort=0,phase=collections.Counter(),joint=collections.Counter(),configuration=collections.Counter(),object=collections.Counter(),held_failed_precision=0))
            g['n']+=1;g['held_failed_precision']+=int(r['final_hold'] and not r['final_success'])
            detail=dict(id=r['id'],subset=key,object=r['object'],sequence=r['sequence'],controls=r['controls'],phase=phase,abort=r['physical_abort'],final_success=r['final_success'],final_hold=r['final_hold'],error_mm=r['final_goal_error_m']*1000)
            if indices:
                g['joint_abort']+=1;g['phase'][phase]+=1;g['object'][r['object']]+=1
                # Native middle rotation branch is a reproducible configuration descriptor.
                config='wrist_pitch_near_gimbal' if abs(np.cos(q[-1,4]))<.2 else 'wrist_pitch_regular'
                g['configuration'][config]+=1;g['joint'].update(names[i] for i in indices)
                detail.update(configuration=config,wrist_q=q[-1,3:6].tolist(),wrist_dq=dq[-1,3:6].tolist(),joints=[dict(index=i,name=names[i],q=float(q[-1,i]),dq=float(dq[-1,i]),last_PD=float(pd[-1,i]),lower=audit['physical_joint_limits']['lower'][i],upper=audit['physical_joint_limits']['upper'][i]) for i in indices],last_translation_action_m=action[-1,:3].tolist())
            if r['final_hold']:
                goal=np.asarray(infos[-1]['postgrasp_object_pose'])[:3,3];lo=max(0,len(action)-60)
                detail.update(goal_error_xyz_mm=((goal-obj[-1,:3])*1000).tolist(),last2s_position_range_mm=(np.ptp(obj[lo:,:3],axis=0)*1000).tolist(),last2s_median_command_mm=(np.median(action[lo:,:3],axis=0)*1000).tolist(),last2s_median_dq=np.median(dq[lo:,:3],axis=0).tolist(),final_force_norms=np.linalg.norm(forces[-1],axis=-1).tolist())
            records.append(detail)
    return dict(groups=groups,records=records,native_dof_names=names,unit='first physical abort per episode; multiple joint violations counted per named joint; configuration descriptor is not a proven cause')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,action='append',required=True);p.add_argument('--native-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    names=json.loads(a.native_audit.read_text())['native_dof_names'];result=profile(a.manifest,names);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['groups'],indent=2))


if __name__=='__main__':main()
