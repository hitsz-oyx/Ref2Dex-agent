"""Recover completed physical records after the v10 NumPy-int JSON error.

Never reruns simulation, changes scores, edits source failures or discards budget.
The regular raw-state rescorer must independently accept the recovered record.
"""
import argparse,importlib.util,json,os,sys,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.worker.resolve();out=a.output.resolve();m=json.loads((run/'manifest.json').read_text())
    if m['status']!='FAILED' or m['error']!="TypeError('Object of type int64 is not JSON serializable')":raise ValueError('only known completed-physical JSON error can be recovered')
    if out.exists():raise ValueError('fresh recovery receipt required')
    reg=json.loads((run/'registration.json').read_text());label=m['active_episode']['id'];folder=run/label;consumer=json.loads((folder/'consumer.json').read_text());stages=json.loads((folder/'stages.json').read_text())
    if json.loads((folder/'exit-receipt.json').read_text())['native_returncode']!=0:raise ValueError('native execution did not finish cleanly')
    with np.load(folder/'trajectory.npz') as z:data={k:z[k].copy() for k in z.files}
    n=len(data['action']);ticks=data['tick'][:,0]
    if not np.array_equal(ticks,np.arange(n+1)) or len(consumer)!=n or len(stages)!=n+1:raise ValueError('incomplete saved physical trajectory')
    frozen=run/'runtime-code';os.environ.update(REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_V6_V5_CODE_ROOT=str(frozen),REF2DEX_V5_BASE_CODE_ROOT=str(frozen));sys.path[:0]=[str(frozen/'src/task/wam-rl-policy-v3/src'),str(frozen/'src/task/trajectory-policy/src'),str(frozen/'src/task/ptv3-shared-wam-v6/src')]
    from wam_rl_policy_v3.physical_safety import physical_safety
    from wam_rl_policy_v3.task_feedback import FeedbackObserver
    from wam_rl_policy_v3.measured_execution import online_support
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM
    from wam_rl_policy_v3.pose_consumer import pose7
    from wam_rl_policy_v3.stage_evaluation import terminal_success
    job=next(j for j in reg['jobs'] if j['id']==label[:-3]);arm=label[-2:];name=job['sequence'];audit=json.loads((folder/'native-audit.json').read_text())
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'];record=next(r for r in catalog if r['sequence']==name)
    with np.load(Path(record['source'])/'canonical.npz') as z:points=z['points'].copy()
    with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/name/'admission.npz') as z:bounds=z['table_bounds'].copy()
    with np.load(job['reference']) as z:ref={k:z[k].copy() for k in ('q','dq','hand')}
    qerr=data['q'][0,0]-ref['q'][0,0];qerr[3:6]=(qerr[3:6]+np.pi)%(2*np.pi)-np.pi
    initial_errors=dict(q=float(np.abs(qerr).max()),dq=float(np.abs(data['dq'][0,0]-ref['dq'][0,0]).max()),hand=float(np.abs(data['hand'][0,0]-ref['hand'][0,0]).max()))
    goal=pose7(data['objects'][0,0]);goal[2,3]+=.08;errors=np.linalg.norm(data['objects'][:,0,:3]-goal[:3,3],axis=-1);observer=FeedbackObserver(points);drop=False
    for t in range(n+1):
        state={k:v[t] for k,v in data.items() if k in ('q','dq','objects','table_pose','object_contact_force','contact_force_valid')};obj=data['objects'][t,0];pose=pose7(obj)
        safe=physical_safety(state,0,points,bounds,audit['physical_joint_limits'])
        if safe['abort'] and t!=n:raise ValueError('execution continued after safety abort')
        feedback=observer.update(pose,data['hand'][t,0,NATIVE_TO_WAM],obj[7:13],t/30,contact_forces=data['contact_forces'][t,0],force_valid=bool(data['contact_force_valid'][t,0]),supported=online_support(state,0,pose,points,bounds),workspace_ok=not safe['abort'])
        drop|=bool(stages[t]['achieved']['hold'] and feedback.released)
    if not safe['abort'] and n!=reg['horizon_controls']:raise ValueError('not a terminal physical trajectory')
    stage=stages[-1];raw=data['raw_action'];actions=data['action'];adapter=data['adapter_raw_action'];error=float(errors[-1])
    row=dict(id=label,trial=job['id'],subset=job['subset'],sequence=name,object=record['object'],arm=arm,order_position=job['order'].index(arm),seed=job['native_seed'],model_seed=job['model_seed'],controls=n,
        final_success=terminal_success(stage,error,n,reg['horizon_controls'],safe['abort']),ever_success=any(s['current']['hold'] and e<=.015 for s,e in zip(stages,errors)),final_hold=bool(n==reg['horizon_controls'] and stage['current']['hold'] and not safe['abort']),stage_achieved=stage['achieved'],stage_final=stage['current'],longest_hold_s=stage['longest_hold_s'],release_applicable=False,drop_after_hold=drop,
        final_goal_error_m=error,goal_within15mm=bool(not safe['abort'] and n==reg['horizon_controls'] and error<=.015),goal_error_quantiles_m=np.quantile(errors,[0,.5,.9,1]).tolist(),physical_abort=safe['reason'] if safe['abort'] else None,physical_safety_final=safe,wam_calls=int(data['tau_valid'].sum()),actor_calls=n,encoder_covered_controls=int(data['tau_valid'][:-1].sum()),final_clipped_controls=int((np.abs(raw-actions)>1e-6).any(-1).sum()),adapter_clipped_controls=int((np.abs(adapter)>1+1e-6).any(-1).sum()),initial_errors=initial_errors,seconds_observed=n/30,serialization_recovered=True)
    spec=importlib.util.spec_from_file_location('repair_worker',TASK/'tools/run/run_grasp_repair_worker.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);row.update(module.control_counts(consumer,actions))
    out.mkdir(parents=True);(out/'runtime-code').symlink_to(frozen,target_is_directory=True);shutil.copyfile(run/'registration.json',out/'registration.json')
    for r in m['runs']:(out/r['id']).symlink_to(run/r['id'],target_is_directory=True)
    (out/label).symlink_to(folder,target_is_directory=True)
    recovered=dict(m,status='RECOVERED_COMPLETED_PHYSICAL',source_failed_manifest=str(run/'manifest.json'),runs=m['runs']+[row],original_error=m['error']);recovered.pop('error')
    (out/'manifest.json').write_text(json.dumps(recovered,indent=2)+'\n');(out/'recovered-result.json').write_text(json.dumps(row,indent=2)+'\n')
    print('RECOVERED',label,'controls',n,'success',row['final_success'],'NO physics rerun; original failure retained')


if __name__=='__main__':main()
