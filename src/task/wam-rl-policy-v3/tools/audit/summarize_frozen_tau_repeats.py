"""Independent saved-state scoring and log-contract checks for the n5 repeats."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/audit')]
from audit_first_tau_loss import load,physical_rows
from wam_rl_policy_v3.tau_replay_contract import require_saved_clock,reference_tau,pd_targets,bouts
from wam_rl_policy_v3.inspire_contract import expand_action,NATIVE_TO_WAM


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=json.loads((a.run/'manifest.json').read_text())
    if report['status']!='COMPLETED' or len(report['runs'])!=25:raise ValueError('complete25-case run required; do not summarize partial data as n5')
    original=json.loads((Path(report['original_run'])/'manifest.json').read_text());source=Path(original['source_pd_reference']);points=load(source/'canonical.npz')['points']
    side=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/report['source_sequence'];bounds=load(side/'admission.npz')['table_bounds']
    ref=load(Path(report['original_run'])/'source-pd/trajectory.npz')['hand'][:,0]
    execution=a.run/'runtime-code/src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/execution_contract.py';loader=importlib.util.spec_from_file_location('repeat_encoder_domain',execution);module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    offset=np.asarray(report['native_initial_audit']['controller']['offset']);scale=np.asarray(report['native_initial_audit']['controller']['scale']);records=[];all_rows={}
    for online in report['runs']:
        data=load(a.run/online['label']/'trajectory.npz');require_saved_clock(data);n=len(data['action']);physical=physical_rows(data,points,bounds,report['native_initial_audit']['physical_joint_limits']);all_rows[online['label']]=physical
        spans=bouts([r['held'] for r in physical]);longest=max(((hi-lo-1)/30 for lo,hi in spans),default=0.);ever=any(hi-lo>=31 for lo,hi in spans);final=bool(spans and spans[-1][1]==len(physical) and spans[-1][1]-spans[-1][0]>=31)
        actual_error=float(np.abs(pd_targets(data['action'],data['q'][:n,0],offset,scale)-data['pd_targets_actual']).max()) if n else 0.
        clipping=float(np.max(np.abs(np.clip(data['raw_action'],-1,1)-data['action']))) if n else 0.
        expansion=float(np.max(np.abs(expand_action(data['action'])-data['expanded_action']))) if n else 0.
        packed_tau_error=float(np.max(np.abs(data['observation'][:,421:685]-data['tau'].reshape(len(data['q']),-1))))
        packed_valid_ok=np.array_equal(data['observation'][:,685],data['tau_valid'].astype(np.float32))
        episode_ok=np.array_equal(data['episodes_input'],data['episodes'][:n,0]) and (np.diff(data['episodes'][:,0])==0).all()
        gt_errors=[]
        if online['chain'].endswith('-gt'):
            for t in range(len(data['q'])):
                obj=data['objects'][t,0];pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
                available=bool(t>=3 and module.encoder_domain_available(points,data['hand'][t,0,NATIVE_TO_WAM],pose));tau,valid=reference_tau(ref,t,available=available)
                if bool(data['tau_valid'][t])!=valid or not np.allclose(data['tau'][t],tau,atol=1e-7,rtol=0):gt_errors.append(t)
        loss=spans[0][1] if spans and spans[0][1]<len(physical) else None
        contact=None;lost=None
        if spans:
            contact=next((r for r in physical[spans[0][0]:] if not r['grasp_established'] or r['force_count']<2),None)
            lost=next((r for r in physical[spans[0][0]:] if r['grasp_lost']),None)
        agrees=ever==online['ever_success'] and final==online['final_success'] and abs(longest-online['longest_hold_elapsed_s'])<1e-9
        records.append(dict(label=online['label'],chain=online['chain'],block=online['block'],controls=n,ever_success=ever,final_success=final,longest_elapsed_s=longest,
            physical_safety_abort=online['physical_safety_abort'],covered_controls=online['encoder_covered_controls'],clipped_controls=online['clipped_controls'],
            first_height_or_hold_failure=None if loss is None else physical[loss],first_contact_or_grasp_condition_loss=contact,first_grasp_lost_flag=lost,
            actual_pd_reconstruction_max_error=actual_error,clipping_log_max_error=clipping,expanded_log_max_error=expansion,packed_tau_max_error=packed_tau_error,packed_validity_ok=bool(packed_valid_ok),episode_alignment_ok=bool(episode_ok),gt_contract_error_ticks=gt_errors,matches_online=agrees))
    chains={}
    for name in ('source-pd','baseline-gt','calibrated-gt','baseline-wam','calibrated-wam'):
        part=[r for r in records if r['chain']==name];values=[r['longest_elapsed_s'] for r in part]
        if len(part)!=5:raise ValueError('missing repeated chain: '+name)
        chains[name]=dict(n=5,successes=sum(r['ever_success'] for r in part),final_successes=sum(r['final_success'] for r in part),median_hold_s=float(np.median(values)),min_hold_s=float(min(values)),max_hold_s=float(max(values)),holds_s=values,
            physical_safety_aborts=sum(r['physical_safety_abort'] is not None for r in part),clipped_controls=sum(r['clipped_controls'] for r in part),controls=sum(r['controls'] for r in part),covered_controls=sum(r['covered_controls'] for r in part))
    indexed={(r['block'],r['chain']):r for r in records};contrasts={}
    for name,left,right in [('base_wam_minus_gt','baseline-wam','baseline-gt'),('gt_calibrated_minus_base','calibrated-gt','baseline-gt'),('wam_calibrated_minus_base','calibrated-wam','baseline-wam')]:
        differences=[indexed[b,left]['longest_elapsed_s']-indexed[b,right]['longest_elapsed_s'] for b in range(5)]
        contrasts[name]=dict(differences_s=differences,median_difference_s=float(np.median(differences)),positive=sum(d>1e-9 for d in differences),negative=sum(d< -1e-9 for d in differences),ties=sum(abs(d)<=1e-9 for d in differences))
    passed=all(r['matches_online'] and r['episode_alignment_ok'] and not r['gt_contract_error_ticks'] and r['actual_pd_reconstruction_max_error']<=2e-5 and r['clipping_log_max_error']==0 and r['expanded_log_max_error']==0 and r['packed_tau_max_error']==0 and r['packed_validity_ok'] for r in records)
    summary=dict(status='PASS' if passed else 'FAIL',protocol=report['protocol'],n_cases=len(records),learning_updates=report['learning_updates'],environment_controls=report['environment_controls'],elapsed_s=report['elapsed_s'],chains=chains,contrasts=contrasts,records=records,
        maximum_initial_error=max([error for row in report['runs'] for error in row['initial_max_errors'].values()]+[0.]),maximum_actual_pd_error=max(r['actual_pd_reconstruction_max_error'] for r in records),
        limitations=['Repeated seen trajectory and balanced reset order, not five independent tasks/seeds.','Old81-state reference and oldv2 condition generator intentionally frozen; not the new productionv3/full-reference test.',
            'Longest-hold ordering does not establish statistical success-rate superiority.','Shared feedback observer remains a heuristic, not independent contact ground truth.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'physical-rows.json').write_text(json.dumps(all_rows)+'\n');print(json.dumps({k:summary[k] for k in ('status','n_cases','learning_updates','environment_controls','elapsed_s','chains','contrasts','maximum_initial_error','maximum_actual_pd_error')}));return int(not passed)


if __name__=='__main__':raise SystemExit(main())
