"""Watch fresh teacher receipts, audit measured fields and emit admission sidecars."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.teacher_quality import QUALITY_SCHEMA,TABLE_SUPPORT_CONTRACT,failure_masks,measured_failures,table_support
from wam_rl_policy_v3.task_annotations import task_from_teacher_archive,task_with_safe_prefix,GOAL_SOURCE,PREFIX_TASK_SEMANTICS
from wam_rl_policy_v3.task_feedback import FeedbackObserver
from wam_rl_policy_v3.robot_adapter import InspireAdapter


def write(path,value):
    temporary=path.with_suffix('.part');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-audit',type=Path,help='Reuse sidecars only when canonical same-execution source paths match')
    p.add_argument('--wait-seconds',type=int,default=7200);p.add_argument('--admit',action='store_true');a=p.parse_args()
    a.corpus=a.corpus.resolve();a.output=a.output.resolve()
    if ROOT/'outputs/wam-rl-policy-v3' not in a.output.parents:raise ValueError('v3 Task output required')
    if a.output.exists() or a.corpus==a.output:raise ValueError('fresh separate audit output required')
    a.output.mkdir(parents=True);(a.output/'sequences').mkdir()
    old=ROOT/'src/task/ptv3-shared-wam-v2/tools/audit/audit_current_rollout.py'
    spec=importlib.util.spec_from_file_location('existing_current_rollout_audit',old)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import torch
    import trimesh
    table_bounds=trimesh.load('/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/mjcf/objects/table/table.obj',force='mesh').bounds
    reviews={};old_manifests={}
    review_root=ROOT/'outputs/ptv3-shared-wam-v2/dexplore-review-20261015-r1/annotations'
    for file in sorted(review_root.glob('*.json')):
        annotation=json.loads(file.read_text())
        if annotation['verdict'] in ('unusable','uncertain') and annotation['reason']!='tracking_error':
            reviews.setdefault(annotation['sequence'],[]).append(annotation)
    started=time.monotonic();rows={};seen=set();timed_out=False
    if a.reuse_audit:
        while not a.reuse_audit.exists():
            if time.monotonic()-started>=a.wait_seconds:raise TimeoutError('prior audit did not finish')
            time.sleep(15)
        prior=json.loads(a.reuse_audit.read_text())
        if (prior.get('schema')!=QUALITY_SCHEMA or prior.get('goal_source')!=GOAL_SOURCE
                or prior.get('table_support_contract')!=TABLE_SUPPORT_CONTRACT):raise ValueError('different prior admission/support contract; re-audit required')
        if prior.get('task_prefix_semantics')!=PREFIX_TASK_SEMANTICS:
            raise ValueError('legacy prefix tasks cannot be reused; migrate full declared task sidecars first')
        root=a.reuse_audit.resolve().parent
        for row in prior['records']:
            if row['engineering']!='PASS':continue
            name=row['sequence'];(a.output/'sequences'/name).symlink_to(root/'sequences'/name)
            rows[name]=row
    while True:
        manifest=json.loads((a.corpus/'manifest.json').read_text())
        for receipt_file in sorted((a.corpus/'batches').glob('*/manifest.json')):
            if str(receipt_file) in seen:continue
            receipt=json.loads(receipt_file.read_text())
            if receipt['status']!='COMPLETED':continue
            contract=receipt['observation_contract'];ids=contract['contact_body_ids']
            if receipt.get('force_contract',{}).get('schema')!='ref2dex.inspire-measured-forces.v1':raise ValueError('same-execution forces missing')
            for record in receipt['records']:
                name=record['sequence'];source=receipt_file.parent/record['path']
                if name in rows:
                    if Path(rows[name]['source']).resolve()!=source.resolve():raise ValueError('different physical execution for reused audit '+name)
                    continue
                folder=a.output/'sequences'/name;folder.mkdir()
                try:
                    with np.load(source/'trajectory.npz',allow_pickle=False) as data:
                        engineering=module.audit_sequence(data,record,contract,receipt['native_controller'],include_reference=False)
                        n=record['frames'];valid=data['dof_force_valid'];contact_valid=data['contact_force_valid']
                        if (data['dof_force'].shape!=(n,18) or valid.dtype!=np.bool_ or valid.shape!=(n,)
                                or not np.isfinite(data['dof_force']).all() or valid[0] or not valid[1:].all()
                                or not np.array_equal(valid,contact_valid) or data['dof_force'][0].any()):raise ValueError('invalid measured-force timing/schema')
                        with np.load(source/'canonical.npz') as g:points=g['points'].copy()
                        # Static native table initialization only, not desired
                        # object/hand motion from the teacher training reference.
                        initialization=torch.load(Path(receipt['geometric_plan_root'])/name/'interaction_hand_inspire.pt',map_location='cpu',weights_only=False)
                        table=initialization[0,238:245].numpy().copy();del initialization
                        support,gap,footprint=table_support(data['obj'],points,table,table_bounds,data['object_contact_force'])
                        observer=FeedbackObserver(points);coupled=[];lost=[];released=[]
                        for t in range(n):
                            feedback=observer.update(data['obj'][t],InspireAdapter.canonical_hand(data['hand'][t]),data['velocity'][t],t/30,
                                contact_forces=data['body_contact_forces'][t,ids],force_valid=bool(contact_valid[t]),supported=bool(support[t]))
                            coupled.append(feedback.grasp_established);lost.append(feedback.grasp_lost);released.append(feedback.released)
                        reasons=measured_failures(data['obj'],coupled,lost,support,gap,footprint)
                        human=np.zeros(n,bool);review_unmatched=0
                        for annotation in reviews.get(name,[]):
                            old_root=Path(annotation['source_root'])
                            if str(old_root) not in old_manifests:
                                old_meta=json.loads((old_root/'manifest.json').read_text())
                                old_manifests[str(old_root)]={r['sequence']:r for r in old_meta['records']}
                            old_record=old_manifests[str(old_root)][name]
                            with np.load(old_root/old_record['path']/'trajectory.npz') as previous:
                                lo=annotation['start_frame'];hi=min(annotation['end_frame_exclusive'],n,len(previous['q']))
                                if hi<=lo:continue
                                same=np.ones(hi-lo,bool)
                                for key in ('q','obj','hand'):
                                    delta=np.abs(data[key][lo:hi]-previous[key][lo:hi]).reshape(hi-lo,-1).max(-1)
                                    same&=delta<=1e-5
                                human[lo:hi]|=same;review_unmatched+=int((~same).sum())
                        reasons['matched_human_unusable_or_uncertain']=human
                        bad=np.logical_or.reduce(list(reasons.values()))
                        commands,windows,unsafe=failure_masks(bad)
                        if not any(coupled):commands[:]=False;windows[:]=False
                        raw_task=task_from_teacher_archive(data,ids,object_points=points)
                        # Only data availability is truncated. The original
                        # complete declaration, terminal and future events stay.
                        cut=int(np.flatnonzero(~commands)[0]) if (~commands).any() else len(commands)
                        prefix=np.arange(len(commands))<cut
                        task,availability=task_with_safe_prefix(raw_task,cut,unsafe_future=bool(unsafe[cut+1:].any()))
                        np.savez_compressed(folder/'admission.npz',command=commands,window24=windows,unsafe_frame=unsafe,
                            training_prefix=prefix,contact_force_valid=contact_valid,dof_force_valid=valid,supported=support,
                            coupled=np.asarray(coupled),released=np.asarray(released),table_pose=table,table_bounds=table_bounds)
                        write(folder/'raw-task.json',raw_task.packet())
                        write(folder/'task.json',task.packet())
                        write(folder/'task-availability.json',availability)
                        rows[name]=dict(sequence=name,object=record['object'],source=str(source),frames=n,commands=n-1,
                            split=manifest['inherited_split'][name],engineering='PASS',maximum_pd_error=engineering['max_pd_error'],
                            ever_coupled=any(coupled),admitted_commands=int(commands.sum()),admitted_windows=int(windows.sum()),
                            training_prefix_commands=cut,excluded_commands=int((~commands).sum()),
                            failure_frames={k:int(v.sum()) for k,v in reasons.items()},terminal_mode=task.terminal_mode.value,
                            raw_terminal_mode=raw_task.terminal_mode.value,review_unmatched_frames=review_unmatched,
                            sidecar=str(folder.relative_to(a.output)),goal_source=GOAL_SOURCE,
                            task_prefix_semantics=PREFIX_TASK_SEMANTICS,task_censored=availability['task_censored'],complete_positive_eligible=False)
                except (ValueError,AssertionError,KeyError) as error:
                    rows[name]=dict(sequence=name,object=record['object'],split=manifest['inherited_split'][name],
                        source=str(source),engineering='REJECTED',error=repr(error),training_prefix_commands=0,admitted_commands=0,admitted_windows=0)
            seen.add(str(receipt_file))
            write(a.output/'progress.json',dict(status='RUNNING',audited_sequences=len(rows),elapsed_s=time.monotonic()-started,
                admitted_prefix_commands=sum(r['training_prefix_commands'] for r in rows.values())))
        if manifest['status'] in ('COMPLETED','FAILED'):break
        if time.monotonic()-started>=a.wait_seconds:timed_out=True;break
        time.sleep(15)
    complete=manifest['status']=='COMPLETED' and len(rows)==660 and not timed_out
    report=dict(schema=QUALITY_SCHEMA,status='COMPLETED' if complete else 'INCOMPLETE',training_allowed=bool(complete and any(r['training_prefix_commands'] for r in rows.values())),
        task='wam-rl-policy-v3',run_id=a.output.name,experiment_id='P-20261019-inspire-task-contract',
        git_commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        corpus=str(a.corpus),python=sys.executable,elapsed_s=time.monotonic()-started,sequences=len(rows),goal_source=GOAL_SOURCE,
        table_support_contract=TABLE_SUPPORT_CONTRACT,
        task_prefix_semantics=PREFIX_TASK_SEMANTICS,
        quality_uses_training_reference=False,shape_features='CAD512 captured; learned shape64 is produced by the later BC encoder',
        policy=dict(preceding_failure_frames=50,history=4,horizon=24,max_crossing_translation_m=.08,max_crossing_rotation_rad=.7,
            support_tolerance_m=.02,penetration_tolerance_m=.02,no_motion_coupling='exclude manipulation example',training='safe prefix controls; complete task declaration retained'),
        engineering_counts=dict(Counter(r['engineering'] for r in rows.values())),
        admitted_prefix_commands=sum(r['training_prefix_commands'] for r in rows.values()),
        retained_sequences=sum(r['training_prefix_commands']>0 for r in rows.values()),
        retained_objects=sorted({r['object'] for r in rows.values() if r['training_prefix_commands']>0}),
        records=sorted(rows.values(),key=lambda r:r['sequence']),
        limits='Contact/coupling/support are measured heuristics, not grasp certificates; no desired training-reference error gate. Raw rejected data remain immutable.')
    write(a.output/'manifest.json',report);write(a.output/'progress.json',{k:v for k,v in report.items() if k!='records'})
    if a.admit and report['training_allowed']:
        snapshot=a.corpus/'pre-measured-audit-manifest.json'
        if not snapshot.exists():snapshot.write_text((a.corpus/'manifest.json').read_text())
        manifest.update(training_allowed=True,admission_pending=False,measured_rollout_audit=str(a.output/'manifest.json'))
        write(a.corpus/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
