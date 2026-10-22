"""Version501 historical TRAIN sidecars without rewriting complete task declarations.

CPU file migration / annotation audit only. No model, physics, training or git.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.task_annotations import task_with_safe_prefix,planned_contact_events,PREFIX_TASK_SEMANTICS
from wam_rl_policy_v3.task_spec import TaskSpec


def write(path,value):
    path=Path(path);part=path.with_suffix(path.suffix+'.part');part.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');part.replace(path)


def distribution(values):
    a=np.asarray(values,float)
    return None if not len(a) else dict(n=len(a),min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),
        p90=float(np.quantile(a,.9)),max=float(a.max()))


def audit_release(row,old_side,raw):
    """Observed robot release is compared to saved reference labels, not verified intent."""
    source=Path(row['source']);last_release=next((e.source_frame for e in reversed(raw.events) if e.kind.value=='RELEASE'),None)
    with np.load(source/'trajectory.npz') as z:
        poses=z['obj'].copy();reference=z['reference_current'].copy() if 'reference_current' in z else None
    if len(poses)!=raw.source_length or len(poses)!=row['frames']:raise ValueError('declared complete source_length mismatch')
    with np.load(old_side/'admission.npz') as z:coupled=z['coupled'].copy();supported=z['supported'].copy()
    result=dict(sequence=row['sequence'],source=str(source),raw_terminal_mode=raw.terminal_mode.value,raw_last_release_frame=last_release,
        endpoint_frame=int(raw.source_frames[-1]),source_length=raw.source_length,reference_labels_readable=False,
        raw_release_is_verified_planned_intent=False)
    if reference is not None:
        try:
            events,mode=planned_contact_events(reference)
            result.update(reference_labels_readable=True,reference_terminal_mode=mode.value,
                reference_events=[dict(source_frame=e.source_frame,kind=e.kind.value) for e in events],
                reference_scope='Saved reference_current binary contacts at aligned recorded ticks; audit-only, not an independent human intent certificate or training admission input.')
        except ValueError as error:result['reference_error']=str(error)
    if last_release is None:return result
    endpoint=int(raw.source_frames[-1]);center=poses[:,:3,3];suffix=center[last_release:endpoint+1]
    ref_releases=[e['source_frame'] for e in result.get('reference_events',[]) if e['kind']=='RELEASE']
    nearest=min((abs(f-last_release) for f in ref_releases),default=None)
    prior_coupled=bool(coupled[:last_release+1].any())
    fell=(center[:,2]<=center[last_release,2]-.05)&~supported&~coupled&(np.arange(len(center))>last_release)
    indices=np.flatnonzero(fell);fall_frame=int(indices[0]) if len(indices) else None
    result.update(release_before_endpoint=last_release<endpoint,post_release_seconds=(endpoint-last_release)/raw.source_fps,
        post_release_center_path_m=float(np.linalg.norm(np.diff(suffix,axis=0),axis=-1).sum()),
        post_release_endpoint_displacement_m=float(np.linalg.norm(suffix[-1]-suffix[0])),
        max_downward_from_release_m=float(max(0,center[last_release,2]-center[last_release:,2].min())),
        prior_motion_coupling=prior_coupled,unsupported_uncoupled_fall5cm_frame=fall_frame,
        unsupported_uncoupled_fall5cm_s=None if fall_frame is None else fall_frame/raw.source_fps,
        fall5cm_after_coupling=bool(prior_coupled and fall_frame is not None),
        nearest_reference_release_frame_gap=nearest,reference_release_within3frames=nearest is not None and nearest<=3,
        reference_keeps_holding=result.get('reference_terminal_mode')=='HOLD',
        drop_like_without_reference_release_witness=bool(prior_coupled and fall_frame is not None and result['reference_labels_readable'] and (nearest is None or nearest>3)),
        drop_scope='Proxy only: after measured last-contact end, object falls>=5cm from that pose while unsupported and uncoupled, with prior coupling. Could include intentional transport/release, flight or numerical failure; not causal slip or verified plan labels.')
    return result


def summarize(releases):
    selected=[r for r in releases if r['raw_terminal_mode']=='RELEASE'];readable=[r for r in selected if r['reference_labels_readable']]
    return dict(raw_release_tasks=len(selected),release_before_complete_raw_endpoint=sum(r['release_before_endpoint'] for r in selected),
        all_tasks_saved_reference_terminal_modes=dict(Counter(r.get('reference_terminal_mode','UNREADABLE') for r in releases)),
        raw_to_saved_reference_terminal_counts=[dict(raw=a,reference=b,count=n) for (a,b),n in sorted(Counter(
            (r['raw_terminal_mode'],r.get('reference_terminal_mode','UNREADABLE')) for r in releases).items())],
        saved_reference_is_independent_verified_intent=False,
        post_release_seconds=distribution([r['post_release_seconds'] for r in selected]),
        post_release_center_path_m=distribution([r['post_release_center_path_m'] for r in selected]),
        post_release_endpoint_displacement_m=distribution([r['post_release_endpoint_displacement_m'] for r in selected]),
        saved_reference_labels_readable=len(readable),reference_terminal_modes=dict(Counter(r['reference_terminal_mode'] for r in readable)),
        reference_release_within3frames=sum(r['reference_release_within3frames'] for r in readable),
        reference_keeps_holding=sum(r['reference_keeps_holding'] for r in readable),
        fall5cm_after_coupling=sum(r['fall5cm_after_coupling'] for r in selected),
        drop_like_without_reference_release_witness=sum(r['drop_like_without_reference_release_witness'] for r in readable),
        classification='Descriptive disagreement/proxies; raw-task RELEASE is not automatically a valid planned release. No raw declaration is relabeled by this audit.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-manifest',type=Path,default=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1')
    p.add_argument('--refresh-audit',action='store_true',help='Refresh summary from saved per-record audits only; do not rewrite task sidecars')
    a=p.parse_args();old_path=a.audit_manifest.resolve();old=json.loads(old_path.read_text());output=a.output.resolve()
    if a.refresh_audit:
        summary=json.loads((output/'repair-audit.json').read_text())
        summary['release_timing_and_reference_comparison']=summarize([r['release_audit'] for r in summary['records']])
        write(output/'repair-audit.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2));return
    if output.exists():raise ValueError('fresh versioned output required; preserve earlier sidecars')
    output.mkdir(parents=True);(output/'sequences').mkdir();started=time.monotonic();rows=[];records=[];cross=Counter()
    selected=[r for r in old['records'] if r['split']=='train' and r['engineering']=='PASS' and r['training_prefix_commands']>0]
    for row in selected:
        source=old_path.parent/row['sidecar'];target=output/'sequences'/row['sequence'];target.mkdir()
        raw_packet=json.loads((source/'raw-task.json').read_text());legacy=json.loads((source/'task.json').read_text());raw=TaskSpec.from_packet(raw_packet)
        with np.load(source/'admission.npz') as z:
            mask=z['training_prefix'].copy();unsafe=z['unsafe_frame'].copy()
        cut=int(row['training_prefix_commands'])
        if not np.array_equal(mask,np.arange(len(mask))<cut):raise ValueError('original prefix mask disagrees with command count')
        task,availability=task_with_safe_prefix(raw,cut,unsafe_future=bool(unsafe[cut+1:].any()))
        assert task.packet()==raw_packet
        for name in ['raw-task.json','admission.npz']:shutil.copyfile(source/name,target/name)
        shutil.copyfile(source/'task.json',target/'legacy-prefix-task.json')
        write(target/'task.json',task.packet());write(target/'task-availability.json',availability)
        fixed=dict(row,sidecar=str(target.relative_to(output)),terminal_mode=raw.terminal_mode.value,raw_terminal_mode=raw.terminal_mode.value,
            task_prefix_semantics=PREFIX_TASK_SEMANTICS,task_censored=availability['task_censored'],complete_positive_eligible=False,
            legacy_prefix_terminal_mode=legacy['terminal_mode'],legacy_sidecar=str(source))
        rows.append(fixed);cross[(legacy['terminal_mode'],raw.terminal_mode.value)]+=1
        release=audit_release(row,source,raw)
        records.append(dict(sequence=row['sequence'],prefix_commands=cut,old_terminal_mode=legacy['terminal_mode'],
            repaired_terminal_mode=raw.terminal_mode.value,task_packet_equals_original_raw=True,
            all_declared_source_frames_preserved=True,all_event_source_frames_preserved=True,source_length_preserved=True,
            censored=availability['task_censored'],unsafe_future=availability['unsafe_future'],
            beyond_prefix_declared_events=availability['events_beyond_observed_prefix'],release_audit=release))
        if len(rows)%50==0:write(output/'progress.json',dict(status='RUNNING',completed=len(rows),total=len(selected),elapsed_s=time.monotonic()-started))
    report=dict(old)
    report.update(status='COMPLETED',run_id=output.name,task_prefix_semantics=PREFIX_TASK_SEMANTICS,
        source_original_admission=str(old_path),code_identity='working-tree CPU fix; no git operations; old frozen3a0779f6 untouched',
        scope='Exactly the501 original TRAIN engineering-PASS nonempty-prefix tasks; no sample deletion, new physics or training.',
        sequences=len(rows),records=rows,declared_task_intent_verified=False,full_task_success_certified=False,new_training_authorized=False,
        retained_sequences=len(rows),admitted_prefix_commands=sum(r['training_prefix_commands'] for r in rows),
        retained_objects=sorted({r['object'] for r in rows}),engineering_counts=dict(Counter(r['engineering'] for r in rows)),
        limits='Original command/window/unsafe masks are byte-preserved; full raw task is preserved, not certified. Censored/unsafe future is not a complete-task positive example.')
    # Selected501 is an explicit manifest scope, not the source660-corpus count.
    report['policy']=dict(old['policy'],training='safe prefix controls only; complete task declarations retained')
    summary=dict(status='COMPLETED',n=len(rows),old_modes=dict(Counter(r['old_terminal_mode'] for r in records)),
        repaired_modes=dict(Counter(r['repaired_terminal_mode'] for r in records)),
        old_to_repaired_mode_counts=[dict(old=a,repaired=b,count=n) for (a,b),n in sorted(cross.items())],
        mode_changes=sum(r['old_terminal_mode']!=r['repaired_terminal_mode'] for r in records),
        full_task_packet_preserved=sum(r['task_packet_equals_original_raw'] for r in records),
        events_and_source_length_preserved=len(records),censored=sum(r['censored'] for r in records),unsafe_future=sum(r['unsafe_future'] for r in records),
        beyond_prefix_event_count=sum(len(r['beyond_prefix_declared_events']) for r in records),
        complete_positive_eligible=0,original_prefix_commands=report['admitted_prefix_commands'],
        release_timing_and_reference_comparison=summarize([r['release_audit'] for r in records]),records=records,
        source=str(old_path),output=str(output),runtime_python=sys.executable,elapsed_s=time.monotonic()-started)
    write(output/'manifest.json',report);write(output/'repair-audit.json',summary)
    write(output/'progress.json',dict(status='COMPLETED',completed=len(rows),total=len(selected),elapsed_s=time.monotonic()-started))
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))


if __name__=='__main__':main()
