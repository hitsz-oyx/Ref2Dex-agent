"""Replay fixed raw-task declarations through isolated old/new TaskControllers.

Uses existing robot archives only. Prefix is a snapshot of the same causal full
replay, not a shorter task. No old controls(), physics, training, model or GPU.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'tools/audit'))
import audit_task_reachability as core


def initialize(task_root):
    core.initialize(task_root)


def score_record(arguments):
    row,manifest_root,trace_enabled=arguments;side=Path(manifest_root)/row['sidecar'];source=Path(row['source'])
    packet=json.loads((side/'task.json').read_text());assert packet==json.loads((side/'raw-task.json').read_text())
    spec=core.TaskSpec.from_packet(packet)
    with np.load(source/'trajectory.npz') as z:
        data={k:z[k].copy() for k in ['obj','hand','velocity','body_contact_forces','contact_force_valid','frame_tick']}
    with np.load(source/'canonical.npz') as z:points=z['points'].copy()
    with np.load(side/'admission.npz') as z:support=z['supported'].copy()
    receipt=json.loads((source.parents[1]/'manifest.json').read_text());ids=receipt['observation_contract']['contact_body_ids']
    n=len(data['obj']);cut=int(row['training_prefix_commands']);assert np.array_equal(data['frame_tick'],np.arange(n))
    assert n==spec.source_length and 0<cut<n
    controller=core.TaskController(spec,points);counts=Counter();first_events={};first_release=None;path_complete_ever=False
    acquired_once=False;max_progress=0.;prefix=None;trace=[];first_endpoint_eligible=None;eligible_since=None;longest_eligible=0.
    release_frame=next((e.source_frame for e in reversed(spec.events) if e.kind.value=='RELEASE'),None)
    for tick in range(n):
        now=tick/30.;packet=controller.observe(data['obj'][tick],data['hand'][tick,core.NATIVE_TO_WAM],data['velocity'][tick],now,
            contact_forces=data['body_contact_forces'][tick,ids],force_valid=bool(data['contact_force_valid'][tick]),supported=bool(support[tick]))
        decision=packet['decision'];feedback=packet['feedback'];scheduler=controller.scheduler;path=scheduler.path
        counts[decision.phase.name]+=1;max_progress=max(max_progress,float(decision.path_progress))
        if decision.phase.name=='RELEASE' and first_release is None:first_release=tick
        path_complete_ever|=bool(decision.path_complete);acquired_once|=bool(feedback.grasp_established)
        for event in decision.transitions:first_events.setdefault(event,tick)
        at_end=bool(path._reached(data['obj'][tick],path.poses[-1]))
        if spec.terminal_mode.value=='RELEASE':
            eligible=at_end and feedback.released and feedback.stable and (bool(support[tick]) or not spec.require_support)
        else:eligible=at_end and feedback.grasp_established and not feedback.grasp_lost and feedback.stable
        if eligible:
            if first_endpoint_eligible is None:first_endpoint_eligible=tick
            if eligible_since is None:eligible_since=now
            longest_eligible=max(longest_eligible,now-eligible_since)
        else:eligible_since=None
        pointer=int(path.pointer);next_index=int(path.milestones[pointer+1]) if pointer+1<len(path.milestones) else None
        current=dict(tick=tick,seconds=now,phase=decision.phase.name,recovering=bool(decision.recovering),
            grasp_candidate=bool(feedback.grasp_candidate),grasp_established=bool(feedback.grasp_established),
            grasp_lost=bool(feedback.grasp_lost),released=bool(feedback.released),stable=bool(feedback.stable),supported=bool(support[tick]),
            contact_observed=bool(feedback.contact_observed),force_valid=bool(feedback.force_valid),
            path_initial_acquired=bool(path.initial_acquired),milestone_pointer=pointer,milestones=len(path.milestones),
            next_milestone_source_frame=None if next_index is None else int(spec.source_frames[next_index]),
            path_reason=None if scheduler.path_decision is None else scheduler.path_decision['reason'],
            confirmed_progress=float(decision.path_progress),confirmed_path_complete=bool(decision.path_complete),
            path_settle_since=path.settle_since,terminal_settle_since=scheduler.settle_since,
            event_index=int(scheduler.event_index),planned_next_event=None if decision.next_event.kind is None else decision.next_event.kind.value,
            source_cursor_frame=float(decision.source_cursor_frame),at_complete_raw_endpoint_pose=at_end,
            independent_endpoint_terminal_eligible=bool(eligible),task_success=bool(decision.task_complete),
            terminated=bool(decision.terminated),failure_reason=decision.failure_reason,transitions=list(decision.transitions))
        if trace_enabled:trace.append(current)
        def snapshot(boundary):
            actual_elapsed=0. if scheduler.settle_since is None else now-scheduler.settle_since
            return dict(task_success=bool(decision.task_complete),failure_reason=decision.failure_reason,
                terminated=bool(decision.terminated),recording_censored=not decision.terminated,observed_end_tick=tick,observed_end_s=now,
                available_boundary_tick=int(boundary),complete_raw_endpoint_frame=int(spec.source_frames[-1]),
                archive_boundary_contains_declared_endpoint=bool(boundary>=spec.source_frames[-1]),
                recorded_seconds_after_declared_endpoint=max(0,int(boundary)-int(spec.source_frames[-1]))/30.,
                release_authorized=first_release is not None,first_release_authorized_tick=first_release,
                first_release_authorized_s=None if first_release is None else first_release/30.,
                path_complete_ever=bool(path_complete_ever),motion_coupling_ever=bool(acquired_once),max_confirmed_progress=max_progress,
                phase_sample_counts=dict(counts),first_event_ticks=dict(first_events),final=current.copy(),
                independent_endpoint_eligible_first_tick=first_endpoint_eligible,
                longest_independent_endpoint_terminal_elapsed_s=longest_eligible,
                independent_endpoint_terminal_elapsed_at_end_s=0. if eligible_since is None else now-eligible_since,
                actual_terminal_settle_elapsed_at_end_s=actual_elapsed,
                endpoint_eligible_but_actual_upstream_gate_incomplete=bool(eligible and not decision.path_complete),
                terminal_eligible_at_end_but_less_than1s=bool(not decision.terminated and eligible_since is not None and now-eligible_since<1.),
                censor_scope='Recording ends without a TaskController terminal; conditions may be missing, upstream blocked, or a full1s terminal window unobserved. Not a physical failure verdict.')
        if tick==cut or (decision.terminated and tick<cut):prefix=snapshot(cut)
        if decision.terminated:break
    if prefix is None:raise RuntimeError('prefix snapshot not reached')
    full=snapshot(n-1)
    result=dict(sequence=row['sequence'],source=str(source),task_path=str(side/'task.json'),terminal_mode=spec.terminal_mode.value,
        prefix_controls=cut,recorded_frames=n,declared_release_frame=release_frame,
        complete_task_unchanged=True,prefix=prefix,full=full)
    if trace_enabled:result['trace']=trace
    return result


def summary(cases):
    output=dict(n=len(cases))
    for window in ['prefix','full']:
        rows=[c[window] for c in cases]
        output[window]=dict(successes=sum(r['task_success'] for r in rows),release_authorized=sum(r['release_authorized'] for r in rows),
            path_complete_ever=sum(r['path_complete_ever'] for r in rows),recording_censored=sum(r['recording_censored'] for r in rows),
            failures=dict(Counter(r['failure_reason'] for r in rows if r['failure_reason'] is not None)),
            final_phases=dict(Counter(r['final']['phase'] for r in rows)),
            path_initial_acquired=sum(r['final']['path_initial_acquired'] for r in rows),
            endpoint_eligible_but_upstream_gate_incomplete=sum(r['endpoint_eligible_but_actual_upstream_gate_incomplete'] for r in rows),
            eligible_end_but_less_than1s=sum(r['terminal_eligible_at_end_but_less_than1s'] for r in rows),
            complete_raw_endpoint_frame_recorded=sum(r['archive_boundary_contains_declared_endpoint'] for r in rows),
            no_recorded_time_after_declared_endpoint=sum(r['archive_boundary_contains_declared_endpoint'] and r['recorded_seconds_after_declared_endpoint']==0 for r in rows))
    return output


def compare(old_path,new_path,output):
    old=json.loads(Path(old_path).read_text());new=json.loads(Path(new_path).read_text())
    before={c['sequence']:c for c in old['cases']};after={c['sequence']:c for c in new['cases']}
    assert set(before)==set(after)
    pairs=[]
    for name in sorted(before):
        a=before[name];b=after[name]
        for key in ['source','task_path','prefix_controls','recorded_frames','declared_release_frame']:assert a[key]==b[key]
        pairs.append(dict(sequence=name,old={w:a[w] for w in ['prefix','full']},new={w:b[w] for w in ['prefix','full']}))
    result=dict(status='COMPLETED',n=len(pairs),old_file=str(old_path),new_file=str(new_path),
        old_scheduler=old['scheduler_file'],new_scheduler=new['scheduler_file'],new_schema=new['scheduler_schema'],
        summary=dict(old=old['summary'],new=new['summary']),
        scope='Same existing robot trajectory and full raw task; only scorer package changes. No new execution, changed action, training, physical qualification or independent policy success-rate estimate.',
        paired={w:dict(success_gained=sum(not c['old'][w]['task_success'] and c['new'][w]['task_success'] for c in pairs),
            success_lost=sum(c['old'][w]['task_success'] and not c['new'][w]['task_success'] for c in pairs),
            release_gained=sum(not c['old'][w]['release_authorized'] and c['new'][w]['release_authorized'] for c in pairs),
            still_no_release=sum(not c['old'][w]['release_authorized'] and not c['new'][w]['release_authorized'] for c in pairs)) for w in ['prefix','full']},
        cases=pairs)
    censored_release=[c['full'] for c in new['cases'] if c['full']['release_authorized'] and not c['full']['task_success']]
    result['new_remaining_gates']=dict(final_pointer_zero=sum(c['full']['final']['milestone_pointer']==0 for c in new['cases']),
        release_authorized_without_success=len(censored_release),
        authorized_still_censored=sum(c['recording_censored'] for c in censored_release),
        authorized_path_not_complete=sum(not c['path_complete_ever'] for c in censored_release),
        authorized_path_complete_actual_terminal_dwell_started_short=sum(c['path_complete_ever'] and c['final']['terminal_settle_since'] is not None for c in censored_release),
        authorized_path_complete_no_valid_terminal_window_at_end=sum(c['path_complete_ever'] and c['final']['terminal_settle_since'] is None for c in censored_release),
        raw_record_end_present_but_no_extra_observed_second='All501 archives contain their declared last source frame; no padding or repeated final state was used. Geometric arrival may precede that source time.',
        full_success_inside_original_safe_prefix=sum(c['full']['task_success'] and c['full']['observed_end_tick']<=c['prefix_controls'] for c in new['cases']),
        full_success_after_original_safe_prefix=sum(c['full']['task_success'] and c['full']['observed_end_tick']>c['prefix_controls'] for c in new['cases']))
    output=Path(output);core.write(output,result)
    lines=['# Complete raw-task scoring: old60ef vs release-at-declared-pose.v2','',result['scope'],'',
        '| saved window / scorer | n | success | RELEASE authorized | path complete | recording censored | failures |',
        '|---|---:|---:|---:|---:|---:|---|']
    for window in ['prefix','full']:
        for label,source in [('old',old),('new',new)]:
            row=source['summary'][window];lines.append('| '+window+' / '+label+' | '+str(len(pairs))+' | '+str(row['successes'])+' | '+str(row['release_authorized'])+' | '+str(row['path_complete_ever'])+' | '+str(row['recording_censored'])+' | '+str(row['failures'])+' |')
    lines+=['', 'Endpoint availability and dwell are separate. A source endpoint present at the last archive frame does not automatically provide a further1s observed hold; geometric arrival can occur earlier, and actual successes must have a full observed terminal window.',
        'Censored records and upstream gate failures are not labels of physical inability. The raw declaration itself still has unverified intended-release semantics.',
        'New remaining-gate accounting: `'+json.dumps(result['new_remaining_gates'],sort_keys=True)+'`. Full-record successes after the safe prefix do not become safe complete-task positive controls.','']
    for name in ['s6_bowl_drink_1','s2_hammer_use_2']:
        a=before[name]['full'];b=after[name]['full']
        lines.append('- '+name+': old success='+str(a['task_success'])+', first RELEASE='+str(a['first_release_authorized_tick'])+'; new success='+str(b['task_success'])+', first RELEASE='+str(b['first_release_authorized_tick'])+', success/final tick='+str(b['observed_end_tick'])+', terminal dwell='+str(b['actual_terminal_settle_elapsed_at_end_s'])+'s.')
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2));return result


def diagnose_rollbacks(detailed_path,output):
    """Verify observed provisional-progress rollback, without changing scoring."""
    data=json.loads(Path(detailed_path).read_text());cases=[]
    for case in data['cases']:
        trace=case.get('trace',[])
        if not trace or case['full']['final']['milestone_pointer']!=0:continue
        drops=[(a,b) for a,b in zip(trace,trace[1:]) if b['milestone_pointer']<a['milestone_pointer']]
        if not drops:continue
        previous,current=drops[0];source=Path(case['source']);spec=core.TaskSpec.from_packet(json.loads(Path(case['task_path']).read_text()))
        with np.load(source/'canonical.npz') as z:points=z['points'].copy()
        with np.load(source/'trajectory.npz') as z:
            poses=z['obj'].copy();hands=z['hand'].copy();forces=z['body_contact_forces'].copy();valid=z['contact_force_valid'].copy()
        ids=json.loads((source.parents[1]/'manifest.json').read_text())['observation_contract']['contact_body_ids']
        path=core.TaskScheduler(spec,points).path;index=int(path.milestones[1]);target=path.poses[index]
        reached=core.endpoint_mask(poses,target,points,spec.goal_type)
        ready=np.array([r['phase'] in ['TRACK','RELEASE'] or ((r['contact_observed'] or r['grasp_candidate'] or r['grasp_established']) and not r['recovering']) for r in trace])
        acquired=next((r for r in trace if r['grasp_established']),None)
        if acquired is None:continue
        tick=acquired['tick'];errors=path._errors(poses[tick],target);witness=[]
        for t in [previous['tick'],current['tick'],current['tick']+1,tick]:
            local=(hands[t,[2,4,6,8,10]]-poses[t,None,:3,3])@poses[t,:3,:3]
            gap=np.linalg.norm(local[:,None]-points[None],axis=-1).min(-1)
            witness.append(dict(tick=t,force_valid=bool(valid[t]),force_bodies=int((np.linalg.norm(forces[t,ids],axis=-1)>=.1).sum()),
                near_fingers=int((gap<=.018).sum()),target_reached=bool(reached[t]),trace=trace[t]))
        cases.append(dict(sequence=case['sequence'],rollback_from=previous['milestone_pointer'],rollback_to=current['milestone_pointer'],
            rollback_tick=current['tick'],rollback_s=current['seconds'],first_grasp_tick=tick,first_grasp_s=tick/30.,
            initial_acquired_at0=trace[0]['path_initial_acquired'],first_pending_source_frame=int(spec.source_frames[index]),
            first_pending_reached_ticks_before_rollback=np.flatnonzero(reached[:current['tick']]).tolist(),
            reached_and_execution_ready_after_rollback=np.flatnonzero(reached[current['tick']:len(ready)]&ready[current['tick']:]).tolist(),
            target_error_at_first_grasp=dict(position_m=errors[0],rotation_rad=errors[1],mean_CAD_m=errors[2]),witness=witness))
    result=dict(status='OBSERVED_CONTRACT_MISMATCH',scope='Only these traced cases; do not assign this mechanism to all76 final-pointer-zero cases.',
        detailed_trace=str(detailed_path),cases=cases,
        interpretation='Initial geometric anchor was acquired; one-frame pre-grasp contact loss discards provisional observed-path progress. Later coupling does not revisit the erased early milestone. This diagnoses a task/path-stage compatibility issue, not physical inability or a mandate to relax thresholds.')
    core.write(Path(output),result);print(json.dumps(dict(status=result['status'],cases=len(cases),output=str(output))));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1/manifest.json')
    p.add_argument('--task-root',type=Path);p.add_argument('--implementation');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sequences',nargs='*');p.add_argument('--trace-all',action='store_true');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--compare',action='store_true');p.add_argument('--old-file',type=Path);p.add_argument('--new-file',type=Path)
    p.add_argument('--diagnose-rollbacks',type=Path);a=p.parse_args()
    if a.compare:
        compare(a.old_file,a.new_file,a.output);return
    if a.task_root is None or (a.implementation is None and not a.diagnose_rollbacks):p.error('--task-root and --implementation required for replay')
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;started=time.monotonic();initialize(a.task_root)
    if a.diagnose_rollbacks:
        diagnose_rollbacks(a.diagnose_rollbacks,a.output);return
    import wam_rl_policy_v3.task_scheduler as scheduler_module
    assert Path(scheduler_module.__file__).resolve().is_relative_to(a.task_root.resolve()) if hasattr(Path('.'),'is_relative_to') else str(Path(scheduler_module.__file__).resolve()).startswith(str(a.task_root.resolve()))
    manifest=json.loads(a.manifest.read_text());rows=manifest['records']
    if a.sequences:rows=[row for row in rows if row['sequence'] in a.sequences]
    if a.sequences and len(rows)!=len(a.sequences):raise ValueError('requested sequences missing')
    trace_names={'s6_bowl_drink_1','s2_hammer_use_2'};cases=[];a.output.parent.mkdir(parents=True,exist_ok=True)
    progress=a.output.with_suffix('.progress.json')
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(a.task_root),)) as pool:
        futures=[pool.submit(score_record,(row,str(a.manifest.parent),a.trace_all or row['sequence'] in trace_names)) for row in rows]
        for future in as_completed(futures):
            cases.append(future.result())
            if len(cases)%25==0:core.write(progress,dict(status='RUNNING',completed=len(cases),total=len(rows),elapsed_s=time.monotonic()-started))
    cases.sort(key=lambda row:row['sequence'])
    result=dict(status='COMPLETED',implementation=a.implementation,scheduler_schema=getattr(scheduler_module,'SCHEDULER_SCHEMA','PRE_V2_NO_SCHEMA'),
        scheduler_file=scheduler_module.__file__,python=sys.executable,device='cpu',workers=a.workers,
        task_root=str(a.task_root.resolve()),manifest=str(a.manifest.resolve()),elapsed_s=time.monotonic()-started,
        scope='Same existing robot measured fields /30Hz; complete raw task unchanged; safe prefix snapshot vs full recording; no controls(), PTv3 gate, simulation, training or regenerated action.',
        summary=summary(cases),cases=cases)
    core.write(a.output,result);core.write(progress,dict(status='COMPLETED',completed=len(cases),total=len(rows),elapsed_s=result['elapsed_s']))
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
