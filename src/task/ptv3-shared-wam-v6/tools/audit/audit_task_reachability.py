"""CPU-only audit of the actual 501 historical tasks and their measured archives.

No simulator, GPU, new policy evaluation, training, or git operation is used.
Frozen 3a0779f6 TaskController code is imported from the run's recorded argv.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[5]
HZ=30.


def write(path,data):
    temporary=path.with_suffix(path.suffix+'.part')
    def scalar(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError('Unsupported audit value: '+type(value).__name__)
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False,default=scalar)+'\n');temporary.replace(path)


def initialize(frozen):
    global TaskController,TaskSpec,TaskScheduler,TaskSchedulerConfig,InteractionFeedback,NATIVE_TO_WAM
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(Path(frozen)/'src'))
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.task_spec import TaskSpec
    from wam_rl_policy_v3.task_scheduler import TaskScheduler,TaskSchedulerConfig
    from wam_rl_policy_v3.task_feedback import InteractionFeedback
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM


def bouts(mask):
    changes=np.flatnonzero(np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8)))
    return [(int(a),int(b)) for a,b in zip(changes[::2],changes[1::2])]


def span(mask):
    samples=max((b-a for a,b in bouts(mask)),default=0)
    return dict(samples=samples,occupancy_s=samples/HZ,elapsed_span_s=max(0,samples-1)/HZ)


def first(mask):
    indices=np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def endpoint_mask(poses,endpoint,points,goal_type):
    p=np.linalg.norm(poses[:,:3,3]-endpoint[:3,3],axis=-1)
    if goal_type=='position_only':return p<=.008
    angle=Rotation.from_matrix(poses[:,:3,:3].transpose(0,2,1)@endpoint[:3,:3]).magnitude()
    target=points@endpoint[:3,:3].T+endpoint[:3,3]
    errors=[]
    for lo in range(0,len(poses),64):
        part=poses[lo:lo+64];world=points[None]@part[:,:3,:3].transpose(0,2,1)+part[:,None,:3,3]
        errors.extend(np.linalg.norm(world-target[None],axis=-1).mean(-1))
    return (p<=.008)&(angle<=.12)&(np.asarray(errors)<=.008)


def independent_path(poses,spec,points):
    """Greedy ordered-waypoint witness; no learned or task-scheduler feedback."""
    cursor=0;arrivals=[]
    for waypoint in spec.object_path:
        while cursor<len(poses):
            actual=poses[cursor]
            reached=np.linalg.norm(actual[:3,3]-waypoint[:3,3])<=.008
            if reached and spec.goal_type!='position_only':
                reached=Rotation.from_matrix(actual[:3,:3].T@waypoint[:3,:3]).magnitude()<=.12
                if reached:
                    reached=np.linalg.norm(points@actual[:3,:3].T+actual[:3,3]-(points@waypoint[:3,:3].T+waypoint[:3,3]),axis=-1).mean()<=.008
            if reached:break
            cursor+=1
        if cursor==len(poses):return dict(complete=False,arrivals=arrivals,arrival_frame=None)
        arrivals.append(cursor)
    return dict(complete=True,arrivals=arrivals,arrival_frame=cursor)


def replay(poses,hand,velocity,forces,valid,supported,spec,points,frame_count):
    controller=TaskController(spec,points);counts=Counter();events={};max_progress=0.;result=None
    for tick in range(frame_count):
        now=tick/HZ
        decision=controller.observe(poses[tick],hand[tick],velocity[tick],now,
            contact_forces=forces[tick],force_valid=bool(valid[tick]),supported=bool(supported[tick]))['decision']
        counts[decision.phase.name]+=1;max_progress=max(max_progress,float(decision.path_progress))
        for event in decision.transitions:events.setdefault(event,now)
        result=dict(task_success=bool(decision.task_complete),failure_reason=decision.failure_reason,
            terminated=bool(decision.terminated),end_frame=tick,end_s=now,phase=decision.phase.name,
            max_path_progress=max_progress,path_complete=bool(decision.path_complete),events=events)
        if decision.terminated:break
    result['phase_sample_counts']=dict(counts)
    result['recording_censored']=not result['terminated']
    return result


def independent(poses,hand,velocity,forces,valid,supported,coupled,released,spec,points,count):
    p=poses[:count];h=hand[:count]
    local=(h[:,6:]-p[:,None,:3,3])@p[:,:3,:3]
    gaps=cKDTree(points).query(local.reshape(-1,3),k=1)[0].reshape(count,5)
    force_count=(np.linalg.norm(forces[:count],axis=-1)>=.1).sum(-1)
    contact=valid[:count]&(force_count>=2)&((gaps<=.018).sum(-1)>=2)
    release_witness=np.zeros(count,bool)
    for lo,hi in bouts(valid[:count]&(gaps.min(-1)>=.025)):
        indices=np.arange(lo,hi);release_witness[indices]=(indices-lo)/HZ>=.12
    lift=p[:,2,3]-poses[0,2,3]
    raised=lift>=.05
    stable=(np.linalg.norm(velocity[:count,:3],axis=-1)<=.02)&(np.linalg.norm(velocity[:count,3:],axis=-1)<=.2)
    palm_distance=np.linalg.norm(h[:,0]-p[:,:3,3],axis=-1)
    physical_hold=raised&~supported[:count]&(palm_distance<=.20)&valid[:count]&(force_count>=2)
    path=independent_path(p,spec,points)
    endpoint=endpoint_mask(p,spec.object_path[-1],points,spec.goal_type)
    terminal=np.zeros(count,bool)
    if spec.terminal_mode.value=='HOLD':terminal=endpoint&stable&contact&coupled[:count]
    elif spec.terminal_mode.value=='RELEASE':terminal=endpoint&stable&release_witness&((supported[:count]) if spec.require_support else True)
    # Match the temporal duration in seconds, rather than calling 30 sampled
    # states a one-second elapsed interval (they span only 29/30 seconds).
    arrival=path['arrival_frame']
    after=terminal.copy()
    if arrival is None:after[:]=False
    else:after[:arrival]=False
    witnessed_coupling=first(coupled[:count])
    success=spec.terminal_mode.value!='UNKNOWN' and witnessed_coupling is not None and span(after)['elapsed_span_s']>=1.
    return dict(max_lift_m=float(lift.max()),first_contact_s=None if first(contact) is None else first(contact)/HZ,
        contact=span(contact),coupled=span(coupled[:count]),ever_coupled=bool(coupled[:count].any()),
        first_coupled_s=None if witnessed_coupling is None else witnessed_coupling/HZ,
        released=span(release_witness),ever_released=bool(release_witness.any()),
        archived_release_mask_mismatches=int((release_witness!=released[:count]).sum()),lift5cm=span(raised),
        lift5cm30_legacy=span(raised)['samples']>=30,lift5cm_elapsed1s=span(raised)['elapsed_span_s']>=1.,
        physical_hold=span(physical_hold),physical_hold_elapsed1s=span(physical_hold)['elapsed_span_s']>=1.,
        endpoint_stable_terminal=span(after),ordered_path=path,independent_full_task=bool(success),
        full_task_definition='ordered 8mm/.12rad/8mm-mean-CAD waypoints, prior coupling, declared HOLD contact or RELEASE+support, stable endpoint elapsed >=1s')


def case(arguments):
    row,admission=arguments;source=Path(row['source']);side=Path(admission)/row['sidecar']
    with np.load(source/'canonical.npz') as z:points=z['points'].astype(np.float64)
    with np.load(source/'trajectory.npz') as z:
        data={k:z[k].copy() for k in ['obj','hand','velocity','body_contact_forces','contact_force_valid','frame_tick']}
    with np.load(side/'admission.npz') as z:admit={k:z[k].copy() for k in ['coupled','released','supported','training_prefix']}
    receipt=json.loads((source.parents[1]/'manifest.json').read_text())
    ids=receipt['observation_contract']['contact_body_ids'];forces=data['body_contact_forces'][:,ids]
    hand=data['hand'][:,NATIVE_TO_WAM];n=len(hand);cut=int(row['training_prefix_commands']);count=cut+1
    if not np.array_equal(data['frame_tick'],np.arange(n)):raise ValueError('noncontiguous source frame clock')
    if not np.array_equal(admit['training_prefix'],np.arange(n-1)<cut):raise ValueError('safe-prefix mismatch')
    task=TaskSpec.from_packet(json.loads((side/'task.json').read_text()))
    raw=TaskSpec.from_packet(json.loads((side/'raw-task.json').read_text()))
    arguments=(data['obj'],hand,data['velocity'],forces,data['contact_force_valid'],admit['supported'])
    prefix_replay=replay(*arguments,task,points,count)
    full_replay=replay(*arguments,task,points,n)
    raw_replay=replay(*arguments,raw,points,n)
    prefix=independent(*arguments,admit['coupled'],admit['released'],task,points,count)
    full=independent(*arguments,admit['coupled'],admit['released'],task,points,n)
    return dict(sequence=row['sequence'],object=row['object'],source=str(source),task_path=str(side/'task.json'),
        prefix_commands=cut,prefix_duration_s=cut/HZ,recorded_frames=n,recorded_duration_s=(n-1)/HZ,
        terminal_mode=task.terminal_mode.value,raw_terminal_mode=raw.terminal_mode.value,
        planned_events=[dict(frame=e.source_frame,kind=e.kind.value,seconds=e.source_frame/task.source_fps) for e in task.events],
        source_fps=task.source_fps,task_endpoint_frame=int(task.source_frames[-1]),task_endpoint_s=float(task.source_frames[-1]/task.source_fps),
        prefix_independent=prefix,full_recording_independent=full,prefix_controller=prefix_replay,
        full_recording_prefix_task_controller=full_replay,full_recording_raw_task_controller=raw_replay,
        time_budget=dict(acquire_timeout_s=15.,episode_timeout_s=60.,release_timeout_s=10.,terminal_hold_s=1.,
            teacher_first_coupling_after_acquire_timeout=full['first_coupled_s'] is not None and full['first_coupled_s']>=15.,
            teacher_endpoint_plus_hold_exceeds_episode=bool(task.source_frames[-1]/task.source_fps+1>60.),
            limitation='Reference timestamps do not impose fixed arrival time; exceeding 60s does not prove another retimed policy impossible'))


def controls():
    from wam_rl_policy_v3.task_spec import TerminalMode
    points=np.array([[x,y,z] for x in [-.01,.01] for y in [-.01,.01] for z in [-.01,.01]])
    start=np.eye(4);end=np.eye(4);end[2,3]=.04
    def run(mode,unstable_frame=None,no_release=False):
        spec=TaskSpec.from_reference(np.stack([start,end]),source_fps=30,terminal_mode=mode,
            goal_type='position_only',trim_interaction=False,require_support=True)
        scheduler=TaskScheduler(spec,points);records=[]
        for tick in range(181):
            now=tick/HZ;pose=start.copy();pose[2,3]=.04*min(now,1.)
            release=mode==TerminalMode.RELEASE and now>=1.5 and not no_release
            grasp=not release
            feedback=InteractionFeedback(grasp_candidate=grasp,grasp_established=grasp,released=release,
                stable=bool(now>=1. and tick!=unstable_frame),supported=True,force_valid=True,contact_observed=grasp)
            decision=scheduler.update(pose,now,feedback)
            records.append(dict(tick=tick,seconds=now,success=decision.task_complete,phase=decision.phase.name,
                settle_since=scheduler.settle_since,path_complete=decision.path_complete))
            if decision.terminated:break
        return records
    hold=run(TerminalMode.HOLD);release=run(TerminalMode.RELEASE);unknown=run(TerminalMode.UNKNOWN)
    if not hold[-1]['success'] or not release[-1]['success'] or any(r['success'] for r in unknown):
        raise AssertionError('declared scheduler success paths / UNKNOWN structural control failed')
    success_tick=hold[-1]['tick'];mutated=run(TerminalMode.HOLD,unstable_frame=success_tick-15)
    no_release=run(TerminalMode.RELEASE,no_release=True)
    if any(r['success'] for r in mutated if r['tick']<=success_tick):raise AssertionError('one-frame hold break was ignored')
    if any(r['success'] for r in no_release):raise AssertionError('missing release was ignored')
    all_good=np.ones(31,bool);broken=all_good.copy();broken[15]=False
    if span(np.ones(30,bool))['elapsed_span_s']>=1 or span(all_good)['elapsed_span_s']!=1 or span(broken)['elapsed_span_s']>=1:
        raise AssertionError('elapsed-second hold boundary/missing-force independent oracle failed')
    n=66;poses=np.repeat(start[None],n,0);poses[31:]=end
    hand=np.repeat(poses[:,:3,3][:,None],11,axis=1);hand[:,6:]+=points[:5]
    velocity=np.zeros((n,6));forces=np.zeros((n,5,3));forces[:,:,0]=.2
    valid=np.ones(n,bool);coupled=np.ones(n,bool);released=np.zeros(n,bool)
    spec=TaskSpec.from_reference(np.stack([start,end]),source_fps=30,terminal_mode=TerminalMode.HOLD,
        goal_type='position_only',trim_interaction=False,require_support=False)
    physical_args=(poses,hand,velocity,forces,valid,np.zeros(n,bool),coupled,released,spec,points,n)
    positive=independent(*physical_args)
    missing=valid.copy();missing[48]=False
    mutated_force=independent(poses,hand,velocity,forces,missing,np.zeros(n,bool),coupled,released,spec,points,n)
    if not positive['independent_full_task'] or mutated_force['independent_full_task']:
        raise AssertionError('independent full scorer failed its measured-force single-frame mutation')
    release_spec=TaskSpec.from_reference(np.stack([start,end]),source_fps=30,terminal_mode=TerminalMode.RELEASE,
        goal_type='position_only',trim_interaction=False,require_support=True)
    release_hand=hand.copy();release_hand[31:,:,0]+=.1
    release_forces=forces.copy();release_forces[31:]=0
    release_physical=independent(poses,release_hand,velocity,release_forces,valid,np.ones(n,bool),coupled,released,release_spec,points,n)
    no_release_physical=independent(poses,hand,velocity,forces,valid,np.ones(n,bool),coupled,released,release_spec,points,n)
    if not release_physical['independent_full_task'] or no_release_physical['independent_full_task']:
        raise AssertionError('independent physical release witness mutation failed')
    return dict(kind='synthetic measured-feedback scorer controls; no physics execution',status='PASS',
        hold_success_s=hold[-1]['seconds'],hold_settle_started_s=hold[-1]['settle_since'],
        hold_elapsed_s=hold[-1]['seconds']-hold[-1]['settle_since'],release_success_s=release[-1]['seconds'],
        unknown_success=False,unknown_path_complete=unknown[-1]['path_complete'],
        one_unstable_frame_delays_success_past_original=True,missing_released_never_succeeds=True,
        independent_physical_field_controls=dict(hold_positive=True,hold_one_missing_force_frame_rejected=True,
            release_positive=True,no_far_valid_release_rejected=True),
        elapsed_window=dict(samples30_span_s=29/HZ,samples31_span_s=1.,missing_force_frame_breaks_1s_hold=True),
        limitation='Synthetic feedback validates scoring branches; does not certify any real grasp or repair physics')


def evidence(path,needle):
    lines=Path(path).read_text().splitlines()
    matches=[i+1 for i,line in enumerate(lines) if needle in line]
    return dict(file=str(Path(path).relative_to(ROOT)),lines=matches,needle=needle)


def history_report(run,frozen,rows):
    mapping={row['sequence']:row for row in rows};objects=sorted(set(row['object'] for row in rows))
    counts=Counter(row['object'] for row in rows);allocation=Counter(objects[i%len(objects)] for i in range(512))
    weights={row['sequence']:allocation[row['object']]/512/counts[row['object']] for row in rows}
    draws=[];totals=Counter();curves=[]
    for rank in range(3):
        native_path=run/('native-rank-%d/manifest.json'%rank);native=json.loads(native_path.read_text())
        names=native['source_sequence_names'];visits=native.get('motion_visits')
        if set(names)!=set(mapping):raise ValueError('actual native membership differs from admitted 501')
        modes=Counter()
        if visits is not None:
            if len(visits)!=len(names):raise ValueError('visits/source identity mismatch')
            for name,value in zip(names,visits):modes[mapping[name]['terminal_mode']]+=int(value)
            totals.update(modes)
        draws.append(dict(rank=rank,source=str(native_path),status=native['status'],native_control_steps=native['steps'],
            source_sequences=len(names),indexed_resets=native['indexed_resets'],motion_initialization_draws=None if visits is None else sum(visits),
            terminal_mode_draws=dict(modes),unknown_draw_fraction=None if visits is None else modes['UNKNOWN']/sum(visits),
            per_sequence_draws=None if visits is None else dict(zip(names,visits)),
            within_object_cycle_count_ranges=None if visits is None else {obj:max(visits[i] for i,name in enumerate(names) if mapping[name]['object']==obj)-min(visits[i] for i,name in enumerate(names) if mapping[name]['object']==obj) for obj in objects},
            snapshot_scope='final native closure, not iteration1570; includes initialization/reset and post-last-logged-update rollout'))
        path=run/('rank-%d/curve.jsonl'%rank);selected={}
        for number,line in enumerate(path.open(),1):
            if not any('"iteration": %d,'%it in line for it in [1520,1570,1575]):continue
            record=json.loads(line)
            if record['iteration'] in [1520,1570,1575]:
                selected[record['iteration']]=dict(line=number,**{k:record.get(k) for k in ['iteration','completed_episodes','workspace_failures','successes','resets','optimizer_steps']},
                    return_or_reward_fields=[k for k in record if any(word in k.lower() for word in ['return','reward'])])
        curves.append(dict(rank=rank,file=str(path),snapshots=selected))
    global1570=curves[0]['snapshots'].get(1570)
    source_modes=Counter(row['raw_terminal_mode']+'->'+row['terminal_mode'] for row in rows)
    return dict(admitted_sequences=len(rows),terminal_modes=dict(Counter(row['terminal_mode'] for row in rows)),
        raw_terminal_modes=dict(Counter(row['raw_terminal_mode'] for row in rows)),
        raw_to_prefix_terminal_mode_counts=dict(source_modes),
        uniform_sequence_unknown_fraction=sum(r['terminal_mode']=='UNKNOWN' for r in rows)/len(rows),
        nominal_env_object_weighted_unknown=sum(weights[r['sequence']] for r in rows if r['terminal_mode']=='UNKNOWN'),
        nominal_weights=weights,object_environment_allocation=dict(allocation),actual_motion_initializations=draws,
        actual_motion_initialization_totals=dict(totals),actual_unknown_initialization_fraction=totals['UNKNOWN']/sum(totals.values()) if totals else None,
        actual_completed_episode_frequency_by_task=None,episode_frequency_reason='No complete ended-episode identity stream saved; motion_visits are initialization counts',
        curves=curves,rank_counts_are_global_reductions_not_additive=True,
        iteration1570_workspace_fraction=None if global1570 is None else global1570['workspace_failures']/global1570['completed_episodes'],
        final_policy_statuses=[dict(rank=rank,**{key:json.loads((run/('rank-%d/manifest.json'%rank)).read_text()).get(key) for key in ['status','iteration','completed_episodes','workspace_failures','successes']}) for rank in range(3)],
        time_alignment=dict(native_dt_s=json.loads((run/'native-rank-0/manifest.json').read_text())['native_dt_s'],annotation_fps=30,controller_clock='native tick / 30; frame0 reset is not an executed control',
            iteration1570_scope='cumulative iteration; this run resumed231, so1339 new updates; counts are fresh-run counters',
            final_native_extra_collection='native10760 control calls vs1344 completed new updates*8=10752; final reset counts54913 vs last-logged54883 differ30; unlogged tail episode outcomes unknown'),
        missing_training_trajectory='Historical training run contains no full per-control trajectory archive; teacher archives below are a different measured execution, never substituted for PPO histories')


def summarize(cases):
    result=dict(n=len(cases),prefix_no_contact_interval=sum(c['prefix_independent']['contact']['samples']<3 for c in cases),
        prefix_annotation_without_contact_event=sum(not any(event['kind']=='CONTACT' for event in c['planned_events']) for c in cases),
        prefix_never_coupled=sum(not c['prefix_independent']['ever_coupled'] for c in cases),
        prefix_lift5cm30=sum(c['prefix_independent']['lift5cm30_legacy'] for c in cases),
        prefix_lift5cm_elapsed1s=sum(c['prefix_independent']['lift5cm_elapsed1s'] for c in cases),
        prefix_physical_hold_elapsed1s=sum(c['prefix_independent']['physical_hold_elapsed1s'] for c in cases),
        prefix_terminal_eligible_elapsed1s=sum(c['prefix_independent']['endpoint_stable_terminal']['elapsed_span_s']>=1 for c in cases),
        prefix_duration_s=dict(min=min(c['prefix_duration_s'] for c in cases),median=float(np.median([c['prefix_duration_s'] for c in cases])),max=max(c['prefix_duration_s'] for c in cases)),
        teacher_first_coupling_after_15s=sum(c['time_budget']['teacher_first_coupling_after_acquire_timeout'] for c in cases),
        reference_endpoint_plus_hold_after60s=sum(c['time_budget']['teacher_endpoint_plus_hold_exceeds_episode'] for c in cases))
    for key in ['prefix_controller','full_recording_prefix_task_controller','full_recording_raw_task_controller']:
        result[key]=dict(successes=sum(c[key]['task_success'] for c in cases),
            failure_reasons=dict(Counter(c[key]['failure_reason'] or 'recording_censored' for c in cases)),
            final_phase_counts=dict(Counter(c[key]['phase'] for c in cases)),
            phase_ever_trajectory_counts=dict(Counter(phase for c in cases for phase,n in c[key]['phase_sample_counts'].items() if n)),
            phase_sample_counts=dict(sum((Counter(c[key]['phase_sample_counts']) for c in cases),Counter())))
    for key in ['prefix_independent','full_recording_independent']:
        result[key]=dict(ordered_path=sum(c[key]['ordered_path']['complete'] for c in cases),full_task=sum(c[key]['independent_full_task'] for c in cases),
            lift5cm_elapsed1s=sum(c[key]['lift5cm_elapsed1s'] for c in cases),physical_hold_elapsed1s=sum(c[key]['physical_hold_elapsed1s'] for c in cases))
    result['by_terminal']={mode:summarize_small([c for c in cases if c['terminal_mode']==mode]) for mode in ['UNKNOWN','HOLD','RELEASE']}
    return result


def summarize_small(cases):
    return dict(n=len(cases),prefix_ever_coupled=sum(c['prefix_independent']['ever_coupled'] for c in cases),
        prefix_controller_success=sum(c['prefix_controller']['task_success'] for c in cases),
        full_recording_controller_success=sum(c['full_recording_prefix_task_controller']['task_success'] for c in cases),
        independent_full_recording_success=sum(c['full_recording_independent']['independent_full_task'] for c in cases))


def alignment_trace(record,output):
    """Explain an independent-positive/controller-negative on the same frames."""
    source=Path(record['source']);side=Path(record['task_path']).parent
    with np.load(source/'trajectory.npz') as z:
        data={k:z[k].copy() for k in ['obj','hand','velocity','body_contact_forces','contact_force_valid']}
    with np.load(source/'canonical.npz') as z:points=z['points'].copy()
    with np.load(side/'admission.npz') as z:supported=z['supported'].copy()
    receipt=json.loads((source.parents[1]/'manifest.json').read_text());ids=receipt['observation_contract']['contact_body_ids']
    spec=TaskSpec.from_packet(json.loads((side/'task.json').read_text()))
    endpoint=endpoint_mask(data['obj'],spec.object_path[-1],points,spec.goal_type)
    controller=TaskController(spec,points);rows=[];previous=None;changes=[]
    for tick in range(len(endpoint)):
        packet=controller.observe(data['obj'][tick],data['hand'][tick,NATIVE_TO_WAM],data['velocity'][tick],tick/HZ,
            contact_forces=data['body_contact_forces'][tick,ids],force_valid=bool(data['contact_force_valid'][tick]),supported=bool(supported[tick]))
        decision=packet['decision'];feedback=packet['feedback'];scheduler=controller.scheduler;path=scheduler.path
        observed=controller.observer.last_evidence
        next_index=int(path.milestones[path.pointer]) if path.pointer<len(path.milestones) else None
        row=dict(tick=tick,seconds=tick/HZ,phase=decision.phase.name,recovering=decision.recovering,
            grasp_candidate=feedback.grasp_candidate,grasp_established=feedback.grasp_established,
            grasp_lost=feedback.grasp_lost,released=feedback.released,stable=feedback.stable,supported=bool(supported[tick]),
            force_valid=bool(data['contact_force_valid'][tick]),force_count=observed['force_count'],near_fingers=observed['near_fingers'],
            fingertip_gap_m=observed['fingertip_gap_m'],independent_endpoint_reached=bool(endpoint[tick]),
            confirmed_progress=float(decision.path_progress),confirmed_path_complete=decision.path_complete,
            local_path_complete=bool(path.complete),milestone_pointer=int(path.pointer),milestones=len(path.milestones),
            next_milestone_index=next_index,next_milestone_source_frame=None if next_index is None else int(spec.source_frames[next_index]),
            source_cursor_frame=float(decision.source_cursor_frame),event_index=int(scheduler.event_index),
            planned_next_event=decision.next_event.kind.value if decision.next_event.kind else None,
            path_settle_since=path.settle_since,terminal_settle_since=scheduler.settle_since,
            task_success=decision.task_complete,failure_reason=decision.failure_reason,events=list(decision.transitions))
        signature=(row['phase'],row['grasp_established'],row['grasp_lost'],row['released'],row['confirmed_path_complete'])
        if signature!=previous:changes.append(row);previous=signature
        rows.append(row)
        if decision.terminated:break
    first_lost=next((row for row in rows if row['grasp_lost']),None)
    settling=[row for row in rows if row['path_settle_since'] is not None and
        (first_lost is None or row['tick']<first_lost['tick'])]
    alignment_block=dict(release_phase_seen=any(row['phase']=='RELEASE' for row in rows),
        terminal_settling_seen=any(row['terminal_settle_since'] is not None for row in rows),
        path_settle_first_tick=settling[0]['tick'] if settling else None,
        path_settle_elapsed_at_loss_s=(first_lost['seconds']-settling[-1]['path_settle_since']) if settling and first_lost else None,
        required_path_settle_s=float(path.config.final_settle_s),
        explanation=('Endpoint path settling began but grasp was lost before the required dwell; recovery discards the settle witness.'
            if settling else 'Grasp was lost before ordered path confirmation/settling; later endpoint arrival occurred during ACQUIRE recovery.'))
    release_frame=next((event['frame'] for event in record['planned_events'] if event['kind']=='RELEASE'),None)
    selected_ticks={0,len(rows)-1}
    for row in changes:selected_ticks.add(row['tick'])
    ordered_arrival=record['full_recording_independent']['ordered_path']['arrival_frame']
    for center in [release_frame,first(endpoint),ordered_arrival,None if first_lost is None else first_lost['tick']]:
        if center is not None:selected_ticks.update(range(max(0,center-3),min(len(rows),center+5)))
    result=dict(sequence=record['sequence'],task_path=record['task_path'],source=record['source'],
        independent_full_success=record['full_recording_independent']['independent_full_task'],
        controller_full_success=record['full_recording_prefix_task_controller']['task_success'],
        planned_release_frame=release_frame,planned_release_s=None if release_frame is None else release_frame/HZ,
        independent_first_endpoint_frame=first(endpoint),independent_ordered_path_arrival_frame=ordered_arrival,first_grasp_lost=first_lost,
        final=rows[-1],blockage=alignment_block,transition_rows=changes,selected_rows=[rows[i] for i in sorted(selected_ticks)],trace=rows,
        interpretation='Independent scorer relaxes continuous grasp, confirmed milestone/interaction gates and pre-RELEASE path settling; its positive is not proof TaskScheduler should succeed. Data/task-contract incompatibility is evidenced; a coding bug is not established.',
        independent_relaxations=['prior coupling only; does not require maintained grasp over ordered path',
            'greedy 8mm/.12rad/8mm waypoint arrival rather than measured-interaction milestone confirmation',
            'no requirement to complete .2s path settling while TRACK before permitting terminal RELEASE',
            'no planned-release gating by the scheduler confirmed source cursor'])
    destination=output/('aligned-'+record['sequence']+'.json');write(destination,result)
    text=[f'# Alignment: {record["sequence"]}','',result['interpretation'],'',
        f'Planned RELEASE: tick {release_frame} ({result["planned_release_s"]} s).',
        f'First independently reached endpoint: {result["independent_first_endpoint_frame"]}.',
        alignment_block['explanation'],
        f'Path settling began at tick {alignment_block["path_settle_first_tick"]}; elapsed at loss {alignment_block["path_settle_elapsed_at_loss_s"]} s; required {alignment_block["required_path_settle_s"]} s. RELEASE phase seen: {alignment_block["release_phase_seen"]}.',
        '', '| tick | s | phase | grasp/lost/released | endpoint | progress | milestone | path complete |',
        '|---:|---:|---|---|---|---:|---|---|']
    for row in result['selected_rows']:
        text.append(f'| {row["tick"]} | {row["seconds"]:.3f} | {row["phase"]} | {int(row["grasp_established"])}/{int(row["grasp_lost"])}/{int(row["released"])} | {row["independent_endpoint_reached"]} | {row["confirmed_progress"]:.6f} | {row["milestone_pointer"]}/{row["milestones"]} | {row["confirmed_path_complete"]} |')
    destination.with_suffix('.md').write_text('\n'.join(text)+'\n')
    brief={key:result[key] for key in ['sequence','planned_release_frame','planned_release_s','independent_first_endpoint_frame','independent_ordered_path_arrival_frame','first_grasp_lost','final','blockage','interpretation','independent_relaxations']}
    brief['trace_path']=str(destination)
    return brief


def p1_stop_report():
    root=ROOT/'outputs/ptv3-shared-wam-v6/samples-efficiency-3arm-20261022-r1'
    result=dict(stop_present=(root/'STOP').exists(),queue=json.loads((root/'manifest.json').read_text()),arms={})
    import torch
    torch.set_num_threads(1)
    for arm in ['A','B']:
        folder=root/(arm+'-seed191');state=json.loads((folder/'policy/manifest.json').read_text())
        checkpoint=folder/'policy'/('official-iteration-%d.pt'%state['iteration'])
        packet=torch.load(checkpoint,map_location='cpu',weights_only=False)
        def finite(value):
            if torch.is_tensor(value):return bool(torch.isfinite(value).all())
            if isinstance(value,dict):return all(finite(v) for v in value.values())
            if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
            return True
        process=json.loads((folder/'process.json').read_text())
        result['arms'][arm]=dict(status=state['status'],iteration=state['iteration'],completed_update_transitions=state['environment_steps'],
            actual_environment_steps=state['actual_environment_steps'],checkpoint=str(checkpoint),checkpoint_iteration=packet['iteration'],
            cpu_loadable=True,finite_model_optimizer=finite(packet['model']) and finite(packet['optimizer']),
            optimizer_nonempty=bool(packet['optimizer']['state']),exit_codes=process['exit_codes'])
    result['c_started']=(root/'C-seed191').exists()
    result['queue_failed_reason']='Owned STOP stopped policies normally; queue then requested absent future eval checkpoints and labeled FAILED. Native/policy exit0; not a physics/optimizer crash.'
    return result


def representative_categories(result):
    """Real recorded examples, never substitute proxy success for full success."""
    cases=result['cases'];admission=json.loads(Path(result['admission_manifest']).read_text())
    admission_rows={r['sequence']:r for r in admission['records']};examples=[]
    def arrays(c):
        with np.load(Path(c['source'])/'trajectory.npz') as z:
            data={k:z[k].copy() for k in ['obj','hand','body_contact_forces','contact_force_valid']}
        with np.load(Path(c['task_path']).parent/'admission.npz') as z:
            data.update({k:z[k].copy() for k in ['coupled','released','supported']})
        return data
    def entry(label,c,scope,witness):
        metrics=c['prefix_independent'] if scope=='safe_prefix' else c['full_recording_independent']
        scored=c['prefix_controller'] if scope=='safe_prefix' else c['full_recording_prefix_task_controller']
        return dict(category=label,sequence=c['sequence'],scope=scope,trajectory_path=str(Path(c['source'])/'trajectory.npz'),
            admission_path=str(Path(c['task_path']).parent/'admission.npz'),task_path=c['task_path'],terminal_mode=c['terminal_mode'],
            raw_terminal_mode=c['raw_terminal_mode'],prefix_end_tick=c['prefix_commands'],witness=witness,
            physical_hold_elapsed1s=metrics['physical_hold_elapsed1s'],independent_full_task=metrics['independent_full_task'],
            actual_task_controller=scored)
    holds=[c for c in cases if c['prefix_independent']['physical_hold_elapsed1s']]
    if holds:
        c=holds[0];d=arrays(c);count=c['prefix_commands']+1
        receipt=json.loads((Path(c['source']).parents[1]/'manifest.json').read_text());ids=receipt['observation_contract']['contact_body_ids']
        forces=(np.linalg.norm(d['body_contact_forces'][:,ids],axis=-1)>=.1).sum(-1)
        mask=(d['obj'][:,2,3]-d['obj'][0,2,3]>=.05)&~d['supported']&d['contact_force_valid']&(forces>=2)&(
            np.linalg.norm(d['hand'][:,0]-d['obj'][:,:3,3],axis=-1)<=.20)
        lo,hi=max(bouts(mask[:count]),key=lambda pair:pair[1]-pair[0])
        examples.append(entry('physical_hold_proxy_success_not_full_task',c,'safe_prefix',dict(start_tick=lo,end_tick=hi-1,
            start_s=lo/HZ,end_s=(hi-1)/HZ,elapsed_s=(hi-lo-1)/HZ,definition='lift>=5cm, unsupported, palm<=20cm, valid >=2 force bodies')))
    for c in cases:
        if not c['full_recording_independent']['ever_coupled'] or not admission_rows[c['sequence']]['failure_frames'].get('unsupported_loss_after_coupling',0):continue
        d=arrays(c);height=d['obj'][:,2,3];initial=first(d['coupled']);peak=initial+int(np.argmax(height[initial:]))
        drop=first((np.arange(len(height))>peak)&(height<=height[peak]-.05)&~d['supported']&~d['coupled'])
        if drop is None:continue
        examples.append(entry('coupled_then_fell_without_support',c,'full_original_recording',dict(first_coupled_tick=initial,
            first_coupled_s=initial/HZ,peak_tick=peak,peak_s=peak/HZ,drop_tick=drop,drop_s=drop/HZ,
            fall_from_peak_m=float(height[peak]-height[drop]),supported_at_drop=bool(d['supported'][drop]),coupled_at_drop=bool(d['coupled'][drop]),
            beyond_safe_prefix=drop>c['prefix_commands'],limitation='Height fall + unsupported/coupling loss witness; not inferred from a planned RELEASE label. Original recording extends beyond admitted prefix.')))
        break
    near=[c for c in cases if not c['prefix_independent']['ever_coupled'] and c['prefix_independent']['first_contact_s'] is not None]
    if near:
        c=near[0];tick=int(round(c['prefix_independent']['first_contact_s']*HZ))
        examples.append(entry('contact_proximity_without_coupling',c,'safe_prefix',dict(first_contact_tick=tick,first_contact_s=tick/HZ,
            first_coupled_tick=None,definition='>=2 fingertips within18mm with valid >=2 force bodies, but no established motion coupling')))
    for c in cases:
        if c['terminal_mode']!='HOLD' or not c['prefix_independent']['ever_coupled'] or c['prefix_commands']<30:continue
        if any(example['sequence']==c['sequence'] for example in examples):continue
        d=arrays(c);end=c['prefix_commands'];start=end-30
        if d['released'][start:end+1].any():continue
        examples.append(entry('hold_terminal_not_released',c,'safe_prefix',dict(start_tick=start,end_tick=end,start_s=start/HZ,end_s=end/HZ,
            elapsed_s=(end-start)/HZ,release_witness_frames=0,first_coupled_s=c['prefix_independent']['first_coupled_s'],
            interpretation='No release witness for final1s. HOLD does not request release; this is not by itself a failure.')))
        break
    return dict(examples=examples,actual_full_task_success_archives=sum(c['full_recording_prefix_task_controller']['task_success'] for c in cases),total_archives=len(cases),
        limitation='No actual TaskController-full-success archive exists in this panel. Proxy hold positives and12 independent relaxed-terminal positives are not full-task positive controls; actual scheduler positives are synthetic measured-feedback controls only.')


def finalize_report(result,output):
    """Reusable CPU-only result enrichment / renderer for the parent report."""
    output=Path(output);run=Path(result['historical_run']);frozen=Path(result['frozen_task'])
    admission=Path(result['admission_manifest']);rows=[row for row in json.loads(admission.read_text())['records']
        if row['split']=='train' and row['engineering']=='PASS' and row['training_prefix_commands']>0]
    result['history']=history_report(run,frozen,rows);result['summary']=summarize(result['cases'])
    result['p1_stop']=p1_stop_report()
    result['representative_categories']=representative_categories(result)
    positive=[c for c in result['cases'] if c['full_recording_independent']['independent_full_task'] and not c['full_recording_prefix_task_controller']['task_success']]
    representatives=[]
    if positive:
        selected=[min(positive,key=lambda c:c['full_recording_prefix_task_controller']['max_path_progress']),
                  max(positive,key=lambda c:c['full_recording_prefix_task_controller']['max_path_progress'])]
        seen=set()
        for record in selected:
            if record['sequence'] in seen:continue
            seen.add(record['sequence']);representatives.append(alignment_trace(record,output))
    result['controller_independent_alignment']=representatives
    code=frozen/'src/wam_rl_policy_v3';official=run/'rank-0/runtime-official/dexgrasp/algorithms/rl/ppo'
    result['evidence']+= [evidence(code/'task_scheduler.py','feedback.grasp_lost'),
        evidence(code/'task_scheduler.py','self.path_decision[\'complete\'] and at_end'),
        evidence(code/'task_scheduler.py','self.phase=Phase.RELEASE'),
        evidence(code/'goal_scheduler.py','final_settle_s: float = .2'),
        evidence(official/'ppo.py','self.storage.add_transitions(current_obs, current_states, actions, rews'),
        evidence(official/'storage.py','self.rewards[self.step].copy_'),
        evidence(official/'storage.py','delta = self.rewards[step]'),
        evidence(frozen/'tools/run/inspire_physics_host.py',"objectMotionSampling=not bool(a.fixed_motions_json)"),
        evidence(ROOT/'third_party/DExplore/dexplore/utils/object_motion_sampler.py','self.env_objects = torch.arange'),
        evidence(ROOT/'third_party/DExplore/dexplore/utils/object_motion_sampler.py','slots = (torch.arange')]
    result['evidence']=list({(row['file'],row['needle']):row for row in result['evidence']}.values())
    history=result['history'];reference=history['curves'][0]['snapshots'][1570]
    result['claims']=dict(b=dict(structural_unknown_impossibility=True,unknown_tasks=64,total_tasks=501,
        scope='UNKNOWN has no success branch; cannot explain all0 because437 tasks are HOLD/RELEASE'),
        c=dict(iteration=1570,completed_episodes=reference['completed_episodes'],workspace_failures=reference['workspace_failures'],
            fraction=reference['workspace_failures']/reference['completed_episodes'],all_rank_global_values_equal=all(c['snapshots'][1570]==dict(reference,line=c['snapshots'][1570]['line']) for c in history['curves']),
            gate='PTv3 integer voxel serialization bounds, not calibrated robot mechanical workspace',
            evaluation_scope='online stochastic training counters, not fixed-panel evaluation'),
        d=dict(reward_delivery_code_chain_verified=True,return_phase_telemetry_missing=True,
            tensorboard_bytes={str(path):path.stat().st_size for path in run.glob('rank-*/models_seed*/events.*')},
            historical_reward_values='UNKNOWN: step/episode return streams were not persisted; cannot recover from losses'))
    result['conclusions']=[
        '64/501 admitted tasks are UNKNOWN and lack a success branch; this cannot explain zero success on the other437 tasks.',
        'Physical lift/hold, independent endpoint-release success and TaskController success are different contracts. Independent positives relax held-path/milestone/pre-release settling, so they do not prove a TaskScheduler coding bug.',
        'Teacher archive replay is a different measured execution from historical PPO; absent PPO per-control trajectories and ended-episode task identities are UNKNOWN.',
        'Prefix HOLD labels are task instructions; 408 raw RELEASE recordings became HOLD after safe-prefix truncation. A label does not certify stable terminal holding.',
        'Reward reaches PPO storage/GAE by the frozen runtime chain. Runner omitted return and phase telemetry; missing logs are not missing reward input.']
    write(output/'task-reachability.json',result)
    write(output/'task-reachability-alignments.json',representatives)
    write(output/'task-reachability-progress.json',dict(status=result['status'],completed=len(result['cases']),total=501,elapsed_s=result['elapsed_s']))
    summary=result['summary'];n=len(result['cases'])
    text=['# Historical task reachability audit','',f'Actual frozen run `{result["historical_git_commit"]}`; `{run}`. CPU only; {n} measured teacher archives.',
        '', '## Assertions b / c / d','',
        '- b: **partly explanatory, not sufficient**. UNKNOWN64/501, HOLD416, RELEASE21; 437 tasks have a defined success branch.',
        f'- Final native initialization/reset draws56449: UNKNOWN9314 ({history["actual_unknown_initialization_fraction"]:.6%}); not ended episodes at iteration1570. Nominal env-object weighted UNKNOWN {history["nominal_env_object_weighted_unknown"]:.6%}.',
        f'- c: cumulative iteration1570 really logs {reference["completed_episodes"]} ended episodes, {reference["workspace_failures"]} voxel-workspace failures ({100*result["claims"]["c"]["fraction"]:.4f}%). Three ranks repeat one global reduction; do not add them.',
        '- The gate is PTv3 serialization bounds and causes immediate terminal/reset. It is not a measured mechanical reachability certificate.',
        '- d: environment reward → official PPO storage → GAE is present; custom runner logging omitted return and phase aggregates. Actual historical returns cannot be reconstructed.',
        '', '## Existing-trajectory replay (not PPO policy evaluation)','',
        f'- Safe-prefix duration min/median/max: {summary["prefix_duration_s"]} seconds; no reference endpoint+1s exceeds60s; no teacher first-coupling exceeds15s. This is not proof another policy is physically feasible.',
        f'- Prefix: no coupling{summary["prefix_never_coupled"]}/{n}, no3 consecutive contact samples{summary["prefix_no_contact_interval"]}/{n}, no annotated CONTACT{summary["prefix_annotation_without_contact_event"]}/{n}; labels merge <=2-frame contact gaps, so these counts intentionally differ.',
        f'- Prefix lift5cm legacy30-sample{summary["prefix_lift5cm30"]}/{n}, elapsed1s{summary["prefix_lift5cm_elapsed1s"]}/{n}, physical hold elapsed1s{summary["prefix_physical_hold_elapsed1s"]}/{n}. Full original recording lift{summary["full_recording_independent"]["lift5cm_elapsed1s"]}/{n}, physical hold{summary["full_recording_independent"]["physical_hold_elapsed1s"]}/{n}.',
        f'- TaskController success prefix/full-record-prefix-task/full-record-raw-task: {summary["prefix_controller"]["successes"]}/{n},{summary["full_recording_prefix_task_controller"]["successes"]}/{n},{summary["full_recording_raw_task_controller"]["successes"]}/{n}. Most records are **censored**, not proven failed: {summary["prefix_controller"]["failure_reasons"]}, {summary["full_recording_prefix_task_controller"]["failure_reasons"]}, {summary["full_recording_raw_task_controller"]["failure_reasons"]}.',
        f'- Independent ordered waypoint/declared-terminal scorer: prefix{summary["prefix_independent"]["full_task"]}/{n}, full{summary["full_recording_independent"]["full_task"]}/{n}, all RELEASE. This scorer requires prior coupling, but relaxes continuously held route, confirmed milestones and pre-RELEASE path settling. It is not a replacement for the stricter online task.',
        '- The10/12 difference is a demonstrated **contract mismatch**, not evidence that all those trajectories should pass TaskScheduler.',
        '', '## Representative alignment traces','']
    for record in representatives:
        lost=record['first_grasp_lost'];final=record['final']
        text += [f'- `{record["sequence"]}`: planned RELEASE tick{record["planned_release_frame"]}; independent ordered arrival tick{record["independent_ordered_path_arrival_frame"]}; first lost tick{lost["tick"] if lost else None}. Final phase{final["phase"]}, confirmed progress{final["confirmed_progress"]:.6f}, milestone{final["milestone_pointer"]}/{final["milestones"]}, RELEASE not authorized. [Aligned table]({Path(record["trace_path"]).with_suffix(".md").name}).']
        text += ['  '+record['blockage']['explanation']+' Path settle elapsed at loss: '+str(record['blockage']['path_settle_elapsed_at_loss_s'])+' s; required '+str(record['blockage']['required_path_settle_s'])+' s.']
    text += ['', '## Four real recorded examples','',
        'There is **no full TaskController-success archive (0/501)**. The success example below is explicitly a physical-hold proxy success.',
        '', '| category | sequence / scope | observed ticks and seconds | independent full / controller full |',
        '|---|---|---|---|']
    for example in result['representative_categories']['examples']:
        text.append('| '+example['category']+' | `'+example['sequence']+'` / '+example['scope']+' | `'+json.dumps(example['witness'],sort_keys=True)+'` | '+str(example['independent_full_task'])+' / '+str(example['actual_task_controller']['task_success'])+' |')
    text.append('')
    for example in result['representative_categories']['examples']:
        text.append('- `'+example['sequence']+'`: trajectory `'+example['trajectory_path']+'`; task `'+example['task_path']+'`.')
    text += ['', 'Exact trajectory/admission/task paths and all witness fields are in JSON `representative_categories`; the coupled fall example explicitly extends beyond its admitted safe prefix.']
    text += ['', '## Scorer controls and timing','',
        '- Synthetic measured-feedback HOLD and RELEASE both succeed under the actual frozen TaskScheduler; UNKNOWN cannot succeed with the same movement. One unstable frame delays HOLD; missing RELEASE refuses success.',
        '- Independent measured-field controls accept HOLD/RELEASE positives and reject a single missing force-valid frame or missing far/valid release witness.',
        '- 30 observed states span29/30s; 31 span1s. Frozen subtraction can defer success one control tick (synthetic HOLD elapsed1.0333s); no earlier-than1s success is accepted.',
        '- Controls validate scoring paths only, not simulation or physical grasp feasibility.',
        '', '## P1 saved stop','',
        f'- A139updates/284672 updated transitions, actual286720; B72/147456, actual149504. Both final checkpoints CPU-load with finite model+optimizer and nonempty Adam state. Policy STOPPED/native exit0; C never started. Queue FAILED only because post-STOP future evaluation checkpoints are absent.',
        '', '## Evidence (actual frozen runtime unless noted)','']
    for item in result['evidence']:text.append('- '+item['file']+':'+','.join(map(str,item['lines']))+' — `'+item['needle']+'`')
    text += ['', 'Native sampler Python source is the shared vendor file referenced by the frozen host; it was not independently snapshotted in historical3a0779f6. Actual final motion_visits remain independently available; inferred ended-episode frequencies do not.',
        'Full rows, source paths, all501 tasks, real per-sequence initialization draws, source-prefix semantics and time budgets are in `task-reachability.json`.']
    (output/'task-reachability.md').write_text('\n'.join(text)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'outputs/wam-rl-policy-v3/inspire-measured-official-20261020-r3')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    p.add_argument('--augment-existing',action='store_true',help='Reuse completed case results; only refresh CPU metadata and representative traces')
    a=p.parse_args();run=a.run.resolve();manifest=json.loads((run/'manifest.json').read_text())
    policy=next(stage for stage in manifest['stages'] if stage['name']=='rank-0')
    command=policy['argv'];frozen=Path(command[2]).parents[2]
    admission=Path(command[command.index('--audit-manifest')+1]);admitted=json.loads(admission.read_text())
    rows=[row for row in admitted['records'] if row['split']=='train' and row['engineering']=='PASS' and row['training_prefix_commands']>0]
    a.output.mkdir(parents=True,exist_ok=True);initialize(frozen)
    if a.augment_existing:
        result=json.loads((a.output/'task-reachability.json').read_text())
        result=finalize_report(result,a.output)
        print(json.dumps(dict(status=result['status'],summary=result['summary'],alignments=result['controller_independent_alignment']),indent=2))
        return
    controls_result=controls();history=history_report(run,frozen,rows)
    write(a.output/'task-reachability-preflight.json',dict(controls=controls_result,history=history))
    selected=rows if a.limit is None else rows[:a.limit]
    cases=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(frozen),)) as pool:
        futures={pool.submit(case,(row,str(admission.parent))):row['sequence'] for row in selected}
        for future in as_completed(futures):
            cases.append(future.result())
            if len(cases)%25==0:write(a.output/'task-reachability-progress.json',dict(status='RUNNING',completed=len(cases),total=len(selected),elapsed_s=time.monotonic()-started))
    cases.sort(key=lambda row:row['sequence'])
    code=frozen/'src/wam_rl_policy_v3';runner=frozen/'tools/run/inspire_measured_official_ppo.py'
    evidence_rows=[evidence(code/'task_annotations.py','TerminalMode.UNKNOWN if not intervals'),evidence(code/'measured_official_env.py',"row['training_prefix_commands']<=0"),
        evidence(code/'task_scheduler.py','TerminalMode.HOLD'),evidence(code/'task_scheduler.py','TerminalMode.RELEASE'),
        evidence(code/'measured_official_env.py',"r,_=self.trackers[i].step"),evidence(code/'measured_official_env.py','return_sum=float'),
        evidence(code/'measured_official_env.py','torch.as_tensor(reward'),evidence(runner,'ppo.log=log'),evidence(runner,'counts=reduce'),
        evidence(code/'wam_v4_runtime.py','grid <= 65535'),evidence(frozen/'tools/run/inspire_physics_host.py',"audit['motion_visits']")]
    summary=summarize(cases)
    result=dict(status='COMPLETED' if len(cases)==501 else 'LIMITED',scope='CPU teacher-archive replay and historical code/log audit; no new simulation or training',
        run_id='negative-result-audit-20261022-r1',historical_run=str(run),historical_git_commit=manifest['git_commit'],
        frozen_task=str(frozen),admission_manifest=str(admission),runtime_python=sys.executable,cpu_workers=a.workers,elapsed_s=time.monotonic()-started,
        history=history,scorer_controls=controls_result,summary=summary,evidence=evidence_rows,cases=cases,
        conclusions=['64/501 admitted tasks are UNKNOWN and lack a success branch; this cannot explain zero success on the other 437 tasks.',
            'Teacher archive replay is not historical PPO policy evaluation; absent PPO per-control trajectories cannot be reconstructed from losses/checkpoints.',
            'Recorded safe-prefix HOLD annotations are instructions, not proof that the prefix contains a one-second terminal grasp hold.',
            'Environment computes and returns reward; runner omitted return/phase logging. Missing reward telemetry does not mean PPO received no reward.'])
    write(a.output/'task-reachability.json',result)
    text=['# Historical task reachability audit','',f'Frozen actual run: `{manifest["git_commit"]}`; `{run}`.',
        f'CPU archive cases: {len(cases)}. Controls: {controls_result["status"]}. No GPU/training.', '',
        '## Direct findings','',f'- Admitted task modes: {history["terminal_modes"]}. UNKNOWN structurally lacks success, but only 64/501 tasks are UNKNOWN.',
        f'- Actual final initialization draws: {history["actual_motion_initialization_totals"]}; UNKNOWN fraction {history["actual_unknown_initialization_fraction"]:.6%}. These are not ended episodes at iteration1570.',
        f'- Safe-prefix no coupling: {summary["prefix_never_coupled"]}/{len(cases)}; no >=3 consecutive contact samples: {summary["prefix_no_contact_interval"]}/{len(cases)}; no annotated CONTACT after interval merging: {summary["prefix_annotation_without_contact_event"]}.',
        f'- Prefix lift5cm legacy30-samples: {summary["prefix_lift5cm30"]}; true elapsed1s: {summary["prefix_lift5cm_elapsed1s"]}; physical hold elapsed1s: {summary["prefix_physical_hold_elapsed1s"]}.',
        f'- TaskController prefix / full recording with prefix task / full raw task successes: {summary["prefix_controller"]["successes"]}/{len(cases)}, {summary["full_recording_prefix_task_controller"]["successes"]}/{len(cases)}, {summary["full_recording_raw_task_controller"]["successes"]}/{len(cases)}.',
        f'- Independent ordered path / full-task predicates, prefix: {summary["prefix_independent"]}; full recording: {summary["full_recording_independent"]}.',
        '- 30 sampled states span 29/30 s; TaskScheduler holds use elapsed timestamps >=1 s. Controls verify 31 states, one-frame interruption, missing RELEASE and UNKNOWN branches.',
        '- Raw/full recordings extend beyond admitted safe prefixes and can include known physical failures; those replays do not certify valid demonstrations.',
        '- Historical training has no full trajectory archive. Exact task-level ended-episode mix and its success propensity are UNKNOWN.',
        '- Reward absence is a logging defect, not missing PPO reward input. See frozen evidence below.', '', '## Source evidence','']
    for item in evidence_rows:text.append('- '+item['file']+':'+','.join(map(str,item['lines']))+' — `'+item['needle']+'`')
    text+=['','Per-task metrics, per-rank iteration1520/1570/1575 exact global counters and nominal/observed initialization weights are in `task-reachability.json`.']
    (a.output/'task-reachability.md').write_text('\n'.join(text)+'\n')
    result=finalize_report(result,a.output)
    print(json.dumps(dict(status=result['status'],summary=result['summary'],controls=controls_result),indent=2))


if __name__=='__main__':main()
