#!/usr/bin/env python3
"""Full reachable recording inventory and fail-closed event replay preflight.

Only new audit artifacts are written. No training, physics, source mutation,
GT-as-intent substitution, finite-difference velocity, or force imputation.
"""
import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import zipfile

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def header(stream):
    magic=stream.read(8)
    if len(magic)!=8 or magic[:6]!=b'\x93NUMPY':raise ValueError('NPY header required')
    width=2 if magic[6]==1 else 4
    size=struct.unpack('<H' if width==2 else '<I',stream.read(width))[0]
    if size>65536:raise ValueError('unbounded NPY header')
    result=ast.literal_eval(stream.read(size).decode('latin1'))
    return dict(shape=list(result['shape']),dtype=str(result['descr']))


def inventory_path(path):
    path=Path(path)
    if not path.exists():return dict(path=str(path),status='MISSING',fields={},metadata={})
    fields={};metadata={}
    if path.is_dir():
        for p in sorted(path.glob('*.npy')):
            with p.open('rb') as stream:fields[p.stem]=header(stream)
        for p in sorted(path.glob('*.json')):
            # Metadata only, no recursive crawl into unrelated tasks/data.
            metadata[p.name]=read(p)
    elif path.suffix=='.npz':
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith('.npy'):
                    with archive.open(name) as stream:fields[name[:-4]]=header(stream)
    elif path.suffix=='.npy':
        with path.open('rb') as stream:fields[path.stem]=header(stream)
    else:
        return dict(path=str(path),status='UNSUPPORTED',fields={},metadata={})
    return dict(path=str(path),status='READABLE',fields=fields,metadata=metadata)


def signal_flags(layers):
    fields={k:v for layer in layers for k,v in layer['fields'].items()}
    def present(*names):return any(n in fields for n in names)
    velocity=present('object_velocity','object_velocities','object_velocity_world')
    velocity|=any(name in fields and fields[name]['shape'] and fields[name]['shape'][-1]==13
                  for name in ('object','objects'))
    # A time/frame grid is distinct from the TaskSpec's instruction clock.
    return dict(observed_pose=present('object_poses','poses','object','objects'),
        observed_hand=present('joints','hand'),
        native_object_velocity=bool(velocity),
        observed_clock=present('time','timestamps','frame_ids','observed_frame'),
        source_cursor=present('source_cursor_frame','plan_cursor_frame'),
        robot_contact_forces=present('contact_forces'),
        robot_force_valid=present('contact_force_valid','force_valid'),
        measured_support=present('supported','support'),
        support_object_force=present('object_contact_force'),
        support_table_pose=present('table_pose'),
        measured_workspace=present('workspace_ok','physical_abort'),
        native_q_dq=present('q') and present('dq'),
        recorded_goal_position=present('goal'),
        task_event_kind=present('planned_event_kind','task_event_kind'),
        task_event_source_frame=present('planned_event_source_frame','task_event_source_frame'),
        task_source_fps=present('task_source_fps'),
        task_object_path=present('task_object_path','scheduler_reference_poses'),
        task_source_frames=present('task_source_frames'),
        task_terminal_mode=present('terminal_mode'),
        task_source_role=present('source_role'),
        observer_snapshot=present('feedback_observer_state','feedback_observer_snapshot'),
        scheduler_snapshot=present('task_scheduler_state','task_scheduler_snapshot'))


def strict_missing(flags,task_packet=None):
    missing=[]
    for key in ('observed_pose','observed_hand','observed_clock','native_object_velocity',
                'robot_contact_forces','robot_force_valid','measured_support','measured_workspace'):
        if not flags.get(key):missing.append(key)
    if task_packet is None:missing.append('bound_predeployment_TaskSpec')
    # Exact deployment hand choice, geometry, validity, and episode reset must
    # additionally be bound by the producer, never selected for a good result.
    return missing


def replay_complete_record(record,tick):
    """Same deployed state machines, ONLY for an explicit complete audit record.

    Observations are indexed at the current frame, never converted wholesale.
    TaskSpec is a separate known instruction supplied before the episode.
    """
    missing=strict_missing(record.get('signals',{}),record.get('task_spec'))
    if missing:raise ValueError('missing exact deployment inputs: '+','.join(missing))
    if record.get('available_before_reset') is not True or record.get('reset_frame')!=0:
        raise ValueError('explicit predeployment instruction/reset provenance required')
    sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))
    from wam_rl_policy_v3.task_spec import TaskSpec
    from wam_rl_policy_v3.task_scheduler import TaskScheduler
    from wam_rl_policy_v3.task_feedback import FeedbackObserver
    spec=TaskSpec.from_packet(record['task_spec'])
    scheduler=TaskScheduler(spec,record['query'],interaction_points=record['surface'])
    observer=FeedbackObserver(record['surface'])
    output=[]
    for frame in range(tick+1):
        current=record['observations'][frame]
        if not current['observation_valid']:raise ValueError('invalid measured prefix; cannot bridge')
        feedback=observer.update(current['pose'],current['hand'],current['velocity'],current['time'],
            contact_forces=current['contact_forces'],force_valid=current['force_valid'],
            supported=current['supported'],workspace_ok=current['workspace_ok'])
        decision=scheduler.update(current['pose'],current['time'],feedback,hand_points=current['hand'])
        event=decision.next_event.wam_condition()
        output.append(dict(tick=frame,**event,known_NONE=bool(event['event_valid'] and event['event_type']==0)))
        if decision.terminated and frame<tick:raise ValueError('terminal scheduler before requested tick; no silent reset')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--curated',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    allowed=TASK/'docs/audit'
    if a.output.resolve().parent!=allowed.resolve():raise ValueError('brief32 artifacts must be new allowed audit files')
    started=time.monotonic();curated=read(a.curated);data_root=Path(curated['data_root'])
    parent=read(data_root/'recipe/manifest.json');plan=read(data_root/'plan.json')
    jobs={j['key']:j for j in plan['jobs']}
    inv=read(data_root.parent/'inventory.json');inventory={r['key']:r for r in inv['rows']}
    selected=defaultdict(list)
    for row in curated['selection']['train_window_rows']:
        if row.get('admitted') is True:selected[int(row['sequence_id'])].append(row)
    rows=[];coverage=Counter();source_counts=Counter();path_cache={}
    def inspect(path):
        path=str(path)
        if path not in path_cache:path_cache[path]=inventory_path(path)
        return path_cache[path]
    for seq_id,windows in sorted(selected.items()):
        meta=parent['sequences'][seq_id];job=jobs[meta['key']];entry=inventory[meta['key']]
        source_counts[str(meta['source_id'])]+=1
        layer_paths=[meta['path']]
        layer_paths.append(job['sequence_folder'] if meta['source_id']<3 else job['source_row']['source_path'])
        if entry.get('packet'):layer_paths.append(entry['packet'])
        if job.get('raw_path'):layer_paths.append(job['raw_path'])
        layers=[inspect(path) for path in dict.fromkeys(layer_paths)]
        flags=signal_flags(layers)
        # Metadata TaskSpec must be explicit; cached GT events are not searched
        # as candidate task instructions. Any discovered packet stops for review.
        packets=[]
        for layer in layers:
            for metadata in layer['metadata'].values():
                if isinstance(metadata,dict) and ('task_spec' in metadata or metadata.get('schema')=='ref2dex.object-task.v1'):
                    packets.append(layer['path'])
        if packets:raise ValueError('inspect newly discovered TaskSpec before admitting: '+repr(packets))
        for key,value in flags.items():coverage[key]+=int(value)
        rows.append(dict(sequence_id=seq_id,key=meta['key'],source_id=int(meta['source_id']),
            windows=len(windows),last_selected_tick=max(int(w['tick']) for w in windows),
            layer_paths=[str(x) for x in dict.fromkeys(layer_paths)],signals=flags,
            missing_exact_deployment_inputs=strict_missing(flags),bound_TaskSpec=False))
    elapsed=time.monotonic()-started
    # 30 mothers: all 3 existing mothers per minority source, 6 GRAB mothers,
    # chosen by stable sequence id; this is preflight, not fake replay.
    pilot=[]
    for sid in sorted(source_counts,key=int):
        pool=[r for r in rows if r['source_id']==int(sid)]
        pilot.extend(pool[:6 if int(sid)==1 else 3])
    assert len(rows)==1064 and len(pilot)==30
    pilot_started=time.monotonic()
    pilot_eligible=[r for r in pilot if not r['missing_exact_deployment_inputs']]
    assert not pilot_eligible,'unexpected complete source must be explicitly bound/reviewed'
    pilot_windows=sum(r['windows'] for r in pilot)
    pilot_result=dict(mothers=30,windows=pilot_windows,eligible_mothers=0,eligible_windows=0,
        status='NOT_RUN_MISSING_EXACT_DEPLOYMENT_SIGNALS',state_machine_calls=0,
        legal_window_ratio=0.,type_agreement=None,delta30_median_difference=None,
        delta30_quantiles=None,valid_mismatch_rate=None,mother_cluster_ci95=None,
        preflight_elapsed_s=time.monotonic()-pilot_started,
        replay_cpu_hours_extrapolation=None,event_export_bytes_extrapolation=None,
        reason='No comparable legal outputs; do not replace missing inputs by GT/default/finite differences',
        selected=[dict(sequence_id=r['sequence_id'],key=r['key'],source_id=r['source_id'],
                      windows=r['windows'],missing=r['missing_exact_deployment_inputs']) for r in pilot])
    result=dict(schema='ref2dex.brief32.raw-event-signal-inventory.v1',
        frozen_context_head='349ff9466f65fc18ad9428e9c792eabe06ff35f5',
        implementation_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment=dict(python=sys.version,numpy=np.__version__,device='CPU header inventory/preflight'),
        denominator_mothers=len(rows),curated_windows=sum(r['windows'] for r in rows),
        counts_by_source=dict(source_counts),signal_coverage=dict(coverage),
        full_chain_eligible_mothers=sum(not r['missing_exact_deployment_inputs'] for r in rows),
        sequence_inventory=rows,recording_layers=list(path_cache.values()),
        scope='Every curated mother: overlay, bound processed/native payload, hand packet, explicitly linked original raw file. No unrelated rollout borrowing, no opaque object pickle deserialization.',
        opaque_note='NPY object dtype is reported as opaque; a nested human label is not certified robot force. Missing exported/native robot signals are not recovered from opaque labels.',
        timing=dict(full_inventory_elapsed_s=elapsed),pilot=pilot_result,
        training_updates=0,production_semantics_changed=False,config_status_changed=False,
        decision='NEW_RECORDING_OR_EXPLICIT_CONDITION_RECIPE_CHANGE; exact release-capable chain cannot be re-exported from available robot signals')
    a.output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(dict(mothers=len(rows),windows=result['curated_windows'],
        coverage=dict(coverage),layers=len(path_cache),sources=dict(source_counts),
        inventory_elapsed_s=elapsed,pilot={k:v for k,v in pilot_result.items() if k!='selected'},
        artifact_bytes=a.output.stat().st_size),indent=2))


if __name__=='__main__':main()
