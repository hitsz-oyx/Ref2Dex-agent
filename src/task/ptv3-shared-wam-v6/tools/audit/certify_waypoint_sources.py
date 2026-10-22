#!/usr/bin/env python3
"""Authenticate r7 source roles against existing provenance, never invent intent.

Current endpoint derivation and candidate waypoint admissibility are separate
columns. Filesystem age, TaskSpec defaults, names and another execution's plan
are never certificates. No pickle, GPU, model, replay, collection or training.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_event_raw_signals import header

LEGAL='CERTIFIED_LEGAL'
LEAK='PROVEN_SELF_FUTURE'
UNKNOWN='UNCERTAIN_MISSING_RECORDS'
INTEREST={'source_role','available_before_reset','task_spec','scheduler_reference_poses',
    'reference_poses','planned_object_poses','source_artifact','source_artifact_sha256',
    'source_execution_id','target_execution_id','target_rollout_id','reset_id',
    'reset_time','reset_time_ns','reset_timestamp','reset_wall_time','reset_event_order',
    'source_frozen_event_order','independent_of_target_rollout','goal_source',
    'waypoint_derivation','created_at','source_created_at'}
SCALARS=INTEREST|{'motion_intent','source_sequence','framerate'}
REFS={'source_manifest','source_manifest_snapshot','parent_source_manifest','source_path',
      'native_source_path','raw_path','original_plan'}


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def hits(value,prefix='$'):
    """Raw metadata only: never instantiate TaskSpec's default source_role."""
    out=[]
    if isinstance(value,dict):
        for key,item in value.items():
            location=prefix+'.'+key
            if key in INTEREST:
                summary=item if isinstance(item,(str,int,float,bool,type(None))) else '<container>'
                out.append(dict(pointer=location,field=key,value=summary))
            if isinstance(item,dict):out.extend(hits(item,location))
            elif isinstance(item,list):
                for i,row in enumerate(item):
                    if isinstance(row,dict):out.extend(hits(row,location+f'[{i}]'))
    return out


def classify(proof):
    """Classify a normalized, content-verified binding, not a guessed candidate.

    Before-reset proof may use producer-recorded order in one execution log.
    The extractor does not synthesize any of these facts from file mtimes.
    """
    bound=proof.get('target_tuple_bound') is True
    source_id=proof.get('source_execution_id')
    target_id=proof.get('target_execution_id')
    identities_present=isinstance(source_id,str) and bool(source_id) and isinstance(target_id,str) and bool(target_id)
    if (bound and proof.get('same_target_execution') is True
            and identities_present and source_id==target_id
            and proof.get('source_reads_target_future') is True
            and proof.get('derivation_verified') is True):
        return LEAK
    if (bound and proof.get('source_role')=='explicit_task_instruction'
            and identities_present and source_id!=target_id
            and proof.get('available_before_reset') is True
            and proof.get('independent_of_target_rollout') is True
            and proof.get('artifact_digest_verified') is True
            and proof.get('reset_binding_verified') is True
            and proof.get('derivation_verified') is True
            and proof.get('goal_semantics')=='spatial_path_waypoint'):
        return LEGAL
    return UNKNOWN


class Inventory:
    def __init__(self):self.records={};self.documents={}

    def inspect(self,path):
        path=Path(path)
        name=str(path)
        if name in self.records:return self.records[name]
        result=dict(id=len(self.records),path=name,exists=path.exists(),metadata_hits=[],
                    fields=[],declared_identities={},read_error=None)
        self.records[name]=result
        if not result['exists']:return result
        stat=path.stat()
        result.update(resolved_path=str(path.resolve()),mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns,birthtime=getattr(stat,'st_birthtime',None),size_bytes=stat.st_size,
            timestamp_scope='filesystem metadata only; does not certify target reset order')
        try:
            if path.is_dir():
                result['fields']=[p.stem for p in sorted(path.glob('*.npy'))]
                result['metadata_hits'].extend(dict(field=key,value='<array field; inspect>',pointer=key+'.npy')
                    for key in result['fields'] if key in INTEREST)
                result['metadata_ids']=[self.inspect(p)['id'] for p in sorted(path.glob('*.json'))]
            elif path.suffix=='.json':
                value=read(path);self.documents[name]=value
                result['content_sha256']=digest(path)
                result['metadata_hits']=hits(value)
                if isinstance(value,dict):
                    result['declared_identities']={k:v for k,v in value.items()
                        if ('sha256' in k or k.endswith('_identity')) and isinstance(v,str)}
                    for field in ('object_poses.npy','poses.npy'):
                        if isinstance(value.get('array_sha256'),dict) and field in value['array_sha256']:
                            result['declared_identities'][field]=value['array_sha256'][field]
            elif path.suffix=='.npz':
                with zipfile.ZipFile(path) as archive:
                    for info in archive.infolist():
                        if not info.filename.endswith('.npy'):continue
                        key=info.filename[:-4];result['fields'].append(key)
                        if key not in SCALARS:continue
                        with archive.open(info) as stream:spec=header(stream)
                        if 'O' in spec['dtype']:
                            result['metadata_hits'].append(dict(field=key,value='<opaque object; not decoded>',pointer=key))
                        elif info.file_size<=4096:
                            value=np.load(io.BytesIO(archive.read(info)),allow_pickle=False)
                            if value.size==1:
                                scalar=value.item()
                                if isinstance(scalar,bytes):scalar=scalar.decode('utf-8')
                                result['metadata_hits'].append(dict(field=key,value=scalar,pointer=key))
                            else:result['metadata_hits'].append(dict(field=key,value='<non-scalar>',pointer=key))
                        else:result['metadata_hits'].append(dict(field=key,value='<array field; inspect>',pointer=key))
            elif path.suffix=='.npy':
                with path.open('rb') as stream:result['npy_header']=header(stream)
                result['opaque']='O' in result['npy_header']['dtype']
                # Only recorded object trajectories are hashed, never huge hand
                # payloads or opaque nested annotations.
                if path.name in ('object_poses.npy','poses.npy'):
                    result['content_sha256']=digest(path)
            elif path.suffix=='.py':result['content_sha256']=digest(path)
        except Exception as error:result['read_error']=type(error).__name__+': '+str(error)
        return result


def explicit_candidates(context):
    """Do not silently overlook a role/reset declaration in any linked layer."""
    return [hit for hit in context if hit['field'] in
            {'source_role','available_before_reset','scheduler_reference_poses',
             'reference_poses','planned_object_poses','task_spec','waypoint_derivation'}]


def run(curated_path,output):
    start=time.monotonic()
    if output.exists():raise FileExistsError('fresh evidence directory required')
    if output.resolve().parent!=(TASK/'docs/audit').resolve():
        raise ValueError('only new task-local audit artifacts permitted')
    free=shutil.disk_usage('/home2').free
    estimate=64*2**20
    if free-estimate<20*2**30:raise ValueError('insufficient space for bounded audit')
    curated=read(curated_path);root=Path(curated['data_root'])
    parent_path=root/'recipe/manifest.json';parent=read(parent_path)
    plan_path=root/'plan.json';plan=read(plan_path);jobs={r['key']:r for r in plan['jobs']}
    previous=read(TASK/'docs/audit/brief32-raw-signal-inventory-20261023.json')
    previous_rows={r['sequence_id']:r for r in previous['sequence_inventory']}
    rows=curated['selection']['train_window_rows']
    tuples=[[int(r['source_id']),[int(r['sequence_id']),int(r['slot']),int(r['tick'])],int(r['tick'])] for r in rows]
    if len(tuples)!=6000 or len({json.dumps(t) for t in tuples})!=6000:
        raise ValueError('r7 requires exactly 6000 ordered distinct tuples')
    selected=defaultdict(list)
    for ordinal,row in enumerate(rows):selected[int(row['sequence_id'])].append((ordinal,row))
    inventory=Inventory();mothers=[];mother_by_id={}
    for number,(sid,windows) in enumerate(sorted(selected.items()),1):
        if time.monotonic()-start>600:raise TimeoutError('600s audit bound; no partial certification')
        seq=parent['sequences'][sid];job=jobs[seq['key']]
        if int(seq['source_id'])!=int(windows[0][1]['source_id']):raise ValueError('source identity mismatch')
        layer_paths=list(previous_rows[sid]['layer_paths'])
        # Freshly trace producer-linked manifests and annotation dependencies;
        # prior inventory only supplies the known paths, not a verdict.
        source_row=job.get('source_row',{})
        for key in REFS:
            value=source_row.get(key) or job.get(key)
            if isinstance(value,str) and value.startswith('/') and Path(value).suffix in ('.json','.npy','.npz'):
                layer_paths.append(value)
        if int(seq['source_id'])==4:
            native_run=Path(source_row['parent_source_manifest']).parent
            for filename in ('launch.json','gpu-started.json','result.json','upstream-runtime-patch.json'):
                candidate=native_run/filename
                if candidate.exists():layer_paths.append(str(candidate))
            patch=native_run/'upstream-runtime-patch.json'
            if patch.exists():
                details=read(patch)
                layer_paths.extend(details[k] for k in ('source','copy') if isinstance(details.get(k),str))
            launch=native_run/'launch.json'
            if launch.exists():
                details=read(launch)
                if details.get('object_manifest'):layer_paths.append(details['object_manifest'])
        pose_path=Path(seq['path'])/'object_poses.npy'
        layer_paths.append(str(pose_path))
        if 'sequence_folder' in job:layer_paths.append(str(Path(job['sequence_folder'])/'poses.npy'))
        artifact_ids=[];context=[]
        def add(path):
            item=inventory.inspect(path);artifact_ids.append(item['id'])
            context.extend(dict(artifact_id=item['id'],**hit) for hit in item['metadata_hits'])
            for child in item.get('metadata_ids',[]):
                record=next(v for v in inventory.records.values() if v['id']==child)
                artifact_ids.append(child)
                context.extend(dict(artifact_id=child,**hit) for hit in record['metadata_hits'])
                value=inventory.documents[record['path']]
                if isinstance(value,dict):
                    for key in ('source_path','native_source_path'):
                        target=value.get(key)
                        if isinstance(target,str) and target.startswith('/'):
                            linked=inventory.inspect(target);artifact_ids.append(linked['id'])
                            context.extend(dict(artifact_id=linked['id'],**hit) for hit in linked['metadata_hits'])
                    for dependency in value.get('dependencies',{}):
                        if str(dependency).endswith(('.npy','.json','.npz')):
                            linked=inventory.inspect(dependency);artifact_ids.append(linked['id'])
                            context.extend(dict(artifact_id=linked['id'],**hit) for hit in linked['metadata_hits'])
        for path in dict.fromkeys(layer_paths):add(path)
        inline=hits(job,'$.plan.jobs['+seq['key']+']')+hits(seq,'$.parent.sequences['+str(sid)+']')
        context.extend(dict(artifact_id='plan-or-parent',**hit) for hit in inline)
        candidates=explicit_candidates(context)
        if candidates:
            # Native formats need individual evidence normalization rather than
            # a guessed schema or blanket UNKNOWN. Preserve discovery to inspect.
            raise ValueError('explicit role/intent candidate requires inspection before census: '+str(candidates[:3]))
        pose=inventory.records[str(pose_path)]
        if not pose.get('content_sha256') or pose.get('read_error'):
            raise ValueError('recorded target trajectory cannot be verified: '+str(pose_path))
        frame_count=pose['npy_header']['shape'][0]
        if any(not 3<=int(row['tick'])<frame_count-24 for _,row in windows):
            raise ValueError('curated tuple outside its recorded trajectory')
        record=dict(sequence_id=sid,key=seq['key'],source_id=int(seq['source_id']),windows=len(windows),
            current_label_artifact=str(pose_path),current_label_sha256=pose.get('content_sha256'),
            original_source_sequence=job['source_sequence'],original_source_episode_id=source_row.get('source_episode_id'),
            processed_pose_artifact=str(Path(job['sequence_folder'])/'poses.npy') if 'sequence_folder' in job else job.get('payload'),
            upstream_artifact_ids=sorted(set(artifact_ids)),metadata_hits=context,
            source_role=None,available_before_reset=None,target_execution_reset=None,
            independent_reference_artifact=None,proof={},status=classify({}),
            missing=['waypoint_derivation_source_binding','explicit_instruction_source_role',
                'target_execution_and_reset_binding','content_frozen_before_target_reset',
                'independence_from_target_rollout'],
            distinction='Recorded trajectory lineage is known; its role as a pre-reset plan for THIS target execution is unproven.')
        if int(seq['source_id'])==4:
            with np.load(job['payload'],allow_pickle=False) as archive:
                goal=archive['goal']
                record['native_goal_position_evidence']=dict(shape=list(goal.shape),
                    first=goal[0].tolist(),constant=bool(np.all(goal==goal[0])),
                    scope='native position target; not a certified scheduler waypoint/reference binding')
        mothers.append(record);mother_by_id[sid]=record
        if number%250==0:print(json.dumps(dict(scanned_mothers=number,artifacts=len(inventory.records))),flush=True)
    # Cross-execution declarations exist, but their label paths cannot be
    # borrowed for human windows. Search exact paths, never object/name aliases.
    external=[]
    candidates=[ROOT/'outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1/manifest.json']
    candidates.extend(sorted((ROOT/'outputs/wam-rl-policy-v3').glob('source-ceiling-20261023-r*/registration.json')))
    targets={str(Path(m['current_label_artifact']).parent.resolve()) for m in mothers}
    for path in candidates:
        if not path.exists():continue
        document=read(path);matches=[]
        for record in document.get('records',[]):
            value=record.get('source')
            if isinstance(value,str) and str(Path(value).resolve()) in targets:matches.append(value)
        external.append(dict(path=str(path),sha256=digest(path),mtime_ns=path.stat().st_mtime_ns,
            declared_goal_source=document.get('goal_source'),
            exact_r7_label_path_matches=matches,records=len(document.get('records',[])),
            registration_jobs=len(document.get('jobs',[])),
            status='OTHER_EXECUTION_NOT_A_TARGET_WINDOW_BINDING'))
    count=Counter(m['status'] for m in mothers for _ in range(m['windows']))
    by_source={str(s):dict(windows=sum(m['windows'] for m in mothers if m['source_id']==s),
        mothers=sum(m['source_id']==s for m in mothers),
        classifications=dict(Counter(m['status'] for m in mothers if m['source_id']==s
                                      for _ in range(m['windows'])))) for s in sorted({m['source_id'] for m in mothers})}
    output.mkdir()
    def write(name,value):
        data=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()+b'\n'
        if len(data)+sum(p.stat().st_size for p in output.iterdir())>estimate:
            raise ValueError('audit write estimate exceeded before write')
        with (output/name).open('xb') as stream:stream.write(data)
    write('artifact-inventory.json',list(inventory.records.values()))
    write('mother-provenance.json',mothers)
    mother_indices={m['sequence_id']:i for i,m in enumerate(mothers)}
    with (output/'window-certification.csv').open('x',newline='') as stream:
        fields=['ordinal','source_id','sequence_id','slot','tick','window_start_frame','sequence_key','phase',
            'current_label_artifact','current_label_sha256','current_legacy_endpoint_frame',
            'current_endpoint_reads_target_future','processed_pose_artifact','source_role',
            'available_before_reset','target_reset_binding','waypoint_source_artifact',
            'waypoint_status','missing','mother_provenance_index']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for ordinal,row in enumerate(rows):
            mother=mother_by_id[int(row['sequence_id'])]
            writer.writerow(dict(ordinal=ordinal,source_id=row['source_id'],sequence_id=row['sequence_id'],
                slot=row['slot'],tick=row['tick'],window_start_frame=row['tick'],sequence_key=row['sequence_key'],
                phase=row['phase'],current_label_artifact=mother['current_label_artifact'],
                current_label_sha256=mother['current_label_sha256'],current_legacy_endpoint_frame=int(row['tick'])+24,
                current_endpoint_reads_target_future=True,processed_pose_artifact=mother['processed_pose_artifact'],
                source_role='UNKNOWN',available_before_reset='UNPROVEN',target_reset_binding='MISSING',
                waypoint_source_artifact='UNBOUND',waypoint_status=mother['status'],
                missing=';'.join(mother['missing']),mother_provenance_index=mother_indices[mother['sequence_id']]))
    write('legal-only-tuples.json',[])
    # Separate current endpoint fact from the candidate waypoint's uncertainty.
    receipt=dict(schema='ref2dex.brief35.source-role-certification.v1',
        experiment_id='P-20261023-source-role-certification',run_id=output.name,
        frozen_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        measured_at=datetime.now(timezone.utc).isoformat(),
        identities=dict(curated_manifest_sha256=digest(curated_path),
            ordered_window_tuples_sha256=hashlib.sha256((json.dumps(tuples,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest(),
            parent_manifest_sha256=digest(parent_path),preparation_plan_sha256=digest(plan_path)),
        counts=dict(total_windows=len(rows),mothers=len(mothers),object_streams=len({(r['sequence_id'],r['slot']) for r in rows}),
            certified_legal=count[LEGAL],proven_self_future=count[LEAK],uncertain=count[UNKNOWN],
            legacy_endpoint_future_derived=len(rows)),by_source=by_source,
        missing_window_counts={key:sum(m['windows'] for m in mothers if key in m['missing']) for key in mothers[0]['missing']},
        native_goal_position_candidate=dict(mothers=sum('native_goal_position_evidence' in m for m in mothers),
            windows=sum(m['windows'] for m in mothers if 'native_goal_position_evidence' in m)),
        external_declarations=external,
        artifact_counts=dict(total=len(inventory.records),missing=sum(not r['exists'] for r in inventory.records.values()),
            read_errors=sum(r['read_error'] is not None for r in inventory.records.values()),
            explicit_source_role_hits=sum(h['field']=='source_role' for r in inventory.records.values() for h in r['metadata_hits']),
            explicit_before_reset_hits=sum(h['field']=='available_before_reset' for r in inventory.records.values() for h in r['metadata_hits'])),
        cache=dict(status='NOT_CONSTRUCTED_NO_CERTIFIED_WAYPOINTS',windows=0,
            note='brief34 reader is an index-only fail-closed seam, not an implemented external cache loader; no new production schema is invented'),
        filesystem_free_before_bytes=free,write_estimate_bytes=estimate,write_hard_limit_bytes=2**30,
        elapsed_s=time.monotonic()-start,environment=dict(python=sys.version,numpy=np.__version__,gpu_used=False),
        training_updates=0,production_code_modified=False,config_status_modified=False,
        limitations=['Bound lineage and explicitly linked upstream metadata only; no claim that an unrecorded independent plan never existed.',
            'NPY object payloads not unpickled; unknown nested semantics remain unproven.',
            'Filesystem mtime/ctime are not creation/reset certificates.'])
    write('receipt.json',receipt)
    actual=sum(p.stat().st_size for p in output.iterdir())
    if actual>estimate:raise ValueError('bounded artifact estimate exceeded')
    print(json.dumps(dict(counts=receipt['counts'],artifact_bytes=actual,elapsed_s=receipt['elapsed_s'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curated',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.curated,args.output)
