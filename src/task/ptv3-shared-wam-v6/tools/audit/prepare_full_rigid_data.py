"""Full local rigid preparation, with immutable sequence receipts and a 30GiB floor.

Inventory is read-only except its own output. Execute requires the saved budget.
Published sequences are never overwritten; an interrupted run resumes at sequence
boundaries. Source/goal provenance is explicitly unknown without reset evidence.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK.parent/'ptv3-shared-wam-v5/src'), str(TASK/'tools/audit')]
from ptv3_shared_wam_v6.rigid_dataset import admission_reasons, numerical_pose_valid, geometry_class, pca_shape
from ptv3_shared_wam_v5.cache_contract import metadata_identity, array_identity
from ptv3_shared_wam_v5.contracts import LABEL_SEMANTICS
from ptv3_shared_wam_v5.event_labels import annotate_events, proximity_counts, DEFAULT_PROTOCOL, OBSERVATION_STRIDE_BY_SOURCE
from prepare_local_data import inspect_job, local_job, load_tool

GIB = 2**30
FLOOR = 30*GIB
META_BYTES = 65536
OLD = ROOT/'outputs/ptv3-shared-wam-v6/local-data-20261021-r2/data'
PLAN = ROOT/'tmp/wam-v5/plan-compact.json'
REGISTRY = ROOT/'outputs/ptv3-shared-wam-v5/object-surfaces-20261019-r2'
PILOT = ROOT/'outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json'
EXPERIMENT = 'P-20261023-full-rigid-data'
SOURCES = [0, 1, 2, 4, 5, 6, 8, 9, 10]
PHASES = ['approach', 'close', 'contact', 'hold', 'unknown']
CLASSES = ['flat', 'elongated', 'compact', 'unknown']


def read(p):
    return json.loads(Path(p).read_text())


def read_receipt(path):
    # write(..., fixed=True) puts JSON on the first line, then pads to64KiB.
    # Avoid rereading gigabytes of padding during a full sequence census.
    with Path(path).open() as stream:
        return json.loads(stream.readline())


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(p, data, fixed=False, durable=True):
    p = Path(p)
    raw = (json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    if fixed:
        if len(raw) > META_BYTES:
            raise ValueError('fixed metadata budget exceeded: '+str(p))
        raw += b' '*(META_BYTES-len(raw))
    pending = p.with_name(p.name+'.part')
    with pending.open('wb') as f:
        f.write(raw); f.flush()
        if durable:os.fsync(f.fileno())
    pending.replace(p)


def guard(root, need=0):
    free = shutil.disk_usage(root).free
    if free - need < FLOOR:
        raise InterruptedError('30GiB storage floor: free=%d need=%d floor=%d' % (free, need, FLOOR))
    return free


def array_size(shape, dtype):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, dict(descr=np.dtype(dtype).str, fortran_order=False, shape=shape))
    return stream.tell()+int(np.prod(shape))*np.dtype(dtype).itemsize


def fixed_conversion_file_sizes(n, slots):
    """Exact fixed-shape NPY + padded metadata files; no transition list file."""
    specifications = [((n,2,21,3),'f4'), ((n,2,21),'?'), ((n,2,778,3),'f2'),
        ((n,2),'?'), ((2,778),'u1'), ((n,slots,4,4),'f4'), ((n,slots),'?'),
        ((n,),'?'), ((n,2),'?'), ((n,slots,2048),'f2'), ((n,slots,2048),'u1'),
        ((n,slots),'?'), ((n,slots),'i1'), ((n,slots),'u1'), ((n,slots),'i4'),
        ((n,slots),'?'), ((n,slots),'u2'), ((n,slots),'u2'), ((n,slots),'?')]
    return [array_size(s,d) for s,d in specifications]+[META_BYTES]*3


def fixed_conversion_sizes(n, slots):
    return sum(fixed_conversion_file_sizes(n,slots))


def inventory(out):
    out.mkdir(parents=True, exist_ok=False)
    jobs = read(PLAN)['jobs']; old = {r['key']:r for r in read(OLD/'recipe/manifest.json')['sequences']}
    prior = {r['key']:r for r in read(ROOT/'outputs/ptv3-shared-wam-v6/local-data-20261021-r2/inventory.json')['rows']}
    rows = []; started = time.monotonic()
    for index, job in enumerate(jobs):
        row = dict(key=job['key'], source_id=job['source_id'], sequence_id=index,
            source_sequence=job['source_sequence'], split=next(iter(job['windows'])))
        if len(job['windows']) != 1:
            raise ValueError('mother must have a single inherited split')
        if job['key'] in old:
            r = old[job['key']]; meta = read(Path(r['path'])/'meta.json')
            # Every reused file is opened during execution, not assumed from a count.
            row.update(status='REUSE', frames=meta['frames'], objects=len(r['objects']),
                source=r['path'], conversion_bytes=0, original_payload_present=Path(local_job(job).get('payload',local_job(job).get('sequence_folder',''))).exists())
        else:
            try:
                inspected = inspect_job(job)
                row.update(inspected, status='CONVERT', sequence_id=index,
                    conversion_bytes=fixed_conversion_sizes(inspected['frames'],inspected['objects']))
            except Exception as exc:
                row.update(status='REJECT', reasons=['native_input_unavailable_or_unbound'], detail=repr(exc),
                    frames=prior[job['key']]['frames'], objects=prior[job['key']]['objects'], conversion_bytes=0)
        rows.append(row)
        if (index+1)%4000==0:
            print(json.dumps(dict(stage='inventory', checked=index+1, total=len(jobs))),flush=True)
    converted = [r for r in rows if r['status']=='CONVERT']
    exact = sum(r['conversion_bytes'] for r in converted)
    # Full dense ticks, fixed-size tuples, feature/dedup tables, provenance JSONL.
    # Upper bound, separately labelled from exact conversion bytes.
    candidates = sum(max(0,r['frames']-27)*(r['objects'] if r['source_id']<3 else 1) for r in rows if r['status']!='REJECT')
    manifest_budget = candidates*2048+len(rows)*(META_BYTES+8192)+512*1024**2
    staging = max((r['conversion_bytes'] for r in converted),default=0)*12
    block=os.statvfs(out).f_frsize
    allocated=sum(((size+block-1)//block)*block for r in converted for size in fixed_conversion_file_sizes(r['frames'],r['objects']))
    filesystem_metadata_reserve=2*GIB
    free = shutil.disk_usage(out).free
    budget = dict(schema='ref2dex.full-rigid-budget.v1', experiment_id=EXPERIMENT,
        run_id=out.name, work_version='full-rigid-preparation.v1', git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        rows=rows, total_sequences=len(rows), by_status=dict(Counter(r['status'] for r in rows)),
        conversion_fixed_allocated_file_bytes=allocated,filesystem_metadata_reserve_bytes=filesystem_metadata_reserve,
        conversion_fixed_exact_bytes=exact, conversion_fixed_exact_gib=exact/GIB,
        conversion_definition='19 fixed-shape NPY files plus three exactly65536-byte JSON metadata files for each new sequence; cached sequences reused read-only',
        candidate_window_upper_bound=candidates, manifest_and_receipt_upper_bound_bytes=manifest_budget,
        partial_attempt_reserve_bytes=staging, total_peak_upper_bound_bytes=allocated+manifest_budget+staging+filesystem_metadata_reserve,
        filesystem_free_before_bytes=free, safety_floor_bytes=FLOOR,
        free_after_peak_lower_bound_bytes=free-allocated-manifest_budget-staging-filesystem_metadata_reserve,
        fits=free-allocated-manifest_budget-staging-filesystem_metadata_reserve>=FLOOR,
        measured_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic()-started,
        plan_sha256=digest(PLAN), registry_sha256=digest(REGISTRY/'manifest.json'),
        seeds=dict(preparation=20261019, full_manifest=170, dedup=170),
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,host=platform.node()),
        training_allowed=False)
    write(out/'inventory.json',budget)
    print(json.dumps({k:v for k,v in budget.items() if k not in ['rows','environment']}),flush=True)


def load_arrays(folder,names):
    return {k:np.load(Path(folder)/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in names}


def phases(event,slot,ticks):
    state=event['state'][ticks,slot]; valid=event['event_valid'][ticks,slot]
    kind=event['event_type'][ticks,slot]; delta=event['event_delta_steps'][ticks,slot]
    p=np.full(len(ticks),4,np.uint8)
    p[valid&(state==0)]=0; p[valid&(state==0)&(kind==1)&(delta<=24)]=1
    p[valid&(state==1)]=3; p[valid&(state==1)&(kind==2)&(delta<=24)]=2
    return p


def window_all(values, ticks, history=3, future=24):
    bad=np.r_[0,np.cumsum(~values,dtype=np.int64)]
    return bad[ticks+future+1]==bad[ticks-history]


def admitted_ticks(seq,geo,contact,event,evidence,slot):
    """Vector equivalent of admission_reasons, audited against the scalar gate."""
    n=len(seq['joints']); ticks=np.arange(3,n-24,dtype=np.int32)
    if not len(ticks): return ticks,{},0
    structural=admission_reasons(seq,geo,contact,event,evidence,int(ticks[0]),slot)
    dynamic={'history_joint21_unobserved','hand21_nonfinite_valid','object_pose_invalid_h4_k24',
        'object_pose_not_finite_se3','hand_surface_nonfinite_valid','contact_confidence_invalid',
        'part_out_of_range','event_type_or_time_invalid','event_state_invalid'}
    fixed=[r for r in structural if r not in dynamic]
    if fixed: return ticks[:0],{k:len(ticks) for k in fixed},len(ticks)
    jv=seq['joint_valid']; joint_hist=np.zeros(len(ticks),bool)
    for side in range(2):
        for joint in range(21):
            joint_hist |= window_all(jv[:,side,joint],ticks,3,0)
    jfinite=(np.isfinite(seq['joints']).all(-1)|~jv).all(axis=(1,2))
    sfinite=(np.isfinite(seq['surface_offsets']).all(axis=(2,3))|~seq['surface_valid']).all(-1)
    cv=contact['valid'][:,slot]; c=contact['contact'][:,slot]; part=contact['part'][:,slot]
    cgood=(np.isfinite(c)&(c>=0)&(c<=1)).all(-1)|~cv
    pgood=((part>=0)&(part<16)).all(-1)|~cv
    ev=event['event_valid'][:,slot]
    egood=(np.isin(event['event_type'][:,slot],[0,1,2])&(event['event_delta_steps'][:,slot]>=0))|~ev
    checks=dict(history_joint21_unobserved=joint_hist,
        hand21_nonfinite_valid=window_all(jfinite,ticks),
        object_pose_invalid_h4_k24=window_all(seq['object_pose_valid'][:,slot],ticks),
        object_pose_not_finite_se3=window_all(numerical_pose_valid(seq['object_poses'][:,slot]),ticks),
        hand_surface_nonfinite_valid=window_all(sfinite,ticks),
        contact_confidence_invalid=window_all(cgood,ticks), part_out_of_range=window_all(pgood,ticks),
        event_type_or_time_invalid=window_all(egood,ticks),
        event_state_invalid=window_all(np.isin(event['state'][:,slot],[-1,0,1]),ticks))
    keep=np.logical_and.reduce(list(checks.values()))
    # Compare first/last/mid plus deterministic invalid examples to the actual
    # scalar predicate, so acceleration cannot silently weaken admission.
    sample=set(np.linspace(0,len(ticks)-1,min(8,len(ticks)),dtype=int).tolist())
    sample.update(np.flatnonzero(~keep)[:4].tolist())
    for i in sample:
        if bool(keep[i]) != (not admission_reasons(seq,geo,contact,event,evidence,int(ticks[i]),slot)):
            raise AssertionError('vector/scalar admission disagreement')
    return ticks[keep],{k:int((~v).sum()) for k,v in checks.items() if not v.all()},len(ticks)


_GEOS={}
_TOOLS=None
_PLAN_SHA=None
_GEOMETRY_ROWS=None


def init_worker(device,out):
    global _TOOLS, _PLAN_SHA, _GEOMETRY_ROWS
    _PLAN_SHA=read(Path(out)/'inventory.json')['plan_sha256']
    _GEOMETRY_ROWS={r['key']:r for r in read(REGISTRY/'manifest.json')['objects']}
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import torch
    torch.set_num_threads(1)
    base=TASK.parent/'ptv3-shared-wam-v5/tools/run'
    _TOOLS=(load_tool('full_hand',base/'prepare_hand_overlay.py'),
            load_tool('full_contact',base/'prepare_surface_contact.py'),device)


def geometry(name,sid):
    key='s%d:%s'%(sid,name) if sid<3 else name
    if key not in _GEOS:
        row=_GEOMETRY_ROWS[key]
        with np.load(REGISTRY/row['path'],allow_pickle=False) as z:
            value={k:z[k] for k in z.files}
        value['identity']=array_identity(value); _GEOS[key]=value
    return key,_GEOS[key]


def process_one(args):
    out, row, job, old, old_event = args
    out=Path(out); key=row['key']; receipt=out/'receipts'/(key+'.json')
    if receipt.exists():
        existing=read_receipt(receipt)
        if existing.get('status') in ['ADMITTED','REJECTED']:
            if existing['input_plan_sha256']!=_PLAN_SHA:raise ValueError('resume plan changed')
            if existing['status']=='ADMITTED':
                if digest(existing['window_shard'])!=existing['window_shard_sha256']:
                    raise ValueError('completed window shard changed')
                for path,field in [('path','hand_meta_identity'),('contact_path','contact_meta_identity'),('event_path','event_meta_identity')]:
                    if metadata_identity(Path(existing[path])/'meta.json')!=existing[field]:
                        raise ValueError('completed sequence metadata changed')
            return existing
    guard(out, row['conversion_bytes']+32*1024**2)
    if (out/'STOP').exists(): raise InterruptedError('owner STOP file')
    start=time.monotonic(); job=local_job(job)
    base=dict(sequence_id=row['sequence_id'],key=key,source_id=row['source_id'],source_sequence=row['source_sequence'],
        split=row['split'],input_plan_sha256=_PLAN_SHA,
        conversion_started=row['status']=='CONVERT', source_read_only=True, source_role='UNKNOWN',
        available_before_reset=None,reset_timestamp_ns=None,source_available_timestamp_ns=None,
        certification='UNCERTIFIABLE_MISSING_PRE_RESET_RECORDS',
        certification_missing=['reset_timestamp','pre_reset_task_intent_source','source_role_attestation','pre_reset_availability_attestation'],
        goal_semantics='recorded_target_future_is_label_only',training_allowed=False)
    if row['status']=='REJECT':
        result=dict(base,status='REJECTED',reasons=row['reasons'],detail=row['detail'],admitted_windows=0)
        write(receipt,result,fixed=True); return result
    try:
        geos=dict(geometry(name,row['source_id']) for name in job['objects'])
        if row['status']=='REUSE':
            folder=Path(old['path']);cf=Path(old['contact_path']); ef=Path(old_event['path'])
            hm=read(folder/'meta.json');cm=read(cf/'meta.json')
            if metadata_identity(folder/'meta.json')!=old['hand_meta_identity'] or metadata_identity(cf/'meta.json')!=old['contact_meta_identity']:
                raise ValueError('reused metadata binding changed')
            if old_event['contact_meta_identity']!=old['contact_meta_identity']:
                raise ValueError('reused event/contact binding changed')
        else:
            handtool,contacttool,device=_TOOLS
            folder=out/'hands/sequences'/key;cf=out/'contacts'/key;ef=out/'events'/key
            contract=dict(identity=read(out/'contract.json')['identity'])
            if (folder/'meta.json').exists(): hm=read(folder/'meta.json')
            else:
                if row.get('packet'):
                    with np.load(row['packet'],allow_pickle=False) as z:packet={k:z[k] for k in z.files}
                else:
                    with np.load(job['payload'],allow_pickle=False) as z:
                        packet=dict(q=z['q'],hand_root=z['hand_root'],robot=np.array('shadow-official-replay'))
                options=SimpleNamespace(output=out/'hands',mano_root=Path('/home2/wyy/oyx_ws/Ref2Dex/dataset/arctic/data/body_models/mano/models'),
                    shadow_asset=ROOT/'third_party/UniGraspTransformer/dexgrasp/hand_assets/open_ai_assets/hand/robot.xml',
                    device=device,cache_contract=contract)
                hm=handtool.prepare(job,packet,options)
            # A process can stop after the original tool's atomic publish and
            # before our fixed-size envelope. Normalize on resume as well;
            # canonical metadata identities do not depend on JSON whitespace.
            write(folder/'meta.json',hm,fixed=True,durable=False)
            if (cf/'meta.json').exists(): cm=read(cf/'meta.json')
            else:
                cm=contacttool.prepare(folder,hm,geos,out/'contacts',contract)
            write(cf/'meta.json',cm,fixed=True,durable=False)
            if not (ef/'meta.json').exists():
                c=load_arrays(cf,['contact','valid']);enter,leave=proximity_counts(c['contact'],c['valid'])
                values=annotate_events(enter,leave,c['valid'],observation_stride=OBSERVATION_STRIDE_BY_SOURCE.get(row['source_id'],1))
                values.pop('transitions'); values.update(enter_count=enter,leave_count=leave,raw_valid=np.asarray(c['valid']))
                pending=ef.with_name(ef.name+'.partial-'+str(os.getpid()));pending.mkdir()
                for name,value in values.items():np.save(pending/(name+'.npy'),value)
                write(pending/'meta.json',dict(key=key,status='COMPLETED',protocol=DEFAULT_PROTOCOL,
                    contact_meta_identity=metadata_identity(cf/'meta.json')),fixed=True,durable=False)
                pending.rename(ef)
        seq=load_arrays(folder,['joints','joint_valid','surface_offsets','surface_valid','surface_parts','object_poses','object_pose_valid'])
        contact=load_arrays(cf,['contact','part','valid']);event=load_arrays(ef,['state','event_type','event_delta_steps','event_valid'])
        tuples=[]; phase_rows=[]; classes=[];counts=Counter(); candidate=0
        slots=range(len(job['objects'])) if row['source_id']<3 else [job['source_row']['target_slot']]
        for slot in slots:
            name=job['objects'][slot];_,geo=geometry(name,row['source_id'])
            bound=cm.get('geometry_identities',{}).get(name)==geo['identity']
            evidence=dict(hm,label_semantics=cm['label_semantics'],binding_verified=cm['hand_identity']==metadata_identity(folder/'meta.json'),
                event_binding_verified=read(ef/'meta.json')['contact_meta_identity']==metadata_identity(cf/'meta.json'),geometry_binding_verified=bound)
            ticks,reasons,total=admitted_ticks(seq,geo,contact,event,evidence,slot)
            candidate+=total;counts.update(reasons)
            tuples.extend([[row['sequence_id'],slot,int(t)] for t in ticks])
            phase_rows.extend(phases(event,slot,ticks).tolist())
            classes.extend([CLASSES.index(geometry_class(pca_shape(geo['surface'])))]*len(ticks))
        # NPY windows are the authoritative ordered sequence_id/slot/tick rows;
        # provenance columns are materialized in the full JSONL manifest later.
        shard=out/'windows'/(key+'.npz')
        pending=shard.with_name(shard.name+'.part')
        with pending.open('wb') as f:
            np.savez(f,tuples=np.asarray(tuples,np.int32).reshape(-1,3),phase=np.asarray(phase_rows,np.uint8),geometry_class=np.asarray(classes,np.uint8))
        pending.replace(shard)
        conversion_files=[p for parent in [folder,cf,ef] for p in parent.iterdir() if p.is_file()]
        result=dict(base,status='ADMITTED' if tuples else 'REJECTED',reasons=[] if tuples else ['no_valid_h4_k24_windows'],
            candidate_windows=candidate,admitted_windows=len(tuples),rejected_windows=candidate-len(tuples),window_rejection_reasons=dict(counts),
            path=str(folder),contact_path=str(cf),event_path=str(ef),objects=job['objects'],
            hand_meta_identity=metadata_identity(folder/'meta.json'),contact_meta_identity=metadata_identity(cf/'meta.json'),event_meta_identity=metadata_identity(ef/'meta.json'),
            label_object_pose_sha256=digest(folder/'object_poses.npy'),label_filesystem_mtime_ns=(folder/'object_poses.npy').stat().st_mtime_ns,
            timestamp_scope='filesystem_only_not_reset_proof',window_shard=str(shard),window_shard_sha256=digest(shard),
            conversion_actual_bytes=sum(p.stat().st_size for p in conversion_files) if row['status']=='CONVERT' else 0,
            conversion_expected_bytes=row['conversion_bytes'],elapsed_s=round(time.monotonic()-start,3),disk_free_bytes=shutil.disk_usage(out).free)
        if result['conversion_actual_bytes']!=row['conversion_bytes']:
            raise ValueError('exact conversion size disagreement')
        write(receipt,result,fixed=True)
        return result
    except (OSError,MemoryError,InterruptedError):raise
    except Exception as exc:
        result=dict(base,status='REJECTED',reasons=['conversion_or_contract_failed'],detail=repr(exc),admitted_windows=0,
            elapsed_s=round(time.monotonic()-start,3))
        write(receipt,result,fixed=True);return result


def execute(out,workers,device):
    inventory=read(out/'inventory.json')
    if inventory['plan_sha256']!=digest(PLAN) or inventory['registry_sha256']!=digest(REGISTRY/'manifest.json'):
        raise ValueError('input drift since preflight')
    if not inventory['fits']: raise ValueError('preflight does not fit30GiB floor')
    for sub in ['hands/sequences','contacts','events','receipts','windows','batches']:(out/sub).mkdir(parents=True,exist_ok=True)
    contract=out/'contract.json'
    if not contract.exists():
        write(contract,dict(identity=hashlib.sha256((inventory['plan_sha256']+'full-rigid-v1').encode()).hexdigest(),training_allowed=False))
    old={r['key']:r for r in read(OLD/'recipe/manifest.json')['sequences']}
    old_events={r['key']:r for r in read(OLD/'events/manifest.json')['sequences']}
    jobs={r['key']:r for r in read(PLAN)['jobs']}
    if (out/'recovered-inputs.json').exists():
        recovery=read(out/'recovered-inputs.json')
        if recovery['status']!='COMPLETED' or recovery['plan_sha256']!=inventory['plan_sha256']:
            raise ValueError('recovery is not bound to frozen plan')
        mapping={r['key']:r for r in recovery['rows']}
        for row in inventory['rows']:
            if row['key'] in mapping:
                if row['status']!='REJECT':raise ValueError('recovery cannot replace existing admitted input')
                row.update(mapping[row['key']]['inventory_updates'])
                jobs[row['key']]=mapping[row['key']]['job']
                row['split'] = next(iter(jobs[row['key']]['windows']))
    if (out/'additional-local-jobs.json').exists():
        additions=read(out/'additional-local-jobs.json')
        if additions['status']!='COMPLETED' or additions['plan_sha256']!=inventory['plan_sha256']:
            raise ValueError('additional local parent is not bound to frozen plan')
        for extra in additions['rows']:
            if extra['job']['key'] in jobs or extra['inventory']['sequence_id']!=len(inventory['rows']):
                raise ValueError('append-only local parent IDs required')
            jobs[extra['job']['key']]=extra['job'];inventory['rows'].append(extra['inventory'])
    started=time.monotonic();total=len(inventory['rows']);done=Counter();windows=0;minimum=shutil.disk_usage(out).free
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    attempt_id=str(time.time_ns())
    run_manifest_path=out/('run-manifest.json' if not (out/'run-manifest.json').exists() else 'run-manifest-resume-%d.json'%time.time_ns())
    write(run_manifest_path,dict(experiment_id=EXPERIMENT,run_id=out.name,work_version='full-rigid-preparation.v1',
        git_commit=commit,branch='main',command=sys.argv,environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,device=device,
        torch=__import__('torch').__version__,cuda=__import__('torch').version.cuda,host=platform.node()),
        additional_local_jobs_sha256=digest(out/'additional-local-jobs.json') if (out/'additional-local-jobs.json').exists() else None,
        native_recovery_sha256=digest(out/'recovered-inputs.json') if (out/'recovered-inputs.json').exists() else None,
        seed=170,workers=workers,storage_floor_bytes=FLOOR,training_allowed=False,training_updates=0,
        source_read_only=True,historical_outputs_deleted=0,conversion_started=True,started_at=datetime.now(timezone.utc).isoformat()))
    # Small bounded batches avoid queuing tens of thousands of large results.
    ordered=sorted(inventory['rows'],key=lambda r:(r['status']!='REUSE',r['status']=='REJECT',r['key']))
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(device,str(out))) as pool:
        for offset in range(0,total,64):
            guard(out,64*1024**2)
            if (out/'STOP').exists():raise InterruptedError('owner STOP')
            batch=ordered[offset:offset+64]
            peak=sum(sorted((r['conversion_bytes'] for r in batch),reverse=True)[:workers])+64*1024**2
            guard(out,peak)
            args=[(str(out),r,jobs[r['key']],old.get(r['key']),old_events.get(r['key'])) for r in batch]
            results=list(pool.map(process_one,args,chunksize=1))
            for r in results:done[r['status']]+=1;windows+=r['admitted_windows']
            minimum=min(minimum,shutil.disk_usage(out).free)
            progress=dict(status='RUNNING',processed=offset+len(batch),total=total,counts=dict(done),admitted_windows=windows,
                elapsed_s=round(time.monotonic()-started,2),disk_free_bytes=shutil.disk_usage(out).free,minimum_free_bytes=minimum,
                estimated_remaining_s=(time.monotonic()-started)*(total-offset-len(batch))/(offset+len(batch)),
                conversion_started=True,training_allowed=False)
            write(out/'batches'/('batch-%05d-attempt-%s.json'%(offset//64,attempt_id)),dict(progress,keys=[r['key'] for r in batch],receipts=['receipts/'+r['key']+'.json' for r in batch]))
            write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    write(out/'progress.json',dict(progress,status='CONVERSION_AND_ADMISSION_COMPLETED'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['inventory','execute'],required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',default='cpu')
    a=parser.parse_args();out=a.output.resolve()
    if out.parent!=(ROOT/'outputs/ptv3-shared-wam-v6').resolve():raise ValueError('task-owned output required')
    if not 1<=a.workers<=12:raise ValueError('1..12 bounded workers')
    if a.mode=='inventory':inventory(out)
    else:
        try:execute(out,a.workers,a.device)
        except BaseException as exc:
            write(out/'stop-receipt.json',dict(status='STOPPED',reason=repr(exc),disk_free_bytes=shutil.disk_usage(out).free,
                resumable=True,command=sys.argv,training_allowed=False));raise


if __name__=='__main__':main()
