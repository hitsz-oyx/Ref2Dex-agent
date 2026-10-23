"""Bound all-window census to sequence receipts, native clocks and frozen splits.

The manifest is normalized: NPY holds every ordered window tuple; JSONL holds
its sequence-level provenance and shard binding. No content-uniqueness claim.
This audit neither trains nor changes the old curated selection.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_full_rigid_data as p
from summarize_full_rigid_data import bound_split

NAMES = {0:'OakInk2',1:'GRAB',2:'ARCTIC',4:'Shadow replay',5:'TACO',
         6:'HOT3D',8:'DexYCB',9:'H2O',10:'HOI4D'}
CERTIFICATE = {'task_spec', 'source_role', 'available_before_reset',
               'reset_timestamp_ns', 'source_available_timestamp_ns',
               'source_artifact_sha256'}


def clock_mask(job, row, tuples, clocks):
    """Return causal-window continuity separately from geometric admission."""
    sid=row['source_id']
    native={}
    n=row['frames']
    bad=np.zeros(max(0,n-1),bool)
    if sid==0:
        evidence=clocks[row['key']]
        if evidence['frames']!=n:raise ValueError('source0 clock length drift')
        bad[np.asarray(evidence['bad_edge_indices'],np.int64)]=True
        native=dict(fields=['observed_frame','native clock audit'],
                    certificate_fields=[],kind='observational_clock_not_reset_availability')
    elif sid==2:
        object_path=job['raw_path'].replace('.mano.npy','.object.npy')
        if len(np.load(object_path,mmap_mode='r',allow_pickle=False))!=n:
            raise ValueError('ARCTIC native length drift')
        native=dict(fields=['mano.npy','object.npy'],certificate_fields=[],
                    kind='recorded_human_motion_no_predeclared_pose')
    else:
        uri=job['raw_path'] if sid==1 else job['source_row']['source_path']
        with np.load(uri,allow_pickle=False) as z:
            fields=set(z.files)
            native=dict(uri=uri,fields=sorted(fields),
                certificate_fields=sorted(CERTIFICATE & fields),
                complete_certificate_fields=CERTIFICATE.issubset(fields))
            if sid==1:
                if float(z['framerate'])!=120 or (int(z['n_frames'])+3)//4!=n:
                    raise ValueError('GRAB native clock drift')
            else:
                t=z['time']
                if t.shape!=(n,) or not np.isfinite(t).all():
                    raise ValueError('native clock invalid')
                bad |= ~np.isclose(np.diff(t),1/30,rtol=0,atol=1e-5)
                if 'segment' in fields:
                    segment=z['segment']
                    if segment.shape!=(n,):raise ValueError('segment length')
                    bad |= np.diff(segment)!=0
            if 'goal' in fields:
                g=z['goal']
                native.update(goal_shape=list(g.shape),goal_finite=bool(np.isfinite(g).all()),
                    goal_constant=bool(np.array_equal(g,np.broadcast_to(g[:1],g.shape))))
            native['intent_label_fields']=sorted(fields & {'native_action_label','native_verb_label'})
    if sid in (0,2):
        source=Path(row['source'])
        meta=(source.parent if source.is_file() else source)/'meta.json'
        if meta.exists():
            fields=set(p.read(meta))
            native.update(metadata_uri=str(meta),metadata_fields=sorted(fields),
                certificate_fields=sorted(CERTIFICATE & fields),
                complete_certificate_fields=CERTIFICATE.issubset(fields))
    prefix=np.r_[0,np.cumsum(bad,dtype=np.int64)]
    ticks=tuples[:,2]
    return prefix[ticks+24]==prefix[ticks-3],native


def disk_census(run):
    """Apparent bytes, actual allocated blocks and partial attempts separately."""
    groups=defaultdict(Counter);partial=Counter();inodes=set();duplicates=Counter()
    for parent,dirs,files in os.walk(run,followlinks=False):
        for name in files:
            file=Path(parent)/name
            if file.is_symlink():continue
            try:st=file.stat()
            except FileNotFoundError:continue
            rel=file.relative_to(run);group=rel.parts[0] if len(rel.parts)>1 else 'run-root'
            key=(st.st_dev,st.st_ino)
            groups[group]['apparent_bytes']+=st.st_size
            groups[group]['files']+=1
            if key in inodes:duplicates['hardlink_aliases']+=1
            else:
                inodes.add(key);groups[group]['allocated_bytes']+=st.st_blocks*512
            if any('.partial-' in part or part.endswith('.part') for part in rel.parts):
                partial['files']+=1;partial['apparent_bytes']+=st.st_size
                partial['allocated_bytes']+=st.st_blocks*512
    return dict(groups={k:dict(v) for k,v in groups.items()},partial_attempts=dict(partial),
                **duplicates,scope='file blocks only; live sequential scan, directory/inode overhead separate')


def audit(run,out):
    if out.exists():raise FileExistsError(out)
    p.guard(run,512*1024**2);out.mkdir()
    started=time.monotonic()
    inventory=p.read(run/'inventory.json')
    rows={r['key']:dict(r) for r in inventory['rows']}
    jobs={r['key']:r for r in p.read(p.PLAN)['jobs']}
    if (run/'recovered-inputs.json').exists():
        for r in p.read(run/'recovered-inputs.json')['rows']:
            rows[r['key']].update(r['inventory_updates']);jobs[r['key']]=r['job']
    if (run/'additional-local-jobs.json').exists():
        for r in p.read(run/'additional-local-jobs.json')['rows']:
            rows[r['job']['key']]=r['inventory'];jobs[r['job']['key']]=r['job']
    receipts=[p.read_receipt(f) for f in sorted((run/'receipts').glob('*.json'))]
    if len({r['key'] for r in receipts})!=len(receipts):raise ValueError('duplicate receipt key')
    if any(r['input_plan_sha256']!=inventory['plan_sha256'] for r in receipts):
        raise ValueError('receipt plan drift')
    sequence_counts=Counter();window_counts=Counter();coverage=Counter();objects=defaultdict(set)
    native_counts=defaultdict(Counter);reasons=defaultdict(Counter);candidates=Counter()
    old=p.read(p.PILOT)['selection']['train_window_rows']
    old_keys={(r['sequence_key'],int(r['slot']),int(r['tick'])) for r in old}
    old_by_seq=defaultdict(set);old_found=set();old_sources=Counter()
    for r in old:
        old_by_seq[r['sequence_key']].add((int(r['slot']),int(r['tick'])))
        old_sources[r['source_id']]+=1
    lengths=Counter()
    for r in receipts:
        split=bound_split(r['split'],jobs[r['key']]);lengths[split]+=r['admitted_windows']
    arrays={split:np.lib.format.open_memmap(out/(split+'-all.npy'),mode='w+',dtype=np.int32,
        shape=(lengths[split],3)) for split in ('train','dev')}
    clock_arrays={split:np.lib.format.open_memmap(out/(split+'-clock-valid.npy'),mode='w+',dtype=bool,
        shape=(lengths[split],)) for split in arrays}
    positions=Counter();completed_conversion=0;reused_conversion=0;bad_split_types=0;nulls=Counter()
    clocks={r['key']:r for r in p.read(run/'source0-clock-audit.json')['rows']}
    progress_before=p.read(run/'progress.json')
    with (out/'sequences.jsonl').open('x') as ledger:
        for index,r in enumerate(sorted(receipts,key=lambda r:r['sequence_id'])):
            sid=r['source_id'];key=r['key'];row=rows[key];job=jobs[key]
            raw_split=r['split'];split=bound_split(raw_split,job)
            bad_split_types+=isinstance(raw_split,list)
            sequence_counts[(sid,split,'decided')]+=1
            sequence_counts[(sid,split,r['status'])]+=1
            reasons[sid].update(r.get('reasons',[]))
            if r.get('source_available_timestamp_ns') is None:nulls[sid]+=r['admitted_windows']
            completed_conversion+=r.get('conversion_actual_bytes',0)
            if r['status']=='ADMITTED':
                shard=Path(r['window_shard'])
                if p.digest(shard)!=r['window_shard_sha256']:raise ValueError('window shard drift')
                with np.load(shard,allow_pickle=False) as z:
                    tuples=z['tuples'];phases=z['phase'];classes=z['geometry_class']
                if len(tuples)!=r['admitted_windows'] or not (tuples[:,0]==r['sequence_id']).all():
                    raise ValueError('shard/receipt mismatch')
                valid,native=clock_mask(job,row,tuples,clocks)
                start=positions[split];arrays[split][start:start+len(tuples)]=tuples
                clock_arrays[split][start:start+len(tuples)]=valid;positions[split]+=len(tuples)
                window_counts[(sid,split,'admitted')]+=len(tuples)
                window_counts[(sid,split,'clock_valid')]+=int(valid.sum())
                candidates[(sid,split)]+=r.get('candidate_windows',len(tuples))
                for phase in range(5):
                    for gc in range(4):
                        count=int(((phases==phase)&(classes==gc)&valid).sum())
                        coverage[(sid,split,p.PHASES[phase],p.CLASSES[gc])]+=count
                for slot in np.unique(tuples[valid,1]):objects[(sid,split)].add(job['objects'][int(slot)])
                for slot,tick in old_by_seq.get(key,set()):
                    if split=='train' and np.any((tuples[:,1]==slot)&(tuples[:,2]==tick)&valid):
                        old_found.add((key,slot,tick))
                native_counts[sid]['inspected_mothers']+=1
                native_counts[sid]['inspected_windows']+=len(tuples)
                if native.get('complete_certificate_fields'):native_counts[sid]['complete_field_candidates']+=len(tuples)
                if 'goal' in native.get('fields',[]):native_counts[sid]['native_goal_windows']+=len(tuples)
                if native.get('intent_label_fields'):native_counts[sid]['native_action_or_verb_windows']+=len(tuples)
                if row['status']=='REUSE':reused_conversion+=sum(f.stat().st_size for parent in [r['path'],r['contact_path'],r['event_path']] for f in Path(parent).iterdir() if f.is_file())
                r=dict(r,split_recorded=raw_split,split=split,native_evidence=native,
                       clock_valid_windows=int(valid.sum()),normalized_offset=start)
            ledger.write(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')
            if (index+1)%4000==0:
                p.guard(run,64*1024**2)
                print(json.dumps(dict(stage='readiness_census',sequences=index+1,total=len(receipts),elapsed_s=time.monotonic()-started)),flush=True)
    for a in list(arrays.values())+list(clock_arrays.values()):a.flush()
    if any(c['complete_field_candidates'] for c in native_counts.values()):
        raise ValueError('potential certificate found; inspect before declaring missing provenance')
    disk=disk_census(run)
    old_inventory=p.read(p.ROOT/'outputs/ptv3-shared-wam-v6/tau-curated-20261023-r5-inventory/inventory.json')
    pending_keys={r['key'] for r in old_inventory['rows']}
    pending_fixed=sum(rows[k]['conversion_bytes'] for k in pending_keys)
    outside_fixed=sum(r['conversion_bytes'] for k,r in rows.items() if k not in pending_keys)
    total_fixed=sum(r['conversion_bytes'] for r in rows.values())
    # Physical fixed files currently published or partial, measured separately
    # from window shards, receipts, restored inputs and later dedup products.
    physical_conversion=sum(disk['groups'].get(k,{}).get('apparent_bytes',0) for k in ['hands','contacts','events'])
    summary_sources=[]
    for sid in p.SOURCES:
        entry=dict(source_id=sid,name=NAMES[sid],planned_mothers=sum(r['source_id']==sid for r in rows.values()),
            old6000_windows=old_sources[sid],objects={s:len(objects[(sid,s)]) for s in arrays},
            native_audit=dict(native_counts[sid]),null_timestamp_windows=nulls[sid],rejection_reasons=dict(reasons[sid]))
        for split in arrays:
            entry[split]=dict(sequences={k:sequence_counts[(sid,split,k)] for k in ['decided','ADMITTED','REJECTED']},
                candidate_windows=candidates[(sid,split)],
                admitted_windows=window_counts[(sid,split,'admitted')],
                clock_valid_windows=window_counts[(sid,split,'clock_valid')],
                phase_geometry={phase:{gc:coverage[(sid,split,phase,gc)] for gc in p.CLASSES} for phase in p.PHASES})
        entry['train_net_count_vs6000']=entry['train']['clock_valid_windows']-old_sources[sid]
        summary_sources.append(entry)
    import csv
    with (out/'phase-geometry-source.csv').open('x') as stream:
        writer=csv.writer(stream);writer.writerow(['source_id','split','phase','geometry_class','windows'])
        for sid in p.SOURCES:
            for split in arrays:
                for phase in p.PHASES:
                    for gc in p.CLASSES:writer.writerow([sid,split,phase,gc,coverage[(sid,split,phase,gc)]])
    free=shutil.disk_usage(run).free
    decided_keys={x['key'] for x in receipts}
    missing=[r['key'] for r in rows.values() if r['key'] not in decided_keys]
    budget=p.read(run/'storage-budget-v7.json')
    full_status=p.read(run/'progress.json')
    complete=not missing and full_status['status']=='CONVERSION_AND_ADMISSION_COMPLETED'
    result=dict(schema='ref2dex.nine-source-readiness.v1',measured_at=datetime.now(timezone.utc).isoformat(),
        status='COMPLETE_CENSUS' if complete else 'LIVE_SNAPSHOT',training_allowed=False,
        task='ptv3-shared-wam-v6',run_id=run.name,experiment_id=p.EXPERIMENT,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,host=platform.node()),
        expected_mothers=len(rows),decided_mothers=len(receipts),missing_receipts=missing,
        historical_receipt_split_lists_normalized=bad_split_types,sources=summary_sources,
        windows={s:dict(admitted=len(arrays[s]),clock_valid=int(clock_arrays[s].sum())) for s in arrays},
        all_nine_sources_admitted=all(r['train']['clock_valid_windows']>0 for r in summary_sources),
        baseline6000=dict(file=str(p.PILOT),file_sha256=p.digest(p.PILOT),rows=len(old),
            rows_present=len(old_found),missing_rows=sorted(old_keys-old_found),
            added_train_rows=int(clock_arrays['train'].sum())-len(old_found),
            note='Exact tuple set difference, NOT independent-content increment; content dedup separate.'),
        provenance=dict(certified_legacy_windows=0,missing_timestamp_windows=sum(nulls.values()),
            certificate_field_candidates=sum(c['complete_field_candidates'] for c in native_counts.values()),
            scope='Bound native payloads and explicitly restored inputs; external lost archives not adjudicated.'),
        storage=dict(disk=disk,current_free_bytes=free,old_estimate_bytes=old_inventory['estimated_full_cache_bytes'],
            pending6000_current_exact_bytes=pending_fixed,outside_pending6000_exact_bytes=outside_fixed,
            total_fixed_exact_bytes=total_fixed,receipted_new_fixed_bytes=completed_conversion,
            reused_existing_bytes=reused_conversion,physical_conversion_bytes=physical_conversion,
            outstanding_unreceipted_exact_bytes=sum(rows[k]['conversion_bytes'] for k in missing),
            remaining_fixed_net_estimate_bytes=max(0,total_fixed-physical_conversion),
            published_or_partial_excess_over_receipted_bytes=physical_conversion-completed_conversion,
            manifest_peak_reserve_bytes=budget['manifest_and_receipt_upper_bound_bytes'],
            safety_floor_bytes=p.FLOOR,filesystem_overhead_reserve_bytes=2*p.GIB),
        coverage_train_zero_cells=sum(coverage[(sid,'train',phase,gc)]==0 for sid in p.SOURCES for phase in p.PHASES[:4] for gc in p.CLASSES[:3]),
        content_dedup_status='SEPARATE_NOT_CLAIMED_COMPLETE',
        progress_before=progress_before,progress_after=full_status,elapsed_s=time.monotonic()-started,
        files={f.name:dict(bytes=f.stat().st_size,sha256=p.digest(f)) for f in out.iterdir() if f.is_file()})
    p.write(out/'manifest.json',result)
    (out/'manifest.sha256').write_text(p.digest(out/'manifest.json')+'  manifest.json\n')
    print(json.dumps(dict(status=result['status'],windows=result['windows'],free_gib=free/p.GIB,manifest=str(out/'manifest.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.run.resolve(),args.output.resolve())
