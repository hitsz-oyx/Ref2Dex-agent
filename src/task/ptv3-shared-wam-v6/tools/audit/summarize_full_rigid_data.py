"""Publish full TRAIN/DEV manifests, conservative content dedup and zero cells.

This is preparation only. Missing pre-reset evidence remains UNKNOWN/null in
every admitted window; a filesystem timestamp is never a causal certificate.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_full_rigid_data as prep
from ptv3_shared_wam_v6.rigid_dataset import duplicate_of, pca_shape
from prepare_local_data import npy_header


def bound_split(recorded, job):
    """Normalize historical singleton-list receipts against frozen membership."""
    expected = list(job['windows'])
    actual = recorded if isinstance(recorded, list) else [recorded]
    if len(expected) != 1 or actual != expected or expected[0] not in ('train', 'dev'):
        raise ValueError('receipt/frozen split disagreement')
    return expected[0]


def jsonline(stream,value):
    stream.write(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')


def eager_npz(path):
    # NpzFile is lazy: repeated member indexing reloads the complete array.
    # Materialize once per sequence before entering a dense window loop.
    with np.load(path,allow_pickle=False) as z:return {name:z[name] for name in z.files}


@lru_cache(maxsize=6)
def sequence(folder,contact):
    s=prep.load_arrays(folder,['joints','joint_valid','object_poses','object_pose_valid'])
    c=prep.load_arrays(contact,['contact','part','valid'])
    # Per-frame part fractions are computed once, not once per overlapping H28.
    n,slots=c['valid'].shape;fractions=np.zeros((n,slots,16),np.float32)
    for slot in range(slots):
        positive=(np.asarray(c['contact'][:,slot])>=.25)&np.asarray(c['valid'][:,slot,None])
        labels=np.asarray(c['part'][:,slot])
        frame,point=np.nonzero(positive)
        bins=frame*16+labels[frame,point]
        fractions[:,slot]=np.bincount(bins,minlength=n*16).reshape(n,16)/2048
        fractions[:,slot]*=c['valid'][:,slot,None]
    return s,c,fractions


def anchored(s,geo,slot,tick,offsets):
    ids=tick+np.asarray(offsets)
    poses=np.asarray(s['object_poses'][ids,slot],np.float64)
    anchor=np.asarray(s['object_poses'][tick,slot],np.float64)
    # Exact inverse of the recorded SE3; use same convention as the scalar
    # dedup feature, including small measured numerical rotation residuals.
    inverse=np.linalg.inv(anchor);rel=inverse@poses
    joints=np.asarray(s['joints'][ids],np.float64)
    valid=np.asarray(s['joint_valid'][ids])
    hand=np.where(valid[...,None],joints@inverse[:3,:3].T+inverse[:3,3],0)
    query=np.einsum('tij,pj->tpi',rel[:,:3,:3],geo['query'])+rel[:,None,:3,3]
    return hand,query,rel


def shape_of(geo):
    if '_shape' not in geo:geo['_shape']=pca_shape(geo['surface'])
    return geo['_shape']


def full_feature(record,slot,tick,phase,geo):
    s,c,f=sequence(record['path'],record['contact_path'])
    hand,query,rel=anchored(s,geo,slot,tick,range(-3,25))
    return dict(shape=shape_of(geo),phase=int(phase),
        trajectory=np.concatenate([hand.reshape(-1),query.reshape(-1)]).astype(np.float32),
        contact=f[tick-3:tick+25,slot].reshape(-1),rotations=rel[:,:3,:3].astype(np.float32),
        mask=np.concatenate([s['joint_valid'][tick-3:tick+25].reshape(-1),
            c['valid'][tick-3:tick+25,slot],s['object_pose_valid'][tick-3:tick+25,slot]]))


def feature_signature(f):
    offsets=[0,3,11,27]
    hand=f['trajectory'][:28*2*21*3].reshape(28,2,21,3)[offsets]
    query=f['trajectory'][28*2*21*3:].reshape(28,32,3)[offsets]
    h=hashlib.blake2b(digest_size=20)
    j=28*2*21
    for value in [np.rint(np.log(np.maximum(f['shape'],1e-9))/np.log(1.05)).astype('<i4'),
        np.rint(hand/.002).astype('<i4'),np.rint(query/.002).astype('<i4'),
        np.rint(f['rotations'][offsets]/.01).astype('<i4'),
        np.rint(f['contact'].reshape(28,16)[offsets]/.02).astype('<i4'),
        np.packbits(f['mask'][:j]),np.packbits(f['mask'][j:j+28]),np.packbits(f['mask'][j+28:]),
        np.asarray([f['phase']],np.uint8)]:h.update(value.tobytes())
    return h.digest()


def init_local_worker(registry):
    prep.REGISTRY=Path(registry)
    prep._GEOMETRY_ROWS={r['key']:r for r in prep.read(prep.REGISTRY/'manifest.json')['objects']}


def timing_gate(record,tuples,n):
    """Dense ticks must not cross a clock gap or an existing segment edge."""
    kind=record['clock_kind'];bad=np.zeros(max(0,n-1),bool)
    evidence=dict(clock_kind=kind,source_role='UNKNOWN',available_before_reset=None,
        scope='bound native payload and exported sidecars; no filesystem mtime used as pre-reset proof',candidate_fields={})
    def headers(z):
        evidence['exported_fields']=list(z.files)
        selected={'goal','motion_intent','native_action_label','native_verb_label','task_spec','task_goal_pose',
            'source_role','available_before_reset','reset_timestamp_ns','source_artifact_sha256','pre_reset_goal'}
        for name in selected.intersection(z.files):
            with z.zip.open(name+'.npy') as f:h=npy_header(f)
            evidence['candidate_fields'][name]=dict(shape=list(h['shape']),dtype=h['descr'])
        complete={'source_role','available_before_reset','reset_timestamp_ns','source_artifact_sha256','task_spec'}
        if complete.issubset(z.files):raise ValueError('potential complete pre-reset certificate requires inspection, not blanket UNKNOWN')
        if 'goal' in z.files:
            g=z['goal']
            evidence.update(native_goal_candidate_role='recorded_task_parameter_snapshot_after_reset',
                native_goal_shape=list(g.shape),native_goal_finite=bool(np.isfinite(g).all()),
                native_goal_constant=bool(np.array_equal(g,np.broadcast_to(g[:1],g.shape))))
            evidence['missing_for_complete_certificate']=['original executor/reset/TaskSpec/source binding','pre-reset role and availability proof','complete intended pose or declared pose rule']
    if kind=='source0_observed_stride4':
        if record['clock_frames']!=n:raise ValueError('source0 clock length changed')
        bad[np.asarray(record['clock_bad_edges'],np.int64)]=True
    elif kind=='grab_native120_stride4':
        with np.load(record['clock_uri'],allow_pickle=False) as z:
            headers(z)
            if float(z['framerate'])!=120 or (int(z['n_frames'])+3)//4!=n:
                raise ValueError('GRAB native clock changed')
    elif kind=='arctic_native30':
        if len(np.load(record['clock_uri'],mmap_mode='r',allow_pickle=False))!=n:
            raise ValueError('ARCTIC native clock length changed')
    elif kind=='packed_grid30':
        with np.load(record['clock_uri'],allow_pickle=False) as z:
            headers(z)
            t=z['time']
            if t.shape!=(n,) or not np.isfinite(t).all():raise ValueError('invalid native clock')
            bad |= ~np.isclose(np.diff(t),1/30,rtol=0,atol=1e-5)
            if 'segment' in z:
                segment=z['segment']
                if segment.shape!=(n,):raise ValueError('invalid segment shape')
                bad |= np.diff(segment)!=0
    else:raise ValueError('unknown clock contract')
    ticks=tuples[:,2];prefix=np.r_[0,np.cumsum(bad,dtype=np.int64)]
    legal=prefix[ticks+24]==prefix[ticks-3]
    if not legal.all():
        raise ValueError('dense admission crossed native clock/segment boundary: %s windows=%d' % (record['key'],int((~legal).sum())))
    record['_native_evidence']=evidence
    return int(bad.sum())


def local_dedup(args):
    """Parallel per-sequence work; global representatives remain serial/strict."""
    r,folder=args;folder=Path(folder)
    shard=np.load(r['window_shard'],allow_pickle=False);tuples=shard['tuples'];phases=shard['phase']
    source,_,_=sequence(r['path'],r['contact_path'])
    bad_edges=timing_gate(r,tuples,len(source['joints']))
    keep=np.ones(len(tuples),bool);representatives=np.full(len(tuples),-1,np.int32)
    keys=np.zeros((len(tuples),20),np.uint8);previous={}
    for i,(_,slot,tick) in enumerate(tuples):
        slot,tick=int(slot),int(tick);phase=int(phases[i]);_,geo=prep.geometry(r['objects'][slot],r['source_id'])
        f=full_feature(r,slot,tick,phase,geo);p=previous.get(slot)
        if p and phase==p[2] and tick-p[1]<24 and duplicate_of(f,p[3]):
            keep[i]=False;representatives[i]=p[0]
        else:
            keys[i]=np.frombuffer(feature_signature(f),np.uint8)
            previous[slot]=(i,tick,phase,f)
    target=folder/(r['key']+'.npz');pending=target.with_name(target.name+'.part')
    with pending.open('wb') as f:np.savez(f,keep=keep,representatives=representatives,signatures=keys,
        timing_checked_windows=np.int64(len(tuples)),native_bad_edges=np.int64(bad_edges),
        native_evidence_json=np.asarray(json.dumps(r['_native_evidence'],sort_keys=True)))
    pending.replace(target)
    return str(target)


def local_results(rows,out,workers):
    folder=out/'local-dedup-shards';folder.mkdir()
    if workers==1:
        for r in rows:yield r,local_dedup((r,str(folder)))
    else:
        with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=init_local_worker,initargs=(str(prep.REGISTRY),)) as pool:
            for start in range(0,len(rows),64):
                batch=rows[start:start+64]
                for r,path in zip(batch,pool.map(local_dedup,[(r,str(folder)) for r in batch])):yield r,path


def matrix(counter,path):
    rows=[dict(phase=p,geometry_class=g,source_id=s,windows=counter[(p,g,s)])
          for p in prep.PHASES for g in prep.CLASSES for s in prep.SOURCES]
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['phase','geometry_class','source_id','windows'])
        writer.writeheader();writer.writerows(rows)
    return dict(zero_requested_cells=sum(r['windows']==0 for r in rows if r['phase']!='unknown' and r['geometry_class']!='unknown'),
        requested_cells=108,zero_cells=[r for r in rows if r['windows']==0 and r['phase']!='unknown' and r['geometry_class']!='unknown'])


def legacy_view_folder(root,row):
    if row.get('view_manifest'):return Path(row['view_manifest']).parent,'declared_view_manifest'
    folder=Path(root)/row['path']
    if folder.is_dir():return folder,'declared_relative_path'
    if row.get('geometry_root'):
        native=Path(row['geometry_root']).parent
        if native.is_dir():return native,'declared_geometry_parent'
    return folder,'declared_path_missing'


def legacy_census(out,receipts):
    data=Path('/home2/wyy/oyx_ws/Ref2Dex/data/processed_data')
    by_parent={r['source_sequence']:r for r in receipts if r['source_id']==1}
    sys.path.insert(0,str(prep.TASK.parent/'cm-pointflow-effect-pretrain/src'))
    from oakink_wm.native_splits import grab_split
    counts=Counter(); details=[]
    with (out/'legacy-mother-verdicts.jsonl').open('x') as f:
        for folder in sorted((data/'dexplore_grab/sequences').iterdir()):
            if not folder.is_dir():continue
            parent='grab_'+folder.name; r=by_parent.get(parent)
            official=grab_split(folder.name.split('_',1)[1])
            record=dict(parent_sequence_id=parent,source_id=1,source_path=str(folder),
                status=r['status'] if r else 'REJECTED',
                admitted_windows=r['admitted_windows'] if r else 0,
                reasons=r['reasons'] if r else (['official_test_holdout_excluded_from_train_dev'] if official=='test' else ['no_frozen_native_sequence_and_bound_v6_adapter']),
                official_split=official,decision_scope='TRAIN_DEV_membership; TEST exclusion is not a physical quality rejection',
                sequence_receipt=('receipts/'+r['key']+'.json') if r else None,
                views_are_not_additional_demonstrations=True)
            counts[record['status']]+=1;details.append(record);jsonline(f,record)
    views=Counter();view_reasons=Counter();candidate=0
    roots=[('cm_decoder_v2',data/'cm_decoder_v2/dexplore_rl_v1_3_full10135/manifest.json'),
           ('coupled',data/'coupled_geometric_cache_v1_20260912/index.json')]
    with (out/'legacy-view-verdicts.jsonl').open('x') as f:
        for name,path in roots:
            for split,rows in prep.read(path)['sequences'].items():
                for row in rows:
                    parent=row.get('parent_seq_id',row['id'])
                    # Official TEST MANO views have a relative path but no
                    # view_manifest; retain their explicit audit verdict too.
                    folder,resolution=legacy_view_folder(path.parent,row)
                    required=['joints.npy','joint_valid.npy','surface_offsets.npy','surface_valid.npy','surface_parts.npy','object_poses.npy','object_pose_valid.npy']
                    missing=[p for p in required if not (folder/p).is_file()]
                    reasons=[]
                    if row.get('qualitative_only'):reasons.append('declared_qualitative_only_view_not_training_source')
                    if any(k in missing for k in ['joints.npy','joint_valid.npy']):reasons.append('native_hand21_semantics_or_validity_missing')
                    if any(k.startswith('surface') for k in missing):reasons.append('native_hand_surface778_part16_adapter_missing')
                    if 'object_pose_valid.npy' in missing:reasons.append('annotation_pose_validity_missing')
                    if not (folder/'contact/meta.json').is_file():reasons.append('bound_v6_contact_part_valid_missing')
                    if not (folder/'events/meta.json').is_file():reasons.append('bound_v6_event_missing')
                    if not reasons:raise ValueError('legacy gained a semantic adapter: explicit full admission required')
                    windows=max(0,int(row['frame_count'])-27);candidate+=windows
                    views[name]+=1;view_reasons.update(reasons)
                    jsonline(f,dict(root=name,parent_sequence_id=parent,split=split,path=str(folder),
                        path_resolution=resolution,variant=row.get('variant'),native_source=row.get('source'),
                        status='REJECTED',reasons=reasons,missing_files=missing,candidate_windows=windows,admitted_windows=0,
                        no_human_parent_labels_substituted=True))
    return dict(unique_mothers=len(details),mother_status=dict(counts),views_by_root=dict(views),
        view_candidate_windows=candidate,view_admitted=0,view_rejection_reasons=dict(view_reasons),
        note='1335 human parent mothers; robot-derived views do not inherit human labels or add mother count')


def summarize(run,out,workers=1):
    progress=prep.read(run/'progress.json')
    if progress['status']!='CONVERSION_AND_ADMISSION_COMPLETED':raise ValueError('complete sequence census required')
    if out.exists():raise FileExistsError('fresh summary attempt required; preserve prior artifacts')
    prep.guard(run,256*1024**2);out.mkdir()
    prep._GEOMETRY_ROWS={r['key']:r for r in prep.read(prep.REGISTRY/'manifest.json')['objects']}
    receipts=[prep.read_receipt(p) for p in sorted((run/'receipts').glob('*.json'))]
    inventory=prep.read(run/'inventory.json')
    additional=len(prep.read(run/'additional-local-jobs.json')['rows']) if (run/'additional-local-jobs.json').exists() else 0
    if len(receipts)!=inventory['total_sequences']+additional:raise ValueError('incomplete receipts')
    by_id={r['sequence_id']:r for r in receipts}
    plan_jobs={r['key']:r for r in prep.read(prep.PLAN)['jobs']}
    if additional:
        plan_jobs.update({r['job']['key']:r['job'] for r in prep.read(run/'additional-local-jobs.json')['rows']})
    origins={r['key']:(r.get('raw_path') if r['key'].startswith('s1-local-extra') else r.get('source_row',{}).get('source_path',r.get('sequence_folder'))) for r in plan_jobs.values()}
    clocks={r['key']:r for r in prep.read(run/'source0-clock-audit.json')['rows']} if any(r['source_id']==0 for r in receipts) else {}
    for r in receipts:
        job=plan_jobs[r['key']];sid=r['source_id']
        r['split_recorded_in_receipt'] = r['split']
        r['split'] = bound_split(r['split'], job)
        if sid==0:r.update(clock_kind='source0_observed_stride4',clock_frames=clocks[r['key']]['frames'],clock_bad_edges=clocks[r['key']]['bad_edge_indices'])
        elif sid==1:r.update(clock_kind='grab_native120_stride4',clock_uri=job['raw_path'])
        elif sid==2:r.update(clock_kind='arctic_native30',clock_uri=job['raw_path'].replace('.mano.npy','.object.npy'))
        else:r.update(clock_kind='packed_grid30',clock_uri=job['source_row']['source_path'])
    canonical=Counter((r['source_id'],origins[r['key']]) for r in receipts)
    if any(v!=1 for v in canonical.values()):raise ValueError('duplicate canonical mother payload URI; cannot count as independent mothers')
    with (out/'mother-identities.jsonl').open('x') as stream:
        for r in sorted(receipts,key=lambda r:r['sequence_id']):
            jsonline(stream,dict(sequence_id=r['sequence_id'],sequence_key=r['key'],source_id=r['source_id'],source_sequence=r['source_sequence'],canonical_original_payload_uri=origins[r['key']]))
    legacy=legacy_census(out,receipts)
    pilot=prep.read(prep.PILOT);old_rows=pilot['selection']['train_window_rows']
    old_dev_rows=[r for r in pilot['audit_rows'] if r['split']=='dev' and r['admitted']]
    old_dev_keys={(r['sequence_key'],int(r['slot']),int(r['tick'])) for r in old_dev_rows}
    old_dev_present=set();old_dev_kept=set()
    old_keys={(r['sequence_key'],int(r['slot']),int(r['tick'])) for r in old_rows}
    old_present=set();old_kept=set()
    old_coverage=Counter()
    for r in old_rows:
        _,geo=prep.geometry(r['object'],r['source_id'])
        old_coverage[(r['phase'],prep.geometry_class(pca_shape(geo['surface'])),r['source_id'])]+=1
    matrix(old_coverage,out/'coverage-old-6000.csv')
    old_dev_coverage=Counter()
    for r in old_dev_rows:
        _,geo=prep.geometry(r['object'],r['source_id'])
        old_dev_coverage[(r['phase'],prep.geometry_class(pca_shape(geo['surface'])),r['source_id'])]+=1
    matrix(old_dev_coverage,out/'coverage-old-dev.csv')
    old_dev_sources=Counter(str(r['source_id']) for r in old_dev_rows)
    old_dev_phases=Counter(r['phase'] for r in old_dev_rows);old_dev_shapes=Counter()
    for (phase,gc,sid),v in old_dev_coverage.items():old_dev_shapes[gc]+=v
    old_sources=Counter(str(r['source_id']) for r in old_rows)
    old_phases=Counter(r['phase'] for r in old_rows)
    old_shapes=Counter()
    for (phase,gc,sid),v in old_coverage.items():old_shapes[gc]+=v
    # DEV representatives take priority. Remove matching TRAIN content and
    # audit each cross-split deletion; no held-out row is moved into TRAIN.
    admitted=sorted([r for r in receipts if r['status']=='ADMITTED'],key=lambda r:(r['split']!='dev',r['sequence_id']))
    n=sum(r['admitted_windows'] for r in admitted)
    prep.guard(run,n*2048+256*1024**2)
    arrays={s:np.lib.format.open_memmap(out/(s+'-all.npy'),mode='w+',dtype=np.int32,
            shape=(sum(r['admitted_windows'] for r in admitted if r['split']==s),3)) for s in ['train','dev']}
    keep_arrays={s:np.lib.format.open_memmap(out/(s+'-keep-mask.npy'),mode='w+',dtype=bool,shape=(len(arrays[s]),)) for s in arrays}
    old_membership=np.lib.format.open_memmap(out/'train-membership-in-old6000.npy',mode='w+',dtype=bool,shape=(len(arrays['train']),))
    old_dev_membership=np.lib.format.open_memmap(out/'dev-membership-in-old-version.npy',mode='w+',dtype=bool,shape=(len(arrays['dev']),))
    db=sqlite3.connect(str(out/'dedup-index.sqlite'))
    # This rebuildable index is published only after the entire audit. Keep
    # rollback pages in bounded process memory instead of rewriting the same
    # random B-tree pages to a disk journal for millions of dense windows.
    # Authoritative conversion receipts retain their independent fsyncs.
    db.execute('PRAGMA cache_size=-1048576')
    db.execute('PRAGMA journal_mode=MEMORY')
    db.execute('CREATE TABLE representatives (signature BLOB, sequence_id INTEGER, slot INTEGER, tick INTEGER, phase INTEGER)')
    db.execute('CREATE INDEX signature_index ON representatives(signature)')
    positions=Counter();count=Counter();removed=Counter();cross=0;comparisons=0;temporal_before=0;temporal_after=0
    cov={s:{t:Counter() for t in ['before','after']} for s in arrays}
    sources={s:{t:Counter() for t in ['before','after']} for s in arrays}
    shapes={s:{t:Counter() for t in ['before','after']} for s in arrays}
    phases={s:{t:Counter() for t in ['before','after']} for s in arrays}
    native_candidate_mothers=Counter();native_candidate_windows=Counter()
    started=time.monotonic()
    manifest_streams={s:(out/(s+'-windows.jsonl')).open('x') for s in arrays}
    def temporal_fraction(ticks):
        if len(ticks)<2:return 0
        diff=np.diff(np.sort(ticks));near=diff<24
        return int((np.r_[False,near]|np.r_[near,False]).sum())
    with (out/'dedup-pairs.jsonl').open('x') as pairs,(out/'native-source-evidence.jsonl').open('x') as evidence_stream:
        for number,(r,local_path) in enumerate(local_results(admitted,out,workers),1):
            prep.guard(run,64*1024**2)
            if (run/'STOP').exists():raise InterruptedError('owner STOP')
            split=r['split']
            shard=np.load(r['window_shard'],allow_pickle=False)
            local=eager_npz(local_path)
            native_evidence=json.loads(local['native_evidence_json'].item())
            jsonline(evidence_stream,dict(sequence_id=r['sequence_id'],sequence_key=r['key'],source_id=r['source_id'],**native_evidence))
            for name in native_evidence['candidate_fields']:
                native_candidate_mothers[name]+=1;native_candidate_windows[name]+=r['admitted_windows']
            tuples=shard['tuples'];phase_ids=shard['phase'];class_ids=shard['geometry_class']
            start=positions[split];arrays[split][start:start+len(tuples)]=tuples
            keep_arrays[split][start:start+len(tuples)]=True
            retained_ticks={}; temporal_representatives={};global_changed_local=False
            for slot in np.unique(tuples[:,1]):
                temporal_before+=temporal_fraction(tuples[tuples[:,1]==slot,2]);retained_ticks[int(slot)]=[]
            for i,(_,slot,tick) in enumerate(tuples):
                slot,tick=int(slot),int(tick);phase=int(phase_ids[i]);gc=int(class_ids[i])
                _,geo=prep.geometry(r['objects'][slot],r['source_id'])
                representative=None;feat=None
                # Trust local removal only while its representative is still
                # globally retained. If an earlier representative was removed
                # globally, recompute against the actual retained sequence.
                if not global_changed_local and not local['keep'][i]:
                    representative=tuples[int(local['representatives'][i])].tolist()
                previous=temporal_representatives.get(slot)
                if representative is None and global_changed_local and previous and tick-previous[0]<24 and phase==previous[1]:
                    feat=full_feature(r,slot,tick,phase,geo);comparisons+=1
                    if duplicate_of(feat,full_feature(r,slot,previous[0],previous[1],geo)):
                        representative=[r['sequence_id'],slot,previous[0]]
                candidates=[];key=None
                if representative is None:
                    if local['keep'][i]:key=local['signatures'][i].tobytes()
                    else:
                        if feat is None:feat=full_feature(r,slot,tick,phase,geo)
                        key=feature_signature(feat)
                    candidates=db.execute('SELECT sequence_id,slot,tick,phase FROM representatives WHERE signature=? ORDER BY rowid',(key,)).fetchall()
                for sid,rs,rt,rp in ([] if representative is not None else candidates):
                    comparisons+=1
                    rr=by_id[sid];_,rgeo=prep.geometry(rr['objects'][rs],rr['source_id'])
                    if feat is None:feat=full_feature(r,slot,tick,phase,geo)
                    if duplicate_of(feat,full_feature(rr,rs,rt,rp,rgeo)):
                        representative=[sid,rs,rt];break
                retained=representative is None
                if retained:
                    db.execute('INSERT INTO representatives VALUES (?,?,?,?,?)',(key,r['sequence_id'],slot,tick,phase))
                    retained_ticks[slot].append(tick)
                    temporal_representatives[slot]=(tick,phase)
                else:
                    if local['keep'][i]:global_changed_local=True
                    keep_arrays[split][start+i]=False;removed[split]+=1
                    cross_split=by_id[representative[0]]['split']!=split
                    cross+=int(cross_split)
                    jsonline(pairs,dict(removed=[r['sequence_id'],slot,tick],retained=representative,
                        split=split,cross_split=cross_split,cross_sequence=representative[0]!=r['sequence_id']))
                cell=(prep.PHASES[phase],prep.CLASSES[gc],r['source_id'])
                for stage in ['before']+(['after'] if retained else []):
                    cov[split][stage][cell]+=1;sources[split][stage][str(r['source_id'])]+=1
                    shapes[split][stage][prep.CLASSES[gc]]+=1;phases[split][stage][prep.PHASES[phase]]+=1
                oldkey=(r['key'],slot,tick)
                if split=='train':old_membership[start+i]=oldkey in old_keys
                if split=='train' and oldkey in old_keys:
                    old_present.add(oldkey)
                    if retained:old_kept.add(oldkey)
                if split=='dev':old_dev_membership[start+i]=oldkey in old_dev_keys
                if split=='dev' and oldkey in old_dev_keys:
                    old_dev_present.add(oldkey)
                    if retained:old_dev_kept.add(oldkey)
                jsonline(manifest_streams[split],dict(sequence_id=r['sequence_id'],sequence_key=r['key'],slot=slot,tick=tick,
                    split=split,source_id=r['source_id'],phase=prep.PHASES[phase],geometry_class=prep.CLASSES[gc],
                    source_role='UNKNOWN',available_before_reset=None,certification='UNCERTIFIABLE_MISSING_PRE_RESET_RECORDS',
                    reset_timestamp_ns=None,source_available_timestamp_ns=None,source_artifact_sha256=None,
                    label_object_pose_sha256=r['label_object_pose_sha256'],label_filesystem_mtime_ns=r['label_filesystem_mtime_ns'],
                    retained_after_dedup=retained,sequence_receipt='../receipts/'+r['key']+'.json'))
            for ticks in retained_ticks.values():temporal_after+=temporal_fraction(ticks)
            positions[split]+=len(tuples);count[split]+=len(tuples)
            if number%64==0:
                # This is a derived index, published only after the whole
                # audit. Group journal syncs; per-window semantics/order stay
                # unchanged and conversion receipts remain independently durable.
                db.commit()
                status=dict(stage='full_manifest',sequences=number,total_sequences=len(admitted),windows=dict(count),removed=dict(removed),
                    elapsed_s=round(time.monotonic()-started,2),disk_free_bytes=prep.shutil.disk_usage(run).free)
                prep.write(out/'progress.json',status);print(json.dumps(status),flush=True)
    for stream in manifest_streams.values():stream.close()
    db.commit();db.close()
    for split in arrays:
        arrays[split].flush();keep_arrays[split].flush()
        np.save(out/(split+'-deduplicated.npy'),arrays[split][keep_arrays[split]])
    old_membership.flush()
    np.save(out/'train-added-vs6000-before.npy',arrays['train'][~old_membership])
    np.save(out/'train-added-vs6000-after.npy',arrays['train'][(~old_membership)&keep_arrays['train']])
    old_dev_membership.flush()
    np.save(out/'dev-added-vs-old-version-before.npy',arrays['dev'][~old_dev_membership])
    np.save(out/'dev-added-vs-old-version-after.npy',arrays['dev'][(~old_dev_membership)&keep_arrays['dev']])
    cover={s:{stage:matrix(cov[s][stage],out/('coverage-'+s+'-'+stage+'.csv')) for stage in ['before','after']} for s in arrays}
    with (out/'sequence-verdicts.jsonl').open('x') as f:
        for r in sorted(receipts,key=lambda r:r['sequence_id']):jsonline(f,r)
    pending=set(pilot['pending_sequence_batch']['selected_keys'])
    pending_results=[r for r in receipts if r['key'] in pending]
    previous_active={r['sequence_key'] for r in old_rows}
    rejection=Counter(reason for r in receipts if r['status']=='REJECTED' for reason in r['reasons'])
    files={p.name:dict(sha256=prep.digest(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='progress.json'}
    def deltas(old,full,keys):
        return [dict(category=k,old6000=old.get(k,0),full_before=full['before'].get(k,0),full_after=full['after'].get(k,0),
            difference_before=full['before'].get(k,0)-old.get(k,0),difference_after=full['after'].get(k,0)-old.get(k,0)) for k in keys]
    result=dict(schema='ref2dex.full-local-rigid-manifest.v1',experiment_id=prep.EXPERIMENT,run_id=run.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch='main',
        status='FULL_LOCAL_CENSUS_PREPARED_WITH_EXPLICIT_GAPS',training_allowed=False,training_updates=0,
        full_definition='all1335 legacy parents plus every39526 frozen nine-source planned mother, including all1177 cached and original6000 pending sequence batch, plus append-only remaining non-TEST local legacy parent; dense H4/K24 tick census, no6000-window cap',
        frozen_recipe_mothers=inventory['total_sequences'],total_prepared_mother_decisions=len(receipts),additional_local_mothers=additional,legacy=legacy,existing_cached_mothers=inventory['by_status'].get('REUSE',0),
        existing_6000_active_mothers=len(previous_active),
        native_recovery=prep.read(run/'recovered-inputs.json')['parity_checks'] if (run/'recovered-inputs.json').exists() else None,
        recovered_input_sequences=len(prep.read(run/'recovered-inputs.json')['rows']) if (run/'recovered-inputs.json').exists() else 0,
        source0_restored_sequences=prep.read(run/'recovered-inputs.json').get('restored_source0',0) if (run/'recovered-inputs.json').exists() else 0,
        mother_identity=dict(canonical_original_payloads=len(canonical),duplicate_payload_uris=0,identity='source_id + original complete payload URI; relative basenames alone are not mother IDs'),
        sequence_status=dict(Counter(r['status'] for r in receipts)),
        sequence_status_by_source={str(s):dict(Counter(r['status'] for r in receipts if r['source_id']==s)) for s in prep.SOURCES},
        sequence_rejection_reasons=dict(rejection),
        pending6000=dict(total=len(pending_results),status=dict(Counter(r['status'] for r in pending_results)),
            conversion_started=sum(r['conversion_started'] for r in pending_results)),
        windows_before=dict(count),windows_after={s:count[s]-removed[s] for s in arrays},
        source_distribution=sources,geometry_distribution=shapes,phase_distribution=phases,coverage=cover,
        dedup=dict(rule='temporal candidate within24ticks of last actual retained representative OR same identity-free fixed quantization signature at H4/K24 offsets[-3,0,8,24], then full28-frame anchored joint21/query32 <=2mm, PCAextents<=5%, object rotation<=1degree, contact16 fractions<=2%, identical full validity and phase; actual representative, no chaining',
            bin_limit='Conservative fixed-bin candidate rule; near pairs across bin boundaries retained. No exhaustive tolerance clustering or canonical-CAD equivalence claim.',
            before=n,after=n-sum(removed.values()),removed=dict(removed),global_full_predicate_comparisons=comparisons,cross_split_train_removed=cross,local_workers=workers,
            temporal_near_repeat_percent_before=100*temporal_before/n if n else 0,
            temporal_near_repeat_percent_after=100*temporal_after/(n-sum(removed.values())) if n>sum(removed.values()) else 0,
            baseline_6000_temporal_percent=61.23,
            derived_index_storage=dict(cache_mib=1024,journal_mode='MEMORY',
                incomplete_index='unpublished and rebuildable; conversion receipts unchanged')),
        provenance=dict(certified_legal=0,uncertifiable=n,source_role='UNKNOWN',available_before_reset=None,
            native_candidate_mothers=dict(native_candidate_mothers),native_candidate_windows=dict(native_candidate_windows),
            label_hash_and_mtime='per-window target-label traceability only; not pre-reset availability proof',
            missing=['reset timestamp','pre-reset independent task intent source','source role attestation','pre-reset availability timestamp/hash']),
        versus_6000=dict(baseline_windows=len(old_rows),baseline_mothers=len(previous_active),
            train_mothers_before=len(np.unique(arrays['train'][:,0])),
            train_mothers_after=len(np.unique(arrays['train'][keep_arrays['train'],0])),
            source_delta=deltas(old_sources,sources['train'],[str(s) for s in prep.SOURCES]),
            geometry_delta=deltas(old_shapes,shapes['train'],prep.CLASSES),
            phase_delta=deltas(old_phases,phases['train'],prep.PHASES),
            dev_baseline_windows=len(old_dev_rows),
            dev_baseline_rows_present_before_dedup=len(old_dev_present),dev_baseline_rows_present_after_dedup=len(old_dev_kept),
            dev_baseline_rows_missing=[dict(sequence_key=k[0],slot=k[1],tick=k[2]) for k in sorted(old_dev_keys-old_dev_present)],
            dev_added_before_dedup=count['dev']-len(old_dev_present),dev_added_after_dedup=count['dev']-removed['dev']-len(old_dev_kept),
            dev_source_delta=deltas(old_dev_sources,sources['dev'],[str(s) for s in prep.SOURCES]),
            dev_geometry_delta=deltas(old_dev_shapes,shapes['dev'],prep.CLASSES),
            dev_phase_delta=deltas(old_dev_phases,phases['dev'],prep.PHASES),
            baseline_rows_present_before_dedup=len(old_present),baseline_rows_present_after_dedup=len(old_kept),
            baseline_rows_missing=[dict(sequence_key=k[0],slot=k[1],tick=k[2]) for k in sorted(old_keys-old_present)],
            train_added_before_dedup=count['train']-len(old_present),train_added_after_dedup=count['train']-removed['train']-len(old_kept),
            dev_before_and_after=[count['dev'],count['dev']-removed['dev']],
            note='Old6000 and fixed512 anchor are unchanged; full sequence IDs refer to frozen39526 plan order. Join uses stable sequence_key,slot,tick.'),
        randomness=dict(manifest='deterministic original-plan order; no RNG consumed',dedup='deterministic representative order; no RNG consumed',shadow_surface_sampling=20261019,geometry='frozen registry; no resampling'),
        timing=dict(status='PASS',checked_admitted_windows=n,rule='H4/K24 cannot cross a non30Hz edge or declared source segment boundary',source0_full_clock_audit='../source0-clock-audit.json'),
        seeds=inventory['seeds'],inventory='../inventory.json',receipt_directory='../receipts',batch_receipt_directory='../batches',
        command=sys.argv,files=files,elapsed_s=round(time.monotonic()-started,3),
        disk_free_after_bytes=prep.shutil.disk_usage(run).free,historical_outputs_deleted=0)
    prep.write(out/'manifest.json',result)
    (out/'manifest.sha256').write_text(prep.digest(out/'manifest.json')+'  manifest.json\n')
    prep.write(out/'progress.json',dict(status='COMPLETED',windows_before=result['windows_before'],windows_after=result['windows_after'],
        manifest_sha256=prep.digest(out/'manifest.json'),disk_free_bytes=prep.shutil.disk_usage(run).free))
    print(json.dumps(dict(status='COMPLETED',windows_before=result['windows_before'],windows_after=result['windows_after'],
        legacy=legacy,sequence_status=result['sequence_status'],manifest_sha256=prep.digest(out/'manifest.json'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if not 1<=a.workers<=8:p.error('1..8 bounded CPU workers')
    summarize(a.run.resolve(),a.output.resolve(),a.workers)
