"""Publish full TRAIN/DEV manifests, conservative content dedup and zero cells.

This is preparation only. Missing pre-reset evidence remains UNKNOWN/null in
every admitted window; a filesystem timestamp is never a causal certificate.
"""
import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_full_rigid_data as prep
from ptv3_shared_wam_v6.rigid_dataset import duplicate_of, pca_shape


def jsonline(stream,value):
    stream.write(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')


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


def signature(s,c,fractions,geo,shape,slot,tick,phase):
    """Identity-free fixed-bin candidate key, followed by full exact checking.

    Quantization makes this conservative: nearby content on opposite bin
    boundaries is retained. No claim of exhaustive tolerance clustering.
    """
    offsets=np.array([-3,0,8,24]);hand,query,rel=anchored(s,geo,slot,tick,offsets)
    h=hashlib.blake2b(digest_size=20)
    for array in [np.rint(np.log(np.maximum(shape,1e-9))/np.log(1.05)).astype('<i4'),
                  np.rint(hand/.002).astype('<i4'), np.rint(query/.002).astype('<i4'),
                  np.rint(rel[:,:3,:3]/.01).astype('<i4'),
                  np.rint(fractions[tick+offsets,slot]/.02).astype('<i4'),
                  np.packbits(s['joint_valid'][tick-3:tick+25]),
                  np.packbits(c['valid'][tick-3:tick+25,slot]),
                  np.packbits(s['object_pose_valid'][tick-3:tick+25,slot]),
                  np.asarray([phase],np.uint8)]:
        h.update(array.tobytes())
    return h.digest()


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


def matrix(counter,path):
    rows=[dict(phase=p,geometry_class=g,source_id=s,windows=counter[(p,g,s)])
          for p in prep.PHASES for g in prep.CLASSES for s in prep.SOURCES]
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['phase','geometry_class','source_id','windows'])
        writer.writeheader();writer.writerows(rows)
    return dict(zero_requested_cells=sum(r['windows']==0 for r in rows if r['phase']!='unknown' and r['geometry_class']!='unknown'),
        requested_cells=108,zero_cells=[r for r in rows if r['windows']==0 and r['phase']!='unknown' and r['geometry_class']!='unknown'])


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
                official_split=official,
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
                    folder=Path(row['view_manifest']).parent if name=='cm_decoder_v2' else path.parent/row['path']
                    required=['joints.npy','joint_valid.npy','surface_offsets.npy','surface_valid.npy','surface_parts.npy','object_poses.npy','object_pose_valid.npy']
                    missing=[p for p in required if not (folder/p).is_file()]
                    reasons=[]
                    if any(k in missing for k in ['joints.npy','joint_valid.npy']):reasons.append('native_robot_hand21_semantics_or_validity_missing')
                    if any(k.startswith('surface') for k in missing):reasons.append('native_hand_surface778_part16_adapter_missing')
                    if 'object_pose_valid.npy' in missing:reasons.append('annotation_pose_validity_missing')
                    if not (folder/'contact/meta.json').is_file():reasons.append('bound_v6_contact_part_valid_missing')
                    if not (folder/'events/meta.json').is_file():reasons.append('bound_v6_event_missing')
                    if not reasons:raise ValueError('legacy gained a semantic adapter: explicit full admission required')
                    windows=max(0,int(row['frame_count'])-27);candidate+=windows
                    views[name]+=1;view_reasons.update(reasons)
                    jsonline(f,dict(root=name,parent_sequence_id=parent,split=split,path=str(folder),
                        status='REJECTED',reasons=reasons,missing_files=missing,candidate_windows=windows,admitted_windows=0,
                        no_human_parent_labels_substituted=True))
    return dict(unique_mothers=len(details),mother_status=dict(counts),views_by_root=dict(views),
        view_candidate_windows=candidate,view_admitted=0,view_rejection_reasons=dict(view_reasons),
        note='1335 human parent mothers; robot-derived views do not inherit human labels or add mother count')


def summarize(run,out):
    progress=prep.read(run/'progress.json')
    if progress['status']!='CONVERSION_AND_ADMISSION_COMPLETED':raise ValueError('complete sequence census required')
    if out.exists():raise FileExistsError('fresh summary attempt required; preserve prior artifacts')
    prep.guard(run,256*1024**2);out.mkdir()
    prep._GEOMETRY_ROWS={r['key']:r for r in prep.read(prep.REGISTRY/'manifest.json')['objects']}
    receipts=[prep.read(p) for p in sorted((run/'receipts').glob('*.json'))]
    inventory=prep.read(run/'inventory.json')
    if len(receipts)!=inventory['total_sequences']:raise ValueError('incomplete receipts')
    by_id={r['sequence_id']:r for r in receipts}
    legacy=legacy_census(out,receipts)
    pilot=prep.read(prep.PILOT);old_rows=pilot['selection']['train_window_rows']
    old_keys={(r['sequence_key'],int(r['slot']),int(r['tick'])) for r in old_rows}
    old_present=set();old_kept=set()
    old_coverage=Counter()
    for r in old_rows:
        _,geo=prep.geometry(r['object'],r['source_id'])
        old_coverage[(r['phase'],prep.geometry_class(pca_shape(geo['surface'])),r['source_id'])]+=1
    matrix(old_coverage,out/'coverage-old-6000.csv')
    # DEV representatives take priority. Remove matching TRAIN content and
    # audit each cross-split deletion; no held-out row is moved into TRAIN.
    admitted=sorted([r for r in receipts if r['status']=='ADMITTED'],key=lambda r:(r['split']!='dev',r['sequence_id']))
    n=sum(r['admitted_windows'] for r in admitted)
    prep.guard(run,n*2048+256*1024**2)
    arrays={s:np.lib.format.open_memmap(out/(s+'-all.npy'),mode='w+',dtype=np.int32,
            shape=(sum(r['admitted_windows'] for r in admitted if r['split']==s),3)) for s in ['train','dev']}
    keep_arrays={s:np.lib.format.open_memmap(out/(s+'-keep-mask.npy'),mode='w+',dtype=bool,shape=(len(arrays[s]),)) for s in arrays}
    db=sqlite3.connect(str(out/'dedup-index.sqlite'))
    db.execute('CREATE TABLE representatives (signature BLOB, sequence_id INTEGER, slot INTEGER, tick INTEGER, phase INTEGER)')
    db.execute('CREATE INDEX signature_index ON representatives(signature)')
    positions=Counter();count=Counter();removed=Counter();cross=0;comparisons=0;temporal_before=0;temporal_after=0
    cov={s:{t:Counter() for t in ['before','after']} for s in arrays}
    sources={s:{t:Counter() for t in ['before','after']} for s in arrays}
    shapes={s:{t:Counter() for t in ['before','after']} for s in arrays}
    phases={s:{t:Counter() for t in ['before','after']} for s in arrays}
    started=time.monotonic()
    manifest_streams={s:(out/(s+'-windows.jsonl')).open('x') for s in arrays}
    def temporal_fraction(ticks):
        if len(ticks)<2:return 0
        diff=np.diff(np.sort(ticks));near=diff<24
        return int((np.r_[False,near]|np.r_[near,False]).sum())
    with (out/'dedup-pairs.jsonl').open('x') as pairs:
        for number,r in enumerate(admitted,1):
            prep.guard(run,64*1024**2)
            if (run/'STOP').exists():raise InterruptedError('owner STOP')
            split=r['split'];s,c,fractions=sequence(r['path'],r['contact_path'])
            shard=np.load(r['window_shard'],allow_pickle=False)
            tuples=shard['tuples'];phase_ids=shard['phase'];class_ids=shard['geometry_class']
            start=positions[split];arrays[split][start:start+len(tuples)]=tuples
            keep_arrays[split][start:start+len(tuples)]=True
            retained_ticks={}; temporal_representatives={}
            for slot in np.unique(tuples[:,1]):
                temporal_before+=temporal_fraction(tuples[tuples[:,1]==slot,2]);retained_ticks[int(slot)]=[]
            for i,(_,slot,tick) in enumerate(tuples):
                slot,tick=int(slot),int(tick);phase=int(phase_ids[i]);gc=int(class_ids[i])
                _,geo=prep.geometry(r['objects'][slot],r['source_id']);shape=shape_of(geo)
                key=signature(s,c,fractions,geo,shape,slot,tick,phase)
                candidates=db.execute('SELECT sequence_id,slot,tick,phase FROM representatives WHERE signature=?',(key,)).fetchall()
                representative=None;feat=None
                previous=temporal_representatives.get(slot)
                if previous and tick-previous[0]<24 and phase==previous[1]:
                    feat=full_feature(r,slot,tick,phase,geo);comparisons+=1
                    if duplicate_of(feat,previous[2]):representative=[r['sequence_id'],slot,previous[0]]
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
                    if feat is None:feat=full_feature(r,slot,tick,phase,geo)
                    temporal_representatives[slot]=(tick,phase,feat)
                else:
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
                if split=='train' and oldkey in old_keys:
                    old_present.add(oldkey)
                    if retained:old_kept.add(oldkey)
                jsonline(manifest_streams[split],dict(sequence_id=r['sequence_id'],sequence_key=r['key'],slot=slot,tick=tick,
                    split=split,source_id=r['source_id'],phase=prep.PHASES[phase],geometry_class=prep.CLASSES[gc],
                    source_role='UNKNOWN',available_before_reset=None,certification='UNCERTIFIABLE_MISSING_PRE_RESET_RECORDS',
                    reset_timestamp_ns=None,source_available_timestamp_ns=None,source_artifact_sha256=None,
                    label_object_pose_sha256=r['label_object_pose_sha256'],label_filesystem_mtime_ns=r['label_filesystem_mtime_ns'],
                    retained_after_dedup=retained,sequence_receipt='../receipts/'+r['key']+'.json'))
            for ticks in retained_ticks.values():temporal_after+=temporal_fraction(ticks)
            positions[split]+=len(tuples);count[split]+=len(tuples)
            db.commit()
            if number%64==0:
                status=dict(stage='full_manifest',sequences=number,total_sequences=len(admitted),windows=dict(count),removed=dict(removed),
                    elapsed_s=round(time.monotonic()-started,2),disk_free_bytes=prep.shutil.disk_usage(run).free)
                prep.write(out/'progress.json',status);print(json.dumps(status),flush=True)
    for stream in manifest_streams.values():stream.close()
    db.close()
    for split in arrays:
        arrays[split].flush();keep_arrays[split].flush()
        np.save(out/(split+'-deduplicated.npy'),arrays[split][keep_arrays[split]])
    cover={s:{stage:matrix(cov[s][stage],out/('coverage-'+s+'-'+stage+'.csv')) for stage in ['before','after']} for s in arrays}
    with (out/'sequence-verdicts.jsonl').open('x') as f:
        for r in sorted(receipts,key=lambda r:r['sequence_id']):jsonline(f,r)
    pending=set(pilot['pending_sequence_batch']['selected_keys'])
    pending_results=[r for r in receipts if r['key'] in pending]
    previous_active={r['sequence_key'] for r in old_rows}
    rejection=Counter(reason for r in receipts if r['status']=='REJECTED' for reason in r['reasons'])
    files={p.name:dict(sha256=prep.digest(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='progress.json'}
    result=dict(schema='ref2dex.full-local-rigid-manifest.v1',experiment_id=prep.EXPERIMENT,run_id=run.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch='main',
        status='FULL_LOCAL_CENSUS_PREPARED_WITH_EXPLICIT_GAPS',training_allowed=False,training_updates=0,
        full_definition='all1335 legacy parents plus every39526 frozen nine-source planned mother, including all1177 cached and original6000 pending sequence batch; dense H4/K24 tick census, no6000-window cap',
        total_recipe_mothers=len(receipts),legacy=legacy,existing_cached_mothers=inventory['by_status'].get('REUSE',0),
        existing_6000_active_mothers=len(previous_active),
        native_recovery=prep.read(run/'recovered-inputs.json')['parity_checks'] if (run/'recovered-inputs.json').exists() else None,
        native_recovered_sequences=len(prep.read(run/'recovered-inputs.json')['rows']) if (run/'recovered-inputs.json').exists() else 0,
        sequence_status=dict(Counter(r['status'] for r in receipts)),sequence_rejection_reasons=dict(rejection),
        pending6000=dict(total=len(pending_results),status=dict(Counter(r['status'] for r in pending_results)),
            conversion_started=sum(r['conversion_started'] for r in pending_results)),
        windows_before=dict(count),windows_after={s:count[s]-removed[s] for s in arrays},
        source_distribution=sources,geometry_distribution=shapes,phase_distribution=phases,coverage=cover,
        dedup=dict(rule='temporal candidate within24ticks of last actual retained representative OR same identity-free fixed quantization signature at H4/K24 offsets[-3,0,8,24], then full28-frame anchored joint21/query32 <=2mm, PCAextents<=5%, object rotation<=1degree, contact16 fractions<=2%, identical full validity and phase; actual representative, no chaining',
            bin_limit='Conservative fixed-bin candidate rule; near pairs across bin boundaries retained. No exhaustive tolerance clustering or canonical-CAD equivalence claim.',
            before=n,after=n-sum(removed.values()),removed=dict(removed),full_predicate_comparisons=comparisons,cross_split_train_removed=cross,
            temporal_near_repeat_percent_before=100*temporal_before/n if n else 0,
            temporal_near_repeat_percent_after=100*temporal_after/(n-sum(removed.values())) if n>sum(removed.values()) else 0,
            baseline_6000_temporal_percent=61.23),
        provenance=dict(certified_legal=0,uncertifiable=n,source_role='UNKNOWN',available_before_reset=None,
            label_hash_and_mtime='per-window target-label traceability only; not pre-reset availability proof',
            missing=['reset timestamp','pre-reset independent task intent source','source role attestation','pre-reset availability timestamp/hash']),
        versus_6000=dict(baseline_windows=len(old_rows),baseline_mothers=len(previous_active),
            baseline_rows_present_before_dedup=len(old_present),baseline_rows_present_after_dedup=len(old_kept),
            baseline_rows_missing=[dict(sequence_key=k[0],slot=k[1],tick=k[2]) for k in sorted(old_keys-old_present)],
            train_added_before_dedup=count['train']-len(old_present),train_added_after_dedup=count['train']-removed['train']-len(old_kept),
            dev_before_and_after=[count['dev'],count['dev']-removed['dev']],
            note='Old6000 and fixed512 anchor are unchanged; full sequence IDs refer to frozen39526 plan order. Join uses stable sequence_key,slot,tick.'),
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
    a=p.parse_args();summarize(a.run.resolve(),a.output.resolve())
