"""Catalog all prepared data, including the concurrently completed certified pilot.

Keep both original manifests immutable. Dedup uses the same physical predicate
and actual input-validity/goal content: an unknown condition and a certified
instruction are distinct inputs, never collapsed just to change a data metric.
"""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import summarize_full_rigid_data as full
p = full.prep


def same_input(a,b):
    if a['goal_valid']!=b['goal_valid']:return False
    if a['goal_valid'] and np.max(np.abs(a['goal_delta']-b['goal_delta']))>.002:return False
    return full.duplicate_of(a,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['historical','fresh','output','check']:parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();history=a.historical.resolve();fresh=a.fresh.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError('preserve existing prepared catalog')
    h=p.read(history/'manifest.json');check=p.read(a.check)
    if check.get('status')!='PASS' or check.get('rollouts')!=96 or check.get('admitted_windows')!=26304:
        raise ValueError('independently rerun fresh source/label audit required')
    if h['provenance']['certified_legal']!=0:raise ValueError('historical condition availability changed; inspect before combining')
    if p.read(history/'progress.json')['status']!='COMPLETED':raise ValueError('historical full manifest is not complete')
    p.guard(out.parent,1024*1024**2);out.mkdir()
    p._GEOMETRY_ROWS={r['key']:r for r in p.read(p.REGISTRY/'manifest.json')['objects']}
    original={r['sequence_id']:r for r in (json.loads(line) for line in (history/'sequence-verdicts.jsonl').open())}
    start_id=max(original)+1
    candidate=p.read(fresh/'candidate-manifest.json');rows=sorted(candidate['sequences'],key=lambda r:r['sequence_ordinal'])
    if [r['sequence_ordinal'] for r in rows]!=list(range(96)):raise ValueError('fresh candidate ordinal map changed')
    audits={}
    for line in (fresh/'window-verdicts.jsonl').open():
        r=json.loads(line);audits[(r['rollout_id'],r['slot'],r['tick'])]=r
    tuples=np.load(fresh/'certified-candidate-tuples.npy',allow_pickle=False)
    source_rows={};geometries={};instructions={};bindings={};new_before=[];new_after=[];new_keep=[]
    for row in rows:
        ordinal=row['sequence_ordinal'];sid=start_id+ordinal;binding=p.read(row['reset_binding']);admission=p.read(row['admission'])
        if admission['goal_certification']!='CERTIFIED_LEGAL' or not binding['available_before_reset']:
            raise ValueError('fresh record is not source certified')
        if not binding['source_available_timestamp_ns']<binding['reset_timestamp_ns']:raise ValueError('source not before reset')
        if p.digest(binding['intent_artifact'])!=binding['source_artifact_sha256']:raise ValueError('source artifact changed')
        instruction=p.read(row['goal_instruction'])
        if p.digest(row['goal_instruction'])!=binding['instruction_sha256'] or instruction['reset']!=binding['reset']:
            raise ValueError('instruction/reset changed')
        if p.digest(row['geometry'])!=row['geometry_sha256']:raise ValueError('fresh geometry changed')
        with np.load(row['geometry'],allow_pickle=False) as z:geometries[sid]={k:z[k] for k in z.files}
        instructions[sid]=instruction;bindings[sid]=binding
        folder=Path(row['sequence'])
        source_rows[sid]=dict(sequence_id=sid,key='fresh-'+fresh.name+'-'+row['rollout_id'],source_id=4,split='train',
            rollout_id=row['rollout_id'],path=str(folder),contact_path=row['contact'],event_path=row['events'],
            geometry=row['geometry'],geometry_sha256=row['geometry_sha256'],reset_binding=row['reset_binding'],
            goal_instruction=row['goal_instruction'],admission=row['admission'],
            original_candidate_ordinal=ordinal,source_role=binding['source_role'],available_before_reset=True,
            reset_timestamp_ns=binding['reset_timestamp_ns'],source_available_timestamp_ns=binding['source_available_timestamp_ns'],
            source_artifact_sha256=binding['source_artifact_sha256'],label_object_pose_sha256=p.digest(folder/'object_poses.npy'),
            label_filesystem_mtime_ns=(folder/'object_poses.npy').stat().st_mtime_ns)
    @lru_cache(maxsize=512)
    def feature(sid,slot,tick,phase):
        if sid in source_rows:
            r=source_rows[sid];g=geometries[sid]
            result=full.full_feature(r,slot,tick,phase,g)
            seq,_,_=full.sequence(r['path'],r['contact_path'])
            relative=np.linalg.inv(np.asarray(seq['object_poses'][tick,slot],np.float64))@np.asarray(instructions[sid]['task_goal_pose'])
            q=g['query'];delta=(q@relative[:3,:3].T+relative[:3,3]-q).astype(np.float32)
            result.update(goal_valid=True,goal_delta=delta)
        else:
            r=original[sid];_,g=p.geometry(r['objects'][slot],r['source_id'])
            result=full.full_feature(r,slot,tick,phase,g);result.update(goal_valid=False,goal_delta=None)
        return result
    db=sqlite3.connect('file:'+str(history/'dedup-index.sqlite')+'?mode=ro',uri=True)
    own=defaultdict(list);last={};cross_physical=0;cross_condition_merged=0
    before=Counter();after=Counter();removed=0
    with (out/'fresh-windows.jsonl').open('x') as stream,(out/'fresh-dedup-pairs.jsonl').open('x') as pairs:
        for number,(ordinal,slot,tick) in enumerate(tuples):
            ordinal,slot,tick=map(int,(ordinal,slot,tick));sid=start_id+ordinal;r=source_rows[sid]
            saved=audits[(r['rollout_id'],slot,tick)]
            if not saved['admitted'] or saved['goal_certification']!='CERTIFIED_LEGAL':raise ValueError('fresh window certification changed')
            phase=full.prep.PHASES.index(saved['phase']);f=feature(sid,slot,tick,phase)
            physical_key=full.feature_signature(f)
            # Geometry-only overlaps are reported separately. Different goal
            # validity masks cannot be merged as identical model inputs.
            cross_matches=db.execute('SELECT sequence_id,slot,tick,phase FROM representatives WHERE signature=? ORDER BY rowid',(physical_key,)).fetchall()
            for old_id,old_slot,old_tick,old_phase in cross_matches:
                old=feature(old_id,old_slot,old_tick,old_phase)
                if full.duplicate_of(f,old):
                    cross_physical+=1
                    if same_input(f,old):cross_condition_merged+=1
                    break
            condition_hash=hashlib.blake2b(np.rint(f['goal_delta']/.002).astype('<i4').tobytes(),digest_size=16).digest()
            key=physical_key+condition_hash;representative=None
            previous=last.get(sid)
            if previous and tick-previous[1]<24 and same_input(f,feature(sid,slot,previous[1],previous[2])):
                representative=[sid,slot,previous[1]]
            if representative is None:
                for rs,rt,rp in own[key]:
                    if same_input(f,feature(rs,0,rt,rp)):representative=[rs,0,rt];break
            retained=representative is None
            record=dict(sequence_id=sid,sequence_key=r['key'],slot=slot,tick=tick,source_id=4,split='train',
                phase=saved['phase'],geometry_class=saved['geometry_class'],source_role=r['source_role'],available_before_reset=True,
                reset_timestamp_ns=r['reset_timestamp_ns'],source_available_timestamp_ns=r['source_available_timestamp_ns'],
                source_artifact_sha256=r['source_artifact_sha256'],goal_certification='CERTIFIED_LEGAL',
                event_condition_certification='UNKNOWN_NO_INDEPENDENT_EVENT_INTENT',goal_instruction=r['goal_instruction'],
                label_object_pose_sha256=r['label_object_pose_sha256'],label_filesystem_mtime_ns=r['label_filesystem_mtime_ns'],retained_after_dedup=retained)
            full.jsonline(stream,record);new_before.append([sid,slot,tick]);new_keep.append(retained)
            cell=(saved['phase'],saved['geometry_class'],4);before[cell]+=1
            if retained:
                own[key].append((sid,tick,phase));last[sid]=(slot,tick,phase);new_after.append([sid,slot,tick]);after[cell]+=1
            else:
                removed+=1;full.jsonline(pairs,dict(removed=[sid,slot,tick],retained=representative))
            if (number+1)%4096==0:
                p.guard(out,64*1024**2);print(json.dumps(dict(stage='certified_partition',processed=number+1,total=len(tuples),removed=removed)),flush=True)
    db.close()
    for split in ['train','dev']:
        for field in ['all','deduplicated']:
            old=np.load(history/(split+'-'+field+'.npy'),mmap_mode='r')
            extra=np.asarray(new_before if field=='all' else new_after,np.int32).reshape(-1,3) if split=='train' else np.empty((0,3),np.int32)
            if split=='dev':(out/(split+'-'+field+'.npy')).symlink_to((history/(split+'-'+field+'.npy')).resolve())
            else:
                target=np.lib.format.open_memmap(out/(split+'-'+field+'.npy'),mode='w+',dtype=np.int32,shape=(len(old)+len(extra),3))
                target[:len(old)]=old;target[len(old):]=extra;target.flush()
    np.save(out/'fresh-all.npy',np.asarray(new_before,np.int32));np.save(out/'fresh-deduplicated.npy',np.asarray(new_after,np.int32).reshape(-1,3))
    np.save(out/'fresh-keep-mask.npy',np.asarray(new_keep,bool))
    with (out/'fresh-sequences.jsonl').open('x') as stream:
        for r in source_rows.values():full.jsonline(stream,r)
    coverage={}
    import csv
    for split in ['train','dev']:
        coverage[split]={}
        for stage in ['before','after']:
            with (history/('coverage-'+split+'-'+stage+'.csv')).open() as f:
                c=Counter({(r['phase'],r['geometry_class'],int(r['source_id'])):int(r['windows']) for r in csv.DictReader(f)})
            if split=='train':c.update(before if stage=='before' else after)
            coverage[split][stage]=full.matrix(c,out/('coverage-'+split+'-'+stage+'.csv'))
    count_before=dict(h['windows_before']);count_after=dict(h['windows_after']);count_before['train']+=len(new_before);count_after['train']+=len(new_after)
    distributions={k:json.loads(json.dumps(h[k])) for k in ['source_distribution','geometry_distribution','phase_distribution']}
    for stage,counts in [('before',before),('after',after)]:
        for (phase,gc,source),count in counts.items():
            for key,category in [('source_distribution',str(source)),('geometry_distribution',gc),('phase_distribution',phase)]:
                d=distributions[key]['train'][stage];d[category]=d.get(category,0)+count
    for stage,extra in [('before',new_before),('after',new_after)]:
        suffix='before' if stage=='before' else 'after'
        old=np.load(history/('train-added-vs6000-'+suffix+'.npy'),mmap_mode='r');arr=np.asarray(extra,np.int32).reshape(-1,3)
        target=np.lib.format.open_memmap(out/('train-added-vs6000-'+suffix+'.npy'),mode='w+',dtype=np.int32,shape=(len(old)+len(arr),3))
        target[:len(old)]=old;target[len(old):]=arr;target.flush()
    previous=np.load(history/'train-membership-in-old6000.npy',mmap_mode='r')
    membership=np.lib.format.open_memmap(out/'train-membership-in-old6000.npy',mode='w+',dtype=bool,shape=(len(previous)+len(new_before),))
    membership[:len(previous)]=previous;membership[len(previous):]=False;membership.flush()
    files={f.name:dict(sha256=p.digest(f),bytes=f.stat().st_size) for f in out.iterdir() if f.is_file()}
    catalog=dict(schema='ref2dex.full-prepared-catalog.v1',status='COMPLETED',training_allowed=False,production_recipe_modified=False,
        historical_manifest=dict(path=str(history/'manifest.json'),sha256=p.digest(history/'manifest.json')),
        fresh_candidate_manifest=dict(path=str(fresh/'candidate-manifest.json'),sha256=p.digest(fresh/'candidate-manifest.json')),
        independent_fresh_check=dict(path=str(a.check.resolve()),sha256=p.digest(a.check)),
        sequence_id_ranges=dict(historical=[0,start_id-1],fresh=[start_id,start_id+95]),mother_decisions=start_id+96,
        official_test_mothers_excluded=202,total_mother_census=start_id+96+202,
        windows_before=count_before,windows_after=count_after,
        fresh=dict(mothers=96,before=len(new_before),after=len(new_after),removed=removed,certified_goal_before=len(new_before),certified_goal_after=len(new_after)),
        provenance=dict(certified_goal_before=len(new_before),certified_goal_after=len(new_after),uncertifiable_before=h['provenance']['uncertifiable'],uncertifiable_after=sum(h['windows_after'].values()),
            source_role_counts=dict(UNKNOWN=h['provenance']['uncertifiable'],explicit_task_instruction=len(new_before)),event_condition_certified=0),
        dedup=dict(physical_rule=h['dedup']['rule'],additional_input_validity_rule='goal-validity masks match; if valid, all32 reset-goal deltas agree within2mm; never merge distinct conditioning inputs',
            cross_partition_physical_matches=cross_physical,cross_partition_condition_identical=cross_condition_merged,
            historical_removed=h['dedup']['before']-h['dedup']['after'],fresh_removed=removed),
        coverage=coverage,files=files,**distributions,
        window_manifest_parts=dict(train=[str(history/'train-windows.jsonl'),str(out/'fresh-windows.jsonl')],dev=[str(history/'dev-windows.jsonl')]),
        required_consumers=dict(historical_goal='explicit UNKNOWN_GOAL_POLICY; never legacy realized-endpoint fallback',fresh_goal='ptv3_shared_wam_v6.fresh_rigid.reset_bound_sample with exact bound instruction/reset',events='UNKNOWN_EVENT_POLICY; goal certification does not certify event intent'),
        note='read-ready data-only catalog; old6k, fixed512 panel, historical full manifest and fresh candidate artifacts are immutable; no training configuration is activated',
        seeds=dict(historical=h['seeds'],fresh_probe=list(range(198,204))),git_commit=p.subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        training_updates=0,new_simulation_controls=0,disk_free_after_bytes=p.shutil.disk_usage(out).free)
    p.write(out/'manifest.json',catalog);(out/'manifest.sha256').write_text(p.digest(out/'manifest.json')+'  manifest.json\n')
    print(json.dumps(dict(status='COMPLETED',windows_before=count_before,windows_after=count_after,fresh=catalog['fresh'],manifest_sha256=p.digest(out/'manifest.json'))),flush=True)


if __name__=='__main__':main()
