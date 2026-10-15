"""Derive a strict window subset from identity-bound human motion decisions."""
import argparse,json,subprocess,sys,time,shutil
from collections import Counter
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];REPO=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from pose_mesh_data.cleaning import canonical_sequence
from pose_mesh_data.statistics import Moments,sequence_moments,subtract_moments
from clean_native_corpus import load_npz,save_npz,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','decisions','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve();started=time.monotonic()
    if out.exists() or REPO/'outputs'/TASK.name not in out.parents:raise ValueError('fresh Task output required')
    meta=json.loads((root/'manifest.json').read_text());decisions=json.loads(a.decisions.read_text())
    if (meta.get('training_allowed') is not False or decisions.get('schema')!='ref2dex.manual-motion-decisions.v1'
            or decisions.get('decision_source')!='explicit_user_visual_review' or not decisions.get('rows')):
        raise ValueError('explicit human motion decisions and engineering parent required')
    path=root/meta['window_index'];stamp=path.stat()
    if [stamp.st_size,stamp.st_mtime_ns]!=[meta['window_index_size_bytes'],meta['window_index_mtime_ns']]:
        raise ValueError('parent frozen index changed')
    onsets={}
    for d in decisions['rows']:
        row=d['row'];r=meta['sequences'][row]
        if row in onsets or r['source_id']!=4 or r['split']!='train' or d['verdict']!='exclude_tail':
            raise ValueError('unique simulator TRAIN tail decisions required')
        if any(d[k]!=r[k] for k in ('source_sequence','source_manifest','payload_size_bytes','payload_mtime_ns')):
            raise ValueError('manual decision identity mismatch')
        onset=d['first_suspect_frame']
        if type(onset) is not int or not 1<=onset<100:raise ValueError('bounded simulator onset frame required')
        onsets[row]=onset
    arrays=load_npz(path);kept={};removed={}
    for key,index in arrays.items():
        bad=np.array([int(row) in onsets and int(tick)+24>=onsets[int(row)] for row,tick in index],dtype=bool)
        kept[key]=index[~bad].copy();removed[key]=index[bad].copy()
    if len(removed['verified_dev']) or not len(removed['verified_train']):raise ValueError('TRAIN-only nonempty quarantine required')
    groups={}
    for row,tick in removed['verified_train']:groups.setdefault(int(row),[]).append(int(tick))
    accum=dict(hand=Moments(),scene=Moments(),windows=0)
    geometry={g['key']:g for g in meta['objects']}
    for row,ticks in groups.items():
        r=meta['sequences'][row];payload=root/r['path'];s=payload.stat()
        if [s.st_size,s.st_mtime_ns]!=[r['payload_size_bytes'],r['payload_mtime_ns']]:raise ValueError('payload changed')
        geos=[load_npz(root/geometry[k]['path']) for k in r['objects']]
        m=sequence_moments(canonical_sequence(load_npz(payload),r),ticks,geos,r['target_slot'])
        for name in ('hand','scene'):accum[name].combine(m[name])
        accum['windows']+=m['windows']
    stats=json.loads((root/'source-moments.json').read_text())
    if stats['input_manifest']!=str(root/'manifest.json'):raise ValueError('parent moments mismatch')
    before=stats['source_moments']['4'];after={**before,'windows':before['windows']-accum['windows']}
    for name in ('hand','scene'):after[name]=subtract_moments(before[name],accum[name].result())
    after['sequences']=len({int(row) for row,tick in kept['verified_train'] if meta['sequences'][row]['source_id']==4})
    stats['source_moments']['4']=after
    if time.monotonic()-started>300 or shutil.disk_usage(out.parent).free<20*2**30+200_000_000:raise RuntimeError('manual view deadline/disk boundary')
    out.mkdir(parents=True);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    save_npz(out/'windows.npz',kept);save_npz(out/'excluded-parent-windows.npz',removed)
    counts=Counter(int(row) for key in ('all_train','all_dev') for row,tick in kept[key]);rows=[]
    manual={d['row']:d for d in decisions['rows']}
    for i,r in enumerate(meta['sequences']):
        row={**r,'path':str((root/r['path']).resolve()),'source_manifest_snapshot':str((root/r['source_manifest_snapshot']).resolve()),
            'parent_clean_windows':r['clean_windows'],'clean_windows':counts[i]}
        if i in onsets:
            old=row.get('motion_quality',{});onset=min(onsets[i],old.get('first_suspect_frame',100))
            row['motion_quality']={**old,'row':i,'split':r['split'],'first_suspect_frame':onset,
                'reasons':{**old.get('reasons',{}),'human_confirmed_ejection':[onsets[i]]},'manual_decision':manual[i]}
        rows.append(row)
    stamp=(out/'windows.npz').stat()
    new={**meta,'run_id':out.name,'git_commit':commit,'sequences':rows,
        'objects':[{**g,'path':str((root/g['path']).resolve())} for g in meta['objects']],
        'input_snapshots':[str(root/'manifest.json')], 'window_index_size_bytes':stamp.st_size,'window_index_mtime_ns':stamp.st_mtime_ns,
        'motion_admission':{**meta['motion_admission'],'manual_decisions':decisions},
        'limitations':meta['limitations']+['Additional two user-confirmed ejection tails; unchanged automatic thresholds.']}
    stats.update(input_manifest=str(out/'manifest.json'),parent_moments=str(root/'source-moments.json'),
        windows=len(kept['verified_train']),git_commit=commit,derivation='Exact subtraction of manually quarantined TRAIN windows.')
    summary=dict(status='COMPLETED',parent=str(root),windows={k:len(v) for k,v in kept.items()},
        excluded={k:len(v) for k,v in removed.items()},manual_rows=list(onsets),
        retained_manual_ticks={str(i):kept['verified_train'][kept['verified_train'][:,0]==i,1].tolist() for i in onsets},
        retained_source4_train_windows=after['windows'])
    for name,value in [('source-moments.json',stats),('manual-quality-decisions.json',decisions),('cleaning-summary.json',summary),
        ('progress.json',dict(status='COMPLETED',git_commit=commit,input_manifest=str(root/'manifest.json'),elapsed_s=time.monotonic()-started,training_allowed=False))]:
        write_json(out/name,value)
    write_json(out/'manifest.json',new);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
