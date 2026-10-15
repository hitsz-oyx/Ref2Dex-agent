"""Turn user-authorized conservative motion caps into a frozen window exclusion view."""
import argparse,gzip,json,sys,subprocess,time,shutil
from pathlib import Path
from collections import Counter
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v3.review_data import Catalog,NAMES
from pose_mesh_data.motion_diagnostics import transition_diagnostics


CAPS={'object_speed_mps':5.,'hand_speed_mps':5.,'angular_speed_radps':40.}


def threshold_candidate(metrics):
    # Full native observation transitions can partly enter an interpolated window
    # while their second native endpoint is outside retained-window coverage.
    return any(metrics[name].get('full_capture',{}).get('maximum',{}).get('value',0.)>cap
               for name,cap in CAPS.items())


def require_unfiltered_base(recipe):
    # Catalog global offsets and descriptor identity must share the same base.
    # This entry rebuilds exclusions; it cannot compose an already-filtered view.
    if recipe.get('quality_admission'):
        raise ValueError('rebuild from the original unfiltered quality base; do not overwrite existing exclusions')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('recipe','audit','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();started=time.monotonic()
    if out.exists() or ROOT/'outputs'/TASK.name not in out.parents:raise ValueError('fresh Task output required')
    if json.loads((a.audit/'progress.json').read_text())['status']!='COMPLETED':raise ValueError('complete scan required')
    c=Catalog(a.recipe);r=json.loads((a.recipe/'recipe.json').read_text());stats=json.loads((a.recipe/'horizon-stats.json').read_text())
    require_unfiltered_base(r)
    candidates=set()
    with gzip.open(a.audit/'records.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line)
            if threshold_candidate(row['metrics']):candidates.add((row['source_id'],row['split'],row['row']))
    candidates.add((10,'train',37130))
    main_offsets={split:np.cumsum([0]+[len(c.indices[sid,split]) for sid in (0,1,2)]) for split in ('train','dev')}
    banned={s:[] for s in ('train','dev')};events=[];removed=Counter()
    for sid,split,row in sorted(candidates):
        record=next(rec for rec in c.select(sid,split) if rec.row==row);seq=c.load(record)
        bad=np.zeros(seq.frames,bool);reasons={}
        for name,values,valid,start,end in transition_diagnostics(seq.data,seq.queries):
            cap={'object_speed_mps':5.,'hand_speed_mps':5.,'angular_speed_radps':40.}.get(name)
            if cap is None:continue
            hit=valid&(values>cap);hit=hit.reshape(len(start),-1).any(1)
            if hit.any():
                reasons[name]=end[hit].tolist()
                for lo,hi in zip(start[hit],end[hit]):bad[lo:hi+1]=True
        if sid==10 and row==37130:
            if record.sequence!='ZY20210800003/H3/C1/N30/S83/s01/T2':raise ValueError('manual row identity changed; remap by source sequence')
            bad[:31]=True;reasons['user_confirmed_bad_initial_hand_fit']=[0,30]
        if not bad.any():continue
        if sid==4:bad[np.flatnonzero(bad)[0]:]=True
        rows=c.indices[(sid,split) if sid<3 else ('native',split)]
        positions=np.flatnonzero(rows[:,0]==row);ticks=rows[positions,2 if sid<3 else 1]
        cumulative=np.r_[0,np.cumsum(bad)]
        hit=cumulative[ticks+25]-cumulative[ticks-3]>0
        offset=main_offsets[split][sid] if sid<3 else main_offsets[split][-1]
        ids=offset+positions[hit];banned[split].extend(ids.tolist());removed[str(sid)+':'+split]+=len(ids)
        events.append(dict(source_id=sid,split=split,row=row,sequence=record.sequence,reasons=reasons,
            removed_windows=len(ids),mode='tail' if sid==4 else 'overlapping H4+K24',
            first_suspect_frame=int(np.flatnonzero(bad)[0]),last_suspect_frame=int(np.flatnonzero(bad)[-1])))
        if time.monotonic()-started>600:raise TimeoutError('strict admission10min deadline')
    if shutil.disk_usage(out.parent).free<20*2**30:raise RuntimeError('20GiB floor')
    out.mkdir();np.savez_compressed(out/'excluded-windows.npz',**{s:np.unique(v).astype(np.int64) for s,v in banned.items()})
    stamp=(out/'excluded-windows.npz').stat()
    descriptor=dict(schema='ref2dex.strict-motion-window-admission.v1',index_path=str(out/'excluded-windows.npz'),
        index_stamp=[stamp.st_size,stamp.st_mtime_ns],base_native_index_stamp=r['native_index_stamp'],base_main3_identity=r['main3_identity'],
        excluded_windows={s:len(np.unique(v)) for s,v in banned.items()},removed_by_source_split=dict(removed),
        user_authorization='Ambiguous samples may be excluded; tighten thresholds, data is abundant.',
        policy=dict(object_speed_mps=5.,hand_landmark_speed_mps=5.,angular_speed_radps=40.,
            simulator='quarantine tail from first flag',other='exclude every H4+K24 window overlapping flagged observed transition',
            clock='actual recorded frame intervals; HOI4D15Hz, no missing/segment bridging',
            hand_hoi4d_37130='user-confirmed initial span0..30'))
    r.update(quality_admission=descriptor,source_counts={s:n-removed.get(s+':train',0) for s,n in r['source_counts'].items()})
    stats['recipe']=r
    for name,value in [('recipe.json',r),('horizon-stats.json',stats),('quality-events.json',events),
        ('manifest.json',dict(status='COMPLETED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            parent_recipe=str(a.recipe.resolve()),audit=str(a.audit.resolve()),elapsed_s=time.monotonic()-started,**descriptor))]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETED',events=len(events),removed=dict(removed),counts=r['source_counts'])),flush=True)


if __name__=='__main__':main()
