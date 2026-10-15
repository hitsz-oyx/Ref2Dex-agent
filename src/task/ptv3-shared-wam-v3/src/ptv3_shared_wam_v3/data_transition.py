"""Allow an explicit user-reviewed window-subset resume with frozen normalization."""
import json
from pathlib import Path
import numpy as np


def validate_subset_resume(parent,config,stats,world):
    if parent['config']!=config or parent['provenance']['world_size']!=world:
        raise ValueError('subset resume requires unchanged config/world size')
    old=parent['stats'];before=old['recipe'];after=stats['recipe']
    excluded={'recipe','normalization_reference'}
    if {k:v for k,v in old.items() if k not in excluded}!={k:v for k,v in stats.items() if k not in excluded}:
        raise ValueError('subset resume requires identical frozen normalization/statistics')
    for key in ('main3_root','main3_identity','source_probabilities'):
        if before[key]!=after[key]:raise ValueError('subset resume changed '+key)
    if not after.get('training_allowed'):raise ValueError('new recipe is not admitted')
    oldroot=Path(before['native_root']);newroot=Path(after['native_root'])
    if oldroot==newroot:raise ValueError('subset resume requires a new frozen view')
    metas=[];arrays=[]
    for root,recipe in ((oldroot,before),(newroot,after)):
        meta=json.loads((root/'manifest.json').read_text());p=root/meta['window_index'];s=p.stat()
        if ([s.st_size,s.st_mtime_ns]!=recipe['native_index_stamp'] or
                [s.st_size,s.st_mtime_ns]!=[meta['window_index_size_bytes'],meta['window_index_mtime_ns']]):
            raise ValueError('frozen index changed during subset transition')
        metas.append(meta)
        with np.load(p,allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in z.files})
    oldmeta,newmeta=metas
    identities=('source_id','source_sequence','split','objects','target_slot','profile','path','payload_size_bytes','payload_mtime_ns')
    if len(oldmeta['sequences'])!=len(newmeta['sequences']):raise ValueError('sequence identity changed')
    for a,b in zip(oldmeta['sequences'],newmeta['sequences']):
        if any(a.get(k)!=b.get(k) for k in identities):raise ValueError('payload/sequence identity changed')
        p=newroot/b['path'];s=p.stat()
        if [s.st_size,s.st_mtime_ns]!=[b['payload_size_bytes'],b['payload_mtime_ns']]:raise ValueError('payload mutated')
    if oldmeta['objects']!=newmeta['objects']:raise ValueError('geometry identity changed')
    decisions=newmeta.get('motion_admission',{}).get('manual_decisions',{})
    if (decisions.get('schema')!='ref2dex.manual-motion-decisions.v1' or
            decisions.get('decision_source')!='explicit_user_visual_review' or not decisions.get('rows')):
        raise ValueError('explicit user-confirmed manual quarantine required')
    onsets={}
    for d in decisions['rows']:
        row=d['row'];r=newmeta['sequences'][row]
        if (row in onsets or r['source_id']!=4 or r['split']!='train' or d['verdict']!='exclude_tail'
                or d['source_sequence']!=r['source_sequence']):raise ValueError('invalid reviewed simulator row')
        onsets[row]=d['first_suspect_frame']
    if set(arrays[0])!=set(arrays[1]):raise ValueError('window index schema changed')
    removed={}
    for key,index in arrays[0].items():
        bad=np.array([int(row) in onsets and int(tick)+24>=onsets[int(row)] for row,tick in index],dtype=bool)
        if not np.array_equal(index[~bad],arrays[1][key]):raise ValueError('new windows differ from reviewed strict subset')
        removed[key]=int(bad.sum())
    if not removed.get('verified_train',0) or removed.get('verified_dev',0):raise ValueError('TRAIN-only strict subset required')
    if after['source_counts']['4']!=before['source_counts']['4']-removed['verified_train']:
        raise ValueError('source4 count mismatch')
    if any(after['source_counts'][s]!=n for s,n in before['source_counts'].items() if s!='4'):
        raise ValueError('other source counts changed')
    return dict(schema='ref2dex.reviewed-subset-resume.v1',parent_native=str(oldroot),native=str(newroot),
        reviewed_rows=list(onsets),removed_windows=removed,normalization='identical numerical parent TRAIN reference',
        optimizer_rng='full parent state retained',main3_unchanged=True,dev_unchanged=True)
