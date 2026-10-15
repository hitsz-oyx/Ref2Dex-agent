"""Replace pilot DexYCB with full s1 TRAIN/DEV; preserve all prior nine-source bans."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v3.data import NineWindows,collate
from ptv3_shared_wam_v3.paths import STATS_SCHEMA
from pose_mesh_data.cleaning import canonical_sequence
from pose_mesh_data.statistics import Moments,sequence_moments,subtract_moments,mix_moments
from pose_mesh_data.native_recipe import source_probabilities,recipe_for_snapshot


def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    part=p.with_suffix('.json.part');part.write_text(json.dumps(value,indent=2)+'\n');part.replace(p)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def remap_row_references(value,mapping):
    if isinstance(value,list):return [remap_row_references(v,mapping) for v in value]
    if not isinstance(value,dict):return value
    result={k:remap_row_references(v,mapping) for k,v in value.items()}
    if 'row' in value:
        result['parent_row']=value['row'];result['row']=mapping[value['row']]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('base-recipe','dex','native-output','recipe-output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();old=read(a.base_recipe/'recipe.json');base=Path(old['native_root']);dex=a.dex.resolve();out=a.native_output.resolve();recipe_dir=a.recipe_output.resolve()
    assert not out.exists() and not recipe_dir.exists() and shutil.disk_usage(out.parent).free>20*2**30
    started=time.monotonic();b=read(base/'manifest.json');d=read(dex/'manifest.json');assert d.get('strict_motion_admission') and d['encoder_admission']['workspace']==b['encoder_admission']['workspace']
    assert all(r['source_id']==8 and r['split'] in ('train','dev') for r in d['sequences'])
    arrays=load(base/b['window_index']);extra=load(dex/d['window_index']);excluded=load(old['quality_admission']['index_path'])
    main=read(Path(old['main3_root'])/'processed/manifest.json');main_sizes={s:sum(x['indices']['train' if s=='train' else 'val']['windows'] for x in main['sources']) for s in ('train','dev')}
    ban_pairs={};main_excluded={};removed_train=defaultdict(list)
    for split in ('train','dev'):
        idx=excluded[split];main_excluded[split]=idx[idx<main_sizes[split]]
        positions=idx[idx>=main_sizes[split]]-main_sizes[split];bad=arrays['verified_'+split][positions]
        ban_pairs[split]={tuple(map(int,x)) for x in bad}
        if split=='train':
            for row,tick in bad:
                if b['sequences'][row]['source_id']!=8:removed_train[int(row)].append(int(tick))
    old_m=read(base/'source-moments.json');new_m=read(dex/'source-moments.json');moments={s:dict(v) for s,v in old_m['source_moments'].items() if s!='8'}
    geometry={g['key']:{**g,'path':str((base/g['path']).resolve())} for g in b['objects']}
    cache={}
    def geo(key):
        if key not in cache:cache[key]=load(geometry[key]['path'])
        return cache[key]
    removed={s:dict(hand=Moments(),scene=Moments(),windows=0) for s in moments}
    for row,ticks in removed_train.items():
        r=b['sequences'][row];sid=str(r['source_id']);seq=canonical_sequence(load(base/r['path']),r)
        m=sequence_moments(seq,ticks,[geo(k) for k in r['objects']],r['target_slot'])
        for name in ('hand','scene'):removed[sid][name].combine(m[name])
        removed[sid]['windows']+=len(ticks)
    for sid,acc in removed.items():
        if acc['windows']:
            for name in ('hand','scene'):moments[sid][name]=subtract_moments(moments[sid][name],acc[name].result())
            moments[sid]['windows']-=acc['windows']
    moments['8']=new_m['source_moments']['8']
    for g in d['objects']:
        if g['key'] in geometry:
            original=geo(g['key']);replacement=load(dex/g['path'])
            if any(not np.array_equal(original[k],replacement[k]) for k in ('surface','normals','query')):raise ValueError('replacement CAD differs: '+g['key'])
        else:geometry[g['key']]={**g,'path':str((dex/g['path']).resolve())}
    rows=[];mapping={}
    for i,r in enumerate(b['sequences']):
        if r['source_id']==8:continue
        mapping[i]=len(rows);rows.append({**r,'path':str((base/r['path']).resolve()),'source_manifest_snapshot':str((base/r['source_manifest_snapshot']).resolve())})
    for row in rows:
        if 'motion_quality' in row:row['motion_quality']=remap_row_references(row['motion_quality'],mapping)
    offset=len(rows)
    rows.extend({**r,'path':str((dex/r['path']).resolve()),'source_manifest_snapshot':str((dex/r['source_manifest_snapshot']).resolve())} for r in d['sequences'])
    combined={}
    for key,index in arrays.items():
        split=key.rsplit('_',1)[1]
        kept=[(mapping[int(row)],int(tick)) for row,tick in index if int(row) in mapping and (int(row),int(tick)) not in ban_pairs[split]]
        add=extra[key].copy();add[:,0]+=offset
        combined[key]=np.concatenate([np.asarray(kept,np.int64).reshape(-1,2),add])
    train_counts=Counter(str(rows[int(row)]['source_id']) for row,tick in combined['verified_train'])
    for sid in moments:
        assert train_counts[sid]==moments[sid]['windows'],sid
        moments[sid]['sequences']=len({int(row) for row,tick in combined['verified_train'] if str(rows[int(row)]['source_id'])==sid})
        if sid!='8':assert train_counts[sid]==old['source_counts'][sid]
    counts=Counter(int(row) for key in ('all_train','all_dev') for row,t in combined[key])
    for i,r in enumerate(rows):r['clean_windows']=counts[i]
    out.mkdir(parents=True);np.savez_compressed(out/'windows.npz',**combined);stamp=(out/'windows.npz').stat()
    used={k for r in rows for k in r['objects']}
    meta={**b,'run_id':out.name,'sequences':rows,'objects':[g for k,g in geometry.items() if k in used],'window_index':'windows.npz','window_index_size_bytes':stamp.st_size,'window_index_mtime_ns':stamp.st_mtime_ns,'input_snapshots':[str(base/'manifest.json'),str(dex/'manifest.json')],'parent_strict_recipe':str(a.base_recipe.resolve()),'materialized_quality_admission':'Prior strict native exclusions preserved by original (row,tick); pilot DexYCB replaced; all source8 TRAIN/DEV already strict-filtered.'}
    if b.get('motion_admission'):meta['motion_admission']=remap_row_references(b['motion_admission'],mapping)
    write(out/'source-moments.json',{**old_m,'input_manifest':str(out/'manifest.json'),'windows':sum(train_counts.values()),'source_moments':moments,'derivation':'Subtract every old strict native TRAIN exclusion by exact contribution, replace source8 with full exact TRAIN-only moments.'})
    write(out/'manifest.json',meta);write(out/'progress.json',dict(status='COMPLETED',elapsed_s=time.monotonic()-started));write(out/'cleaning-summary.json',dict(status='COMPLETED',windows={k:len(v) for k,v in combined.items()},source_sequences=dict(Counter(str(r['source_id'])+':'+r['split'] for r in rows))))
    native_recipe,_=recipe_for_snapshot(out,old['native_engineering_recipe']['policy'])
    recipe_dir.mkdir(parents=True);np.savez_compressed(recipe_dir/'excluded-windows.npz',**main_excluded);st=(recipe_dir/'excluded-windows.npz').stat()
    quality={**old['quality_admission'],'index_path':str(recipe_dir/'excluded-windows.npz'),'index_stamp':[st.st_size,st.st_mtime_ns],'base_native_index_stamp':[stamp.st_size,stamp.st_mtime_ns],'excluded_windows':{s:len(v) for s,v in main_excluded.items()},'removed_by_source_split':{k:v for k,v in old['quality_admission']['removed_by_source_split'].items() if int(k.split(':')[0])<3},'native_exclusions':'materialized in native snapshot'}
    policy=dict(quality['policy'])
    policy.pop('hand_hoi4d_37130',None)
    policy['hand_hoi4d']=dict(parent_row=37130,row=mapping[37130],excluded_initial_span=[0,30])
    quality['policy']=policy
    effective={s:n for s,n in old['source_counts'].items() if int(s)<3};effective.update(dict(train_counts));weights=source_probabilities(effective,.5,{'8':.01})
    recipe={**old,'source_counts':effective,'source_probabilities':weights,'native_root':str(out),'native_index_stamp':[stamp.st_size,stamp.st_mtime_ns],'native_engineering_recipe':native_recipe,'quality_admission':quality,'sampling':dict(source='sqrt(effective TRAIN windows), DexYCB<=1%',main3='existing actual motion strata after strict exclusions',native='uniform permitted frozen windows')}
    admission=dict(old['admission'],native_snapshot_unchanged=False,user_request='2026-10-15: complete DexYCB subjects',parent_recipe=str(a.base_recipe.resolve()))
    if admission.get('motion_admission'):admission['motion_admission']=remap_row_references(admission['motion_admission'],mapping)
    recipe['admission']=admission
    data=NineWindows(recipe['main3_root'],out,'train',recipe);selected={}
    for source in data.main.base.sources:source.sequence=lru_cache(maxsize=32)(source.sequence)
    for sid in ('0','1','2'):
        hand,scene=Moments(),Moments();ids=data.draw_source(sid,4096,np.random.default_rng(np.random.SeedSequence([19,int(sid)])));selected[sid]=ids
        for idx in ids:
            sample=data[int(idx)][0];hand.update(sample['trajectory_target'],sample['trajectory_valid']&sample['start_hand_valid'][None]);scene.update(sample['scene_target'].transpose(1,0,2,3),sample['scene_valid'].transpose(1,0,2))
        moments[sid]=dict(windows=4096,hand=hand.result(),scene=scene.result(),estimation='actual strict motion-stratum TRAIN replacement draws')
        print(json.dumps(dict(moment_source=sid,samples=4096,elapsed_s=time.monotonic()-started)),flush=True)
        if time.monotonic()-started>1800 or (recipe_dir/'STOP').exists():raise RuntimeError('snapshot preparation deadline/STOP')
    stats=dict(schema=STATS_SCHEMA,split='train',units='m',fps=30,history=4,horizon=24,source_ids=[0,1,2,4,5,6,8,9,10],source_probabilities=weights,recipe=recipe,std_floor_m=.001,**mix_moments(moments,weights,.001),estimation='Strict main3 actual-strata4096/source TRAIN samples; native exact full retained TRAIN moments; no DEV/TEST fit.',main3_samples_per_source=4096,main3_moment_seed=19)
    np.savez(recipe_dir/'main3-train-stat-indices.npz',**selected)
    for name,value in [('recipe.json',recipe),('horizon-stats.json',stats),('source-moments.json',moments),('manifest.json',dict(status='COMPLETED',training_started=False,normalization='Refitted for expanded data; current running parent not modified; no implicit resume permitted.',elapsed_s=time.monotonic()-started,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))]:write(recipe_dir/name,value)
    print(json.dumps(dict(status='COMPLETED',counts=effective,probabilities=weights,elapsed_s=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
