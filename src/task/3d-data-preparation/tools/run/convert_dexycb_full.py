"""Convert full DexYCB native labels with official subject-disjoint s1 lineage."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import yaml
from convert_hot3d_pilot import geometry,SCHEMA,validate_sequence
from fetch_dexycb_labels import CAMERA,REVISION,MIRROR,atomic

CLASSES=['002_master_chef_can','003_cracker_box','004_sugar_box','005_tomato_soup_can','006_mustard_bottle','007_tuna_fish_can','008_pudding_box','009_gelatin_box','010_potted_meat_can','011_banana','019_pitcher_base','021_bleach_cleanser','024_bowl','025_mug','035_power_drill','036_wood_block','037_scissors','040_large_marker','051_large_clamp','052_extra_large_clamp','061_foam_brick']
JOINTS=[0,1,5,9,13,17,4,8,12,16,20]


def split_for_subject(subject):
    if subject in (1,2,3,4,5,6,10):return 'train'
    if subject==7:return 'dev'
    if subject in (8,9):return 'test'
    raise ValueError('unknown official subject')


def decode(folder):
    m=yaml.safe_load((folder/'meta.yml').read_text());n=int(m['num_frames'])
    if CAMERA not in m['serials'] or not 28<=n<=1000:raise ValueError('fixed camera/frame count mismatch')
    assert len(m['mano_sides'])==1 and m['mano_sides'][0] in ('right','left')
    hand=np.full((n,2,11,3),np.nan,np.float32);hv=np.zeros(hand.shape[:-1],bool)
    poses=np.full((n,len(m['ycb_ids']),4,4),np.nan,np.float32);known=np.zeros(poses.shape[:2],bool)
    side=0 if m['mano_sides'][0]=='right' else 1;missing=0
    for frame in range(n):
        file=folder/CAMERA/f'labels_{frame:06d}.npz'
        if not file.exists():missing+=1;continue
        with np.load(file,allow_pickle=False) as z:points=z['joint_3d'][0];params=z['pose_m'][0];raw=z['pose_y']
        if points.shape!=(21,3) or params.shape!=(51,) or raw.shape!=(len(m['ycb_ids']),3,4):raise ValueError('native array shape mismatch')
        valid=np.isfinite(points).all(-1)&~(points==-1).all(-1)
        valid &= np.isfinite(params).all() and not (params==-1).all() and not (params==0).all() and not (points==0).all()
        hand[frame,side]=points[JOINTS];hv[frame,side]=valid[JOINTS]
        for slot,value in enumerate(raw):
            rot=value[:3,:3]
            if np.isfinite(value).all() and np.allclose(rot.T@rot,np.eye(3),atol=2e-4) and np.isclose(np.linalg.det(rot),1,atol=2e-4):
                poses[frame,slot]=np.eye(4);poses[frame,slot,:3,:]=value;known[frame,slot]=True
    sequence=dict(time=np.arange(n)/30,hand=hand,hand_valid=hv,object_poses=poses,object_pose_valid=known,
                  segment=np.zeros(n,np.int64),native_frame_id=np.arange(n,dtype=np.int64))
    validate_sequence(sequence)
    return m,sequence,missing


def compare_existing(raw,existing):
    """Compare all overlapping native arrays, before admitting a community repack."""
    original=existing/'subject-10/20201022-subject-10';copy=raw/'20201022-subject-10';files=0
    folders=sorted(p for p in original.iterdir() if (p/'meta.yml').exists());assert len(folders)==100
    for folder in folders:
        other=copy/folder.name
        a=yaml.safe_load((folder/'meta.yml').read_text());b=yaml.safe_load((other/'meta.yml').read_text())
        for k in ('num_frames','ycb_ids','ycb_grasp_ind','mano_sides','serials'):assert a[k]==b[k],(folder.name,k)
        for f in range(a['num_frames']):
            p=folder/CAMERA/f'labels_{f:06d}.npz';q=other/CAMERA/p.name
            assert p.exists()==q.exists(),str(p)
            if p.exists():
                with np.load(p,allow_pickle=False) as x,np.load(q,allow_pickle=False) as y:
                    for key in ('joint_3d','pose_m','pose_y'):assert np.array_equal(x[key],y[key],equal_nan=True),(folder.name,f,key)
                files+=1
    return dict(status='EXACT_NATIVE_OVERLAP_PASS',sessions=100,label_files=files,arrays_per_file=3,source_revision=REVISION)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--existing-raw',type=Path,required=True)
    p.add_argument('--existing-posemesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare-only',action='store_true');a=p.parse_args()
    audit=compare_existing(a.raw,a.existing_raw)
    if a.compare_only:print(json.dumps(audit),flush=True);return
    out=a.output.resolve();assert not out.exists();out.mkdir(parents=True);(out/'sequences').mkdir();(out/'geometry').mkdir()
    started=time.monotonic();atomic(out/'mirror-overlap-audit.json',audit)
    old=json.loads((a.existing_posemesh/'manifest.json').read_text());objects={x['key']:{**x,'path':str((a.existing_posemesh/x['path']).resolve())} for x in old['objects']}
    rows=[];heldout=[];records=[]
    for folder in sorted(a.raw.glob('*-subject-*/*')):
        if not (folder/'meta.yml').is_file():continue
        if (out/'STOP').exists() or time.monotonic()-started>1800 or shutil.disk_usage(out).free<20*2**30:raise RuntimeError('conversion STOP/deadline/free disk')
        subject=int(folder.parent.name.rsplit('-',1)[1]);split=split_for_subject(subject)
        m,data,missing=decode(folder);name=folder.name
        np.savez_compressed(out/'sequences'/(name+'.npz'),**data)
        keys=[]
        for oid in m['ycb_ids']:
            key='DexYCB/'+str(oid);keys.append(key)
            if key not in objects:
                g=geometry(a.existing_raw/'models'/CLASSES[oid-1]/'textured_simple.obj')
                dest=out/'geometry'/(str(oid)+'.npz');np.savez_compressed(dest,**g);objects[key]=dict(key=key,path=str(dest))
        row=dict(path='sequences/'+name+'.npz',format='canonical',objects=keys,target_slot=int(m['ycb_grasp_ind']),target_selection='native ycb_grasp_ind',source_id=8,profile='dexycb-native-joint3d11',split=split,group_id=f'DexYCB/subject{subject:02d}',source_sequence=name,reference_frame='fixed camera '+CAMERA,source_clock='native frame IDs; original30Hz; no temporal subsampling',mirror_revision=REVISION)
        (heldout if split=='test' else rows).append(row)
        records.append(dict(subject=subject,sequence=name,split=split,frames=len(data['time']),missing_labels=missing,valid_hand_frames=int(data['hand_valid'].all(-1).any(-1).sum())))
        if len(records)%100==0:print(json.dumps(dict(converted=len(records),elapsed_s=time.monotonic()-started)),flush=True)
    counts=Counter(r['subject'] for r in records);assert counts=={i:100 for i in range(1,11)},dict(counts)
    common=dict(schema=SCHEMA,units='m',fps=30,source_fps_verified=True,training_allowed=False,objects=list(objects.values()),source_root=str(a.raw.resolve()),source_revision=REVISION,upstream='https://dex-ycb.github.io/',mirror=MIRROR,split_policy='official s1: train01..06,10; dev07; heldout test08,09',limitations=['Community image-free repack; entire subject10 native-label overlap exactly checked against existing official download.','One fixed camera, no view duplication; missing/absent hand stays masked; no MANO reconstruction.'])
    atomic(out/'manifest.json',dict(**common,sequences=rows))
    atomic(out/'heldout-test-manifest.json',dict(**common,sequences=heldout))
    atomic(out/'conversion-audit.json',dict(status='COMPLETE',sequences=len(records),frames=sum(r['frames'] for r in records),split_sequences=dict(Counter(r['split'] for r in records)),records=records,mirror_audit=audit,elapsed_s=time.monotonic()-started))
    # The existing cleaner consumes a source-index wrapper with explicit lineage.
    atomic(out/'clean-input.json',dict(**common,sequences=[{**r,'path':str(out/r['path'])} for r in rows],source_manifests=[str(out)]))
    print(json.dumps(dict(status='COMPLETED',sequences=len(records),split_counts=dict(Counter(r['split'] for r in records)))),flush=True)


if __name__=='__main__':main()
