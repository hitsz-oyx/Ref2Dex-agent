"""Stream H2O native joints/poses; preserve official splits and moving-camera geometry."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import zipfile
import numpy as np
from convert_hot3d_pilot import geometry, SCHEMA, validate_sequence

NAMES=['book','espresso','lotion','spray','milk','cocoa','chips','cappuccino']
POINTS=[0,1,5,9,13,17,4,8,12,16,20]
KINDS={'hand_pose':128,'hand_pose_mano':124,'cam_pose':16,'obj_pose_rt':17,'action_label':1,'verb_label':1}


def write(path,value):
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def rigid(value):
    return (value.shape==(4,4) and np.isfinite(value).all()
        and np.allclose(value[3],[0,0,0,1],atol=1e-6)
        and np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=2e-4)
        and np.isclose(np.linalg.det(value[:3,:3]),1,atol=2e-4))


def native_split(archive):
    result={};groups={}
    with zipfile.ZipFile(archive) as z:
        for name,split in [('train','train'),('val','dev'),('test','test')]:
            for line in z.read('label_split/pose_'+name+'.txt').decode().splitlines():
                p=Path(line.strip());take='/'.join(p.parts[:3]);key=(take,int(p.stem))
                if key in result and result[key]!=split:raise ValueError('native frame split overlap')
                result[key]=split
                group='/'.join(p.parts[:2])
                if group in groups and groups[group]!=split:raise ValueError('native session split overlap')
                groups[group]=split
    return result,groups


def read_subject(archive,root,deadline):
    takes=defaultdict(lambda:defaultdict(dict));members=0
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if members%2048==0:
                if time.monotonic()>deadline or (root/'STOP').exists() or shutil.disk_usage(root).free<20*2**30:
                    raise InterruptedError('stop/deadline/disk reserve; no partial manifest published')
                write(root/'progress.json',dict(status='READING_SUBJECT',pid=os.getpid(),archive=archive.name,
                    parsed_members=members,checked_at=time.time(),training_allowed=False))
            members+=1
            if not member.isfile():continue
            p=Path(member.name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe native archive member')
            if len(p.parts)!=6 or p.parts[3]!='cam4' or p.parts[4] not in KINDS:continue
            kind=p.parts[4]
            if member.size>65536:raise ValueError('bounded numeric label exceeded64KiB')
            value=np.fromstring(tar.extractfile(member).read().decode(),sep=' ')
            if value.shape!=(KINDS[kind],):raise ValueError('unexpected native numeric label shape: '+kind)
            take='/'.join(p.parts[:3]);frame=int(p.stem)
            if frame in takes[take][kind]:raise ValueError('duplicate native frame kind')
            takes[take][kind][frame]=value
    return takes


def convert_take(native,ids):
    n=len(ids);hand=np.full((n,2,11,3),np.nan,np.float32);hv=np.zeros(hand.shape[:-1],bool)
    classes=sorted({int(native['obj_pose_rt'][i][0]) for i in ids if i in native['obj_pose_rt']} - {0})
    if len(classes)!=1 or not 1<=classes[0]<=8:raise ValueError('one stable native object identity required')
    oid=classes[0];poses=np.full((n,1,4,4),np.nan,np.float32);known=np.zeros((n,1),bool)
    cameras=np.full((n,4,4),np.nan,np.float64);camera_valid=np.zeros(n,bool)
    mano=np.full((n,2,62),np.nan,np.float32);mano_available=np.zeros((n,2),bool)
    native_joints=np.full((n,2,21,3),np.nan,np.float64)
    actions=np.full(n,-1,np.int64);verbs=actions.copy()
    for k,i in enumerate(ids):
        for kind,out in [('action_label',actions),('verb_label',verbs)]:
            if i in native[kind]:out[k]=int(native[kind][i][0])
        if i in native['hand_pose_mano']:
            raw=native['hand_pose_mano'][i].reshape(2,62)[::-1]
            mano[k]=raw;mano_available[k]=(raw[:,0]==1)&np.isfinite(raw).all(-1)
        if i in native['hand_pose']:
            raw=native['hand_pose'][i].reshape(2,64)[::-1];native_joints[k]=raw[:,1:].reshape(2,21,3)
        if i not in native['cam_pose']:continue
        camera=native['cam_pose'][i].reshape(4,4);cameras[k]=camera
        if not rigid(camera):continue
        camera_valid[k]=True
        if i in native['hand_pose']:
            raw=native['hand_pose'][i].reshape(2,64)[::-1]
            points=native_joints[k][:,POINTS]
            hand[k]=points@camera[:3,:3].T+camera[:3,3]
            hv[k]=(raw[:,0,None]==1)&np.isfinite(points).all(-1)&~(native_joints[k]==0).all(axis=(-1,-2))[:,None]
        if i in native['obj_pose_rt']:
            raw=native['obj_pose_rt'][i];obj=raw[1:].reshape(4,4)
            if int(raw[0])==oid and rigid(obj):
                poses[k,0]=camera@obj;known[k,0]=True
    seq=dict(time=np.arange(n,dtype=np.float64)/30,hand=hand,hand_valid=hv,object_poses=poses,
        object_pose_valid=known,segment=np.zeros(n,np.int64),native_frame_id=np.array(ids,np.int64),
        native_joints_camera=native_joints,native_camera_to_world=cameras,native_camera_valid=camera_valid,
        native_mano_camera=mano,native_mano_available=mano_available,native_action_label=actions,native_verb_label=verbs)
    validate_sequence(seq)
    return seq,oid


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=7200);a=p.parse_args()
    if not 60<=a.seconds<=86400:p.error('bounded deadline required')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    for name in ['sequences','geometry','scratch']:(root/name).mkdir(exist_ok=True)
    start=time.monotonic();deadline=start+a.seconds;archives=a.input/'archives'
    while not all((archives/name).exists() for name in ['object.zip','label_split.zip']):
        write(root/'progress.json',dict(status='WAITING_NATIVE_ARCHIVES',pid=os.getpid(),checked_at=time.time(),training_allowed=False))
        if time.monotonic()>deadline or (root/'STOP').exists():return
        time.sleep(10)
    splits,groups=native_split(archives/'label_split.zip')
    meta=dict(schema=SCHEMA,units='m',fps=30,source_fps_verified=False,training_allowed=False,objects=[],sequences=[],
        reference_frame='native initial camera world; per-frame cam_to_world applied; no arbitrary gravity tilt',
        limitations=['30Hz nominal; native frame IDs retained, original timestamp evidence pending.',
            'Native hand_pose joints used directly; MANO retained in camera coordinates, not robot commands.',
            'Official pose train/val used; subject4 official test isolated in heldout-test-manifest.json.',
            'Official session groups used: subject3 occurs across distinct TRAIN/DEV sessions.'])
    heldout={**meta,'objects':[],'sequences':[]};ledger={};known=set()
    if (root/'manifest.json').exists():
        meta=json.loads((root/'manifest.json').read_text());heldout=json.loads((root/'heldout-test-manifest.json').read_text())
        ledger=json.loads((root/'conversion-ledger.json').read_text());known={r['key'] for r in meta['objects']}
    def publish(status):
        write(root/'manifest.json',meta);write(root/'heldout-test-manifest.json',heldout);write(root/'conversion-ledger.json',ledger)
        write(root/'progress.json',dict(status=status,pid=os.getpid(),converted_sequences=sum(r['status']=='CONVERTED' for r in ledger.values()),
            converted_frames=sum(r.get('frames',0) for r in ledger.values()),failed_sequences=sum(r['status']=='FAILED' for r in ledger.values()),
            train_sequences=sum(r['split']=='train' for r in meta['sequences']),dev_sequences=sum(r['split']=='dev' for r in meta['sequences']),
            heldout_test_sequences=len(heldout['sequences']),elapsed_s=time.monotonic()-start,checked_at=time.time(),training_allowed=False))
    with zipfile.ZipFile(archives/'object.zip') as z:
        for oid,name in enumerate(NAMES,1):
            key='H2O/'+name
            if key in known:continue
            member=next(n for n in z.namelist() if n.startswith('object/'+name+'/') and n.endswith('.obj'))
            scratch=root/'scratch'/f'{oid}.obj';scratch.write_bytes(z.read(member))
            np.savez_compressed(root/'geometry'/f'{oid}.npz',**geometry(scratch));scratch.unlink()
            row=dict(key=key,path=f'geometry/{oid}.npz');meta['objects'].append(row);heldout['objects'].append(row);known.add(key)
    for subject in range(1,5):
        if all(take in ledger for take in {t for t,_ in splits if t.startswith(f'subject{subject}/')}):continue
        archive=archives/f'subject{subject}_pose_v1_1.tar.gz'
        while not archive.exists():
            publish('WAITING_NATIVE_ARCHIVES')
            if time.monotonic()>deadline or (root/'STOP').exists():publish('STOPPED');return
            time.sleep(10)
        try: takes=read_subject(archive,root,deadline)
        except InterruptedError:publish('STOPPED');return
        for take,native in sorted(takes.items()):
            if take in ledger:continue
            if time.monotonic()>deadline or (root/'STOP').exists():publish('STOPPED');return
            try:
                split=groups['/'.join(take.split('/')[:2])]
                ids=sorted(i for i in set().union(*(set(v) for v in native.values())) if splits.get((take,i))==split)
                chunks=np.split(np.array(ids,np.int64),np.flatnonzero(np.diff(ids)!=1)+1)
                frames=0;pieces=0;short_frames=0
                for part,chosen in enumerate(chunks):
                    if len(chosen)<28:short_frames+=len(chosen);continue
                    if len(chosen)>20000:raise ValueError('native take length exceeds20000')
                    seq,oid=convert_take(native,chosen.tolist());filename=take.replace('/','__')+f'__{part:02d}.npz'
                    target=root/'sequences'/filename
                    if target.exists():raise ValueError('unpublished sequence exists; refuse overwrite')
                    pending=root/'sequences'/('pending-'+filename);np.savez_compressed(pending,**seq);pending.replace(target)
                    row=dict(path='sequences/'+filename,format='canonical',objects=['H2O/'+NAMES[oid-1]],target_slot=0,
                        target_selection='single native object class',source_id=9,profile='h2o-native-joint3d11',split=split,
                        group_id='H2O/'+'/'.join(take.split('/')[:2]),source_sequence=take,camera='cam4',
                        native_start_frame=int(chosen[0]),native_end_frame=int(chosen[-1]))
                    (heldout if split=='test' else meta)['sequences'].append(row);frames+=len(chosen);pieces+=1
                ledger[take]=dict(status='CONVERTED',frames=frames,pieces=pieces,short_frames_excluded=short_frames,split=split)
                print('CONVERTED',take,split,pieces,frames,flush=True)
            except Exception as error:
                ledger[take]=dict(status='FAILED',error_type=type(error).__name__,reason=str(error));print('FAILED',take,type(error).__name__,flush=True)
            publish('CONVERTING')
        del takes
    publish('FINISHED')


if __name__=='__main__':main()
