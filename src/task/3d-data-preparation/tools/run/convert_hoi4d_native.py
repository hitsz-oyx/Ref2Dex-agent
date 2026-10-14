"""Pair native HOI4D rigid poses/CAD with cached camera-hand labels, without inventing a 30Hz clock."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation
from convert_hot3d_pilot import geometry

CLASSES={1:'ToyCar',2:'Mug',5:'Bottle',7:'Bowl',12:'Kettle',13:'Knife',14:'TrashCan',20:'Chair'}
POINTS=[0,1,5,9,13,17,4,8,12,16,20]


def write(path,value):
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def rigid(matrix):
    return (matrix.shape==(4,4) and np.isfinite(matrix).all() and np.allclose(matrix[3],[0,0,0,1],atol=1e-6)
        and np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=2e-4)
        and np.isclose(np.linalg.det(matrix[:3,:3]),1,atol=2e-4))


def parse_camera_log(text):
    """Raw Redwood LOG is camera-to-world; Open3D's loaded .extrinsic would be inverted."""
    lines=[v.strip() for v in text.splitlines() if v.strip() and not v.lstrip().startswith('#')]
    if len(lines)%5:raise ValueError('five-line camera records required')
    result={}
    for k in range(0,len(lines),5):
        header=list(map(int,lines[k].split()))
        if len(header)!=3 or header[0]!=header[1] or header[0] in result:raise ValueError('unique native camera frame headers required')
        value=np.array([list(map(float,line.split())) for line in lines[k+1:k+5]])
        if value.shape!=(4,4):raise ValueError('native4x4 required')
        result[header[0]]=value
    return result


def object_label(data,expected_class):
    if data.get('isEffective',1)!=1:return None,None,None
    candidates=data.get('dataList',data.get('objects',[]))
    wanted=''.join(c for c in expected_class.lower() if c.isalnum())
    matches=[v for v in candidates if ''.join(c for c in str(v.get('label','')).lower() if c.isalnum())==wanted]
    if len(matches)==1:row=matches[0]
    elif len(candidates)==1:row=candidates[0]
    else:return None,None,None
    trans=np.array([row['center'][k] for k in 'xyz'],np.float64)
    angles=np.array([row['rotation'][k] for k in 'xyz'],np.float64)
    dims=np.array([row['dimensions'][k] for k in ['length','width','height']],np.float64)
    if not np.isfinite(np.r_[trans,angles,dims]).all() or (dims<=0).any():return None,None,None
    out=np.eye(4);out[:3,:3]=Rotation.from_euler('XYZ',angles).as_matrix();out[:3,3]=trans
    return out,dims,str(row.get('label',''))


def pair_native(ids,hand,hand_valid,cameras,objects):
    n=len(ids);hw=np.full((n,2,11,3),np.nan,np.float32);hv=np.zeros(hw.shape[:-1],bool)
    poses=np.full((n,1,4,4),np.nan,np.float32);known=np.zeros((n,1),bool)
    cw=np.full((n,4,4),np.nan,np.float64);cv=np.zeros(n,bool)
    oc=np.full((n,4,4),np.nan,np.float64);dims=np.full((n,3),np.nan,np.float32)
    internal=np.full(n,-1,np.int64)
    for k,i in enumerate(ids):
        if i in objects:
            obj,dimension,label,frame_id=objects[i];internal[k]=frame_id
            if obj is not None:oc[k]=obj;dims[k]=dimension
        if i not in cameras:continue
        cw[k]=cameras[i]
        if not rigid(cw[k]):continue
        cv[k]=True
        if i in hand:
            hw[k]=hand[i][:,POINTS]@cw[k,:3,:3].T+cw[k,:3,3]
            hv[k]=hand_valid[i][:,POINTS]&np.isfinite(hw[k]).all(-1)
        if i in objects and objects[i][0] is not None and rigid(oc[k]):poses[k,0]=cw[k]@oc[k];known[k,0]=True
    return dict(native_frame_id=np.array(ids,np.int64),hand=hw,hand_valid=hv,object_poses=poses,object_pose_valid=known,
        native_camera_to_world=cw,native_camera_valid=cv,native_object_pose_camera=oc,native_object_dimensions=dims,
        native_json_frame_id=internal)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--hands',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--seconds',type=int,default=7200);a=p.parse_args()
    if not 60<=a.seconds<=86400:p.error('bounded deadline required')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    for name in ['sequences','geometry','scratch']:(root/name).mkdir(exist_ok=True)
    start=time.monotonic();deadline=start+a.seconds
    # Freeze the hand source index before pairing; a failed hand cache remains traceable in its own ledger.
    hand_meta=json.loads((a.hands/'manifest.json').read_text());hand_rows={v['source_sequence']:v for v in hand_meta['sequences']}
    official_release={v.strip() for v in (a.input/'release.txt').read_text().splitlines() if v.strip()}
    official_test={v.strip() for v in (a.input/'testset.txt').read_text().splitlines() if v.strip()}
    eligible=[]
    for identity in sorted(hand_rows):
        parts=identity.split('/');category=int(parts[2][1:])
        if category in CLASSES and identity in official_release and identity not in official_test:eligible.append(identity)
    if not eligible:raise ValueError('no released rigid native sequence found')
    instances=sorted({(int(v.split('/')[2][1:]),int(v.split('/')[3][1:])) for v in eligible});dev=set(instances[4::5])
    if a.limit:eligible=eligible[:a.limit]
    meta=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else dict(
        schema='ref2dex.native-pose-mesh.v1',units='m',reference_frame='native initial-camera world',training_allowed=False,
        objects=[],sequences=[],storage='one native sequence and shared original metric CAD; no overlapping windows',
        limitations=['Native frame geometry only; no uniform30Hz time field or training admission.',
            'Official decode15fps conflicts with some300-frame action durations10s; infer neither clocks nor missing labels silently.',
            'Rigid classes only; primary annotated object, no extra unannotated context.',
            'Original CAD canonical origin/scale retained; native bbox dimensions audited, no per-frame mesh deformation.',
            'Official release included, official test excluded; engineering object-instance-disjoint TRAIN/DEV.'])
    ledger=json.loads((root/'conversion-ledger.json').read_text()) if (root/'conversion-ledger.json').exists() else {}
    known={v['key'] for v in meta['objects']};completed_at_start=len(ledger)
    write(root/'progress.json',dict(status='INDEXING_NATIVE_ARCHIVES',pid=os.getpid(),checked_at=time.time(),training_allowed=False))
    with zipfile.ZipFile(a.input/'archives/HOI4D_annotations.zip') as labels,zipfile.ZipFile(a.input/'archives/HOI4D_CAD_Model_for_release.zip') as cad:
        label_names=set(labels.namelist());mesh_names=set(cad.namelist());phase='FINISHED'
        object_members={identity:[] for identity in eligible}
        for member in label_names:
            if '/objpose/' in member and member.endswith('.json'):
                identity=member.split('/objpose/',1)[0][len('HOI4D_annotations/'):]
                if identity in object_members:object_members[identity].append(member)
        for identity in eligible:
            if identity in ledger:continue
            if time.monotonic()>deadline or (root/'STOP').exists() or shutil.disk_usage(root).free<20*2**30:phase='STOPPED';break
            try:
                parts=identity.split('/');category=int(parts[2][1:]);instance=int(parts[3][1:]);group=(category,instance);name=CLASSES[category]
                prefix='HOI4D_annotations/'+identity+'/'
                log_name=prefix+'3Dseg/output.log';action_name=prefix+'action/color.json'
                if log_name not in label_names or action_name not in label_names:raise ValueError('native camera/action record missing')
                cameras=parse_camera_log(labels.read(log_name).decode());action=json.loads(labels.read(action_name));duration=float(action['info']['duration'])
                object_prefix=prefix+'objpose/'
                objects={};bad_json=0;id_mismatch=0;labels_seen=set()
                for member in object_members[identity]:
                    frame=int(Path(member).stem)
                    if frame in objects:raise ValueError('duplicate native object filename ID')
                    try:
                        data=json.loads(labels.read(member));obj,dim,label=object_label(data,name);internal=int(data.get('frameId',-1))
                        objects[frame]=(obj,dim,label,internal);labels_seen.add(str(label));id_mismatch+=int(internal!=frame)
                    except (ValueError,KeyError,TypeError):bad_json+=1
                with np.load(a.hands/hand_rows[identity]['path'],allow_pickle=False) as z:
                    hand_ids=z['native_frame_id'];hand={int(i):v for i,v in zip(hand_ids,z['hand_camera'])};valid={int(i):v for i,v in zip(hand_ids,z['hand_valid'])}
                ids=sorted(cameras)
                if not 28<=len(ids)<=20000 or not np.all(np.diff(ids)==1):raise ValueError('bounded contiguous native camera frames required')
                sequence=pair_native(ids,hand,valid,cameras,objects)
                if not sequence['object_pose_valid'].any() or not sequence['hand_valid'].any():raise ValueError('no finite paired native hand/object geometry')
                key=f'HOI4D/C{category}/N{instance}'
                if key not in known:
                    member=f'HOI4D_CAD_Model_for_release/rigid/{name}/{instance:03d}.obj'
                    if member not in mesh_names:raise ValueError('original rigid CAD absent')
                    scratch=root/'scratch'/f'C{category}-N{instance}.obj';scratch.write_bytes(cad.read(member));values=geometry(scratch)
                    # Full native mesh bounds, not sampled bounds, preserve physical dimensions for audit.
                    import trimesh
                    bounds=np.asarray(trimesh.load(str(scratch),process=False).bounds);scratch.unlink()
                    geometry_name=f'C{category}-N{instance}.npz';np.savez_compressed(root/'geometry'/geometry_name,**values)
                    meta['objects'].append(dict(key=key,path='geometry/'+geometry_name,full_native_bounds_m=bounds.tolist()));known.add(key)
                mesh_row=next(v for v in meta['objects'] if v['key']==key);bounds=np.array(mesh_row['full_native_bounds_m']);extent=bounds[1]-bounds[0]
                ratio=sequence['native_object_dimensions']/extent
                filename=identity.replace('/','__')+'.npz';target=root/'sequences'/filename
                if target.exists():raise ValueError('unpublished native sequence exists; refuse overwrite')
                pending=root/'sequences'/('pending-'+filename);np.savez_compressed(pending,**sequence);pending.replace(target)
                inferred=len(ids)/duration if duration>0 else None
                clock='CONFLICTING_OFFICIAL_DECODE_AND_DURATION' if inferred and abs(inferred-30)<.05 else ('CONSISTENT_WITH_NOMINAL15' if inferred and abs(inferred-15)<.05 else 'UNRESOLVED')
                row=dict(path='sequences/'+filename,source_sequence=identity,objects=[key],source_id=10,profile='hoi4d-author-native11',
                    group_id=key,split='dev' if group in dev else 'train',native_frames=len(ids),action_duration_s=duration,
                    duration_inferred_fps=inferred,clock_status=clock,filename_internal_id_mismatches=id_mismatch,
                    invalid_json_members=bad_json,native_object_labels=sorted(labels_seen),bbox_to_cad_extent_ratio_median=np.nanmedian(ratio,axis=0).tolist())
                meta['sequences'].append(row);ledger[identity]=dict(status='CONVERTED',frames=len(ids),clock_status=clock)
                print('CONVERTED',identity,len(ids),clock,flush=True)
            except Exception as error:
                ledger[identity]=dict(status='FAILED',error_type=type(error).__name__,reason=str(error));print('FAILED',identity,type(error).__name__,flush=True)
            write(root/'manifest.json',meta);write(root/'conversion-ledger.json',ledger)
            elapsed=time.monotonic()-start
            write(root/'progress.json',dict(status='CONVERTING_NATIVE_GEOMETRY',pid=os.getpid(),converted_sequences=len(meta['sequences']),
                converted_frames=sum(v['native_frames'] for v in meta['sequences']),total_sequences=len(eligible),objects=len(meta['objects']),
                failed_sequences=sum(v['status']=='FAILED' for v in ledger.values()),elapsed_s=elapsed,
                estimated_remaining_minutes=(len(eligible)-len(ledger))*elapsed/max(len(ledger)-completed_at_start,1)/60,
                checked_at=time.time(),training_allowed=False))
    write(root/'progress.json',dict(status=phase if not any(v['status']=='FAILED' for v in ledger.values()) else 'PARTIAL',pid=os.getpid(),
        converted_sequences=len(meta['sequences']),converted_frames=sum(v['native_frames'] for v in meta['sequences']),objects=len(meta['objects']),
        failed_sequences=sum(v['status']=='FAILED' for v in ledger.values()),elapsed_s=time.monotonic()-start,checked_at=time.time(),training_allowed=False))


if __name__=='__main__':main()
