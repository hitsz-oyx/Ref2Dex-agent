"""Import already authorized HO-Cap/DexYCB geometry without image processing."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from convert_hot3d_pilot import geometry, SCHEMA, validate_sequence


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hocap','dexycb'],required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'sequences').mkdir();(root/'geometry').mkdir()
    meta=dict(schema=SCHEMA,units='m',fps=30,source_fps_verified=False,training_allowed=False,objects=[],sequences=[])
    known=set();records=[];errors=[]
    if a.dataset=='hocap':
        source=json.loads((a.source/'manifest.json').read_text())
        if source['schema']!='ref2dex.hocap-frame-probe.v1' or source['training_allowed'] is not False:
            raise ValueError('known held-out geometry provenance required')
        meta['limitations']=['Entire HO-Cap remains held-out dev; no previous test data enters TRAIN.',
            'Original frame clock unverified;30Hz is explicitly nominal for engineering.']
        for name in source['sequences']:
            folder=a.source/'sequences'/name;row=json.loads((folder/'meta.json').read_text());n=row['frames']
            objects=row['objects']
            hand=np.load(folder/'hand.npy',allow_pickle=False)
            hv=np.load(folder/'hand_valid.npy',allow_pickle=False)
            sequence=dict(time=np.arange(n)/30,hand=hand,hand_valid=np.broadcast_to(hv[...,None],(n,2,11)).copy(),
                object_poses=np.load(folder/'poses.npy',allow_pickle=False),object_pose_valid=np.load(folder/'pose_valid.npy',allow_pickle=False),
                segment=np.zeros(n,np.int64),native_frame_id=np.load(folder/'frame_ids.npy',allow_pickle=False))
            validate_sequence(sequence)
            np.savez_compressed(root/'sequences'/(name+'.npz'),**sequence)
            for key in objects:
                if key not in known:
                    with np.load(a.source/'canonical'/(key+'.npz'),allow_pickle=False) as z:
                        points=z['points'];normals=z['normals']
                    np.savez_compressed(root/'geometry'/(key+'.npz'),surface=points,normals=normals,query=points[np.linspace(0,len(points)-1,32,dtype=int)])
                    meta['objects'].append(dict(key=key,path='geometry/'+key+'.npz'));known.add(key)
            meta['sequences'].append(dict(path='sequences/'+name+'.npz',format='canonical',objects=objects,target_slot=0,
                source_id=7,profile='hocap-author-mano-native11',split='dev',group_id='HO-Cap/'+name.split('__')[0],
                source_sequence=row['source_sequence'],source_clock='array frame IDs; original FPS unverified'))
            records.append(dict(sequence=name,frames=n));print('IMPORTED',name,n,flush=True)
    else:
        import yaml
        classes=['002_master_chef_can','003_cracker_box','004_sugar_box','005_tomato_soup_can','006_mustard_bottle',
                 '007_tuna_fish_can','008_pudding_box','009_gelatin_box','010_potted_meat_can','011_banana',
                 '019_pitcher_base','021_bleach_cleanser','024_bowl','025_mug','035_power_drill','036_wood_block',
                 '037_scissors','040_large_marker','051_large_clamp','052_extra_large_clamp','061_foam_brick']
        meta['limitations']=['Only already authorized subject10; all sequences official s1 TRAIN.',
            'One fixed stationary camera reference per sequence; eight views not duplicated.',
            'Native joint_3d/pose_y labels, no MANO re-estimation;30Hz nominal pending source-clock evidence.']
        folders=sorted((a.source/'subject-10/20201022-subject-10').glob('*'))
        for folder in folders:
            if not (folder/'meta.yml').exists():continue
            try:
                m=yaml.safe_load((folder/'meta.yml').read_text());n=m['num_frames'];serial='836212060125'
                if serial not in m['serials']:raise ValueError('fixed camera absent')
                hand=np.full((n,2,11,3),np.nan,np.float32);hv=np.zeros(hand.shape[:-1],bool)
                poses=np.full((n,len(m['ycb_ids']),4,4),np.nan,np.float32);known_pose=np.zeros(poses.shape[:2],bool)
                side=0 if m['mano_sides'][0]=='right' else 1
                for frame in range(n):
                    file=folder/serial/(f'labels_{frame:06d}.npz')
                    if not file.exists():continue
                    with np.load(file,allow_pickle=False) as z:
                        points=z['joint_3d'][0];params=z['pose_m'][0];raw=z['pose_y']
                    valid=np.isfinite(points).all(-1)&~(points==-1).all(-1)
                    valid &= np.isfinite(params).all() and not (params==-1).all() and not (points==0).all()
                    ids=[0,1,5,9,13,17,4,8,12,16,20];hand[frame,side]=points[ids];hv[frame,side]=valid[ids]
                    for slot,value in enumerate(raw):
                        rot=value[:3,:3]
                        if (np.isfinite(value).all() and np.allclose(rot.T@rot,np.eye(3),atol=2e-4)
                            and np.isclose(np.linalg.det(rot),1,atol=2e-4)):
                            poses[frame,slot]=np.eye(4);poses[frame,slot,:3,:]=value;known_pose[frame,slot]=True
                sequence=dict(time=np.arange(n)/30,hand=hand,hand_valid=hv,object_poses=poses,object_pose_valid=known_pose,
                              segment=np.zeros(n,np.int64),native_frame_id=np.arange(n,dtype=np.int64))
                validate_sequence(sequence);objects=[]
                for oid in m['ycb_ids']:
                    key='DexYCB/'+str(oid);objects.append(key)
                    if key not in known:
                        values=geometry(a.source/'models'/classes[oid-1]/'textured_simple.obj')
                        np.savez_compressed(root/'geometry'/(str(oid)+'.npz'),**values)
                        meta['objects'].append(dict(key=key,path='geometry/'+str(oid)+'.npz'));known.add(key)
                np.savez_compressed(root/'sequences'/(folder.name+'.npz'),**sequence)
                meta['sequences'].append(dict(path='sequences/'+folder.name+'.npz',format='canonical',objects=objects,
                    target_slot=int(m['ycb_grasp_ind']),target_selection='native ycb_grasp_ind',source_id=8,
                    profile='dexycb-native-joint3d11',split='train',group_id='DexYCB/subject10',source_sequence=folder.name,
                    reference_frame='fixed camera '+serial,source_clock='array frame IDs;30Hz nominal'))
                records.append(dict(sequence=folder.name,frames=n));print('IMPORTED',folder.name,n,flush=True)
            except Exception as error:
                errors.append(dict(sequence=folder.name,error_type=type(error).__name__,reason=str(error)))
                print('FAILED',folder.name,type(error).__name__,flush=True)
    meta['source_root']=str(a.source.resolve());write(root/'manifest.json',meta)
    write(root/'conversion-audit.json',dict(status='COMPLETE' if not errors else 'PARTIAL',sequences=len(records),
        frames=sum(r['frames'] for r in records),objects=len(known),records=records,errors=errors,training_allowed=False))


if __name__=='__main__':main()
