"""Extract original hand annotations into atomic, compact reconstruction packets.

Runs where the licensed raw annotations already reside; never writes into them.
The remote consumer reconstructs MANO and validates against the frozen scaffold.
"""
import argparse
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[5]
PREP = ROOT/'outputs/3d-data-preparation'
MAIN = ROOT/'outputs/cm-pointflow-effect-pretrain'


def atomic_json(path, value):
    pending = path.with_suffix('.json.part')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def local_folder(job):
    if job['source_id'] < 3:
        root = MAIN/('oakink2-wm30-k24-20261006' if job['source_id']==0 else 'native-official-full-20261007')
        return root/'processed/sequences'/job['source_sequence']
    return Path(job['source_row']['source_path'])


def blank(n):
    return dict(pose=np.zeros((n,2,48),np.float32), betas=np.zeros((n,2,10),np.float32),
        transl=np.zeros((n,2,3),np.float32), parameter_valid=np.zeros((n,2),bool),
        world=np.broadcast_to(np.eye(4,dtype=np.float32),(n,4,4)).copy(),
        center_wrist=np.array(False), flat_mean=np.array(True), pca=np.array(0),
        tip_convention=np.array('manopth'), resample=np.array(False))


class Extractor:
    def __init__(self):
        self.taco = None

    def packet(self, job):
        sid = job['source_id']
        folder = local_folder(job)
        if sid < 3:
            ids = np.load(folder/('frame_ids.npy' if sid==0 else 'source_frame_ids.npy'))
            out = blank(len(ids))
            if sid==0:
                with (MAIN/'oakink2-wm30-k24-20261006/download'/(job['source_sequence']+'.pkl')).open('rb') as f:
                    raw = pickle.load(f)
                out['center_wrist'] = np.array(True)
                for i, fid in enumerate(ids):
                    row = raw['raw_mano'].get(int(fid), {})
                    for side, prefix in enumerate(('rh__','lh__')):
                        if not all(prefix+k in row for k in ('pose_coeffs','betas','tsl')):
                            continue
                        q = np.asarray(row[prefix+'pose_coeffs']).reshape(16,4)
                        if not np.isfinite(q).all() or (np.linalg.norm(q,axis=-1)<1e-6).any():
                            continue
                        out['pose'][i,side] = Rotation.from_quat(q[:,[1,2,3,0]]).as_rotvec().reshape(48)
                        out['betas'][i,side] = np.asarray(row[prefix+'betas']).reshape(10)
                        out['transl'][i,side] = np.asarray(row[prefix+'tsl']).reshape(3)
                        out['parameter_valid'][i,side] = True
            elif sid==1:
                import trimesh
                raw_path = Path(job['raw_path'])
                with np.load(raw_path, allow_pickle=True) as z:
                    raw = {k:z[k].item() if z[k].shape==() else z[k] for k in z.files}
                templates = []
                for side, key in enumerate(('rhand','lhand')):
                    r = raw[key]['params']
                    out['pose'][:,side] = np.concatenate((r['global_orient'][ids],r['fullpose'][ids]),-1)
                    out['transl'][:,side] = r['transl'][ids]
                    out['parameter_valid'][:,side] = True
                    mesh = trimesh.load(raw_path.parents[2]/raw[key]['vtemp'],process=False)
                    templates.append(np.asarray(mesh.vertices,np.float32))
                out['templates'] = np.stack(templates)
                out['tip_convention'] = np.array('smplx-cache')
            else:
                raw = np.load(job['raw_path'],allow_pickle=True).item()
                for side, name in enumerate(('right','left')):
                    r = raw[name]
                    out['pose'][:,side] = np.concatenate((r['rot'][ids],r['pose'][ids]),-1)
                    out['betas'][:,side] = np.broadcast_to(r['shape'],(len(ids),10))
                    out['transl'][:,side] = r['trans'][ids]
                    out['parameter_valid'][:,side] = True
                out['flat_mean'] = np.array(False)
                out['tip_convention'] = np.array('smplx-cache')
            return out
        with np.load(folder, allow_pickle=False) as z:
            base = {k:z[k] for k in z.files}
        n = len(base['time'])
        out = blank(n)
        if sid==4:
            return dict(q=base['q'], hand_root=base['hand_root'], robot=np.array('shadow-official-replay'))
        if sid==5:
            sys.path.insert(0,str(ROOT/'src/task/3d-data-preparation/tools/run'))
            from convert_taco_pilot import CPUReader
            if self.taco is None:
                self.taco = zipfile.ZipFile(PREP/'taco-native-full-20261014-r1/archives/Hand_Poses.zip')
            names = set(self.taco.namelist())
            prefix = 'Hand_Poses/'+job['source_sequence']
            ids = base['native_frame_id']
            out['center_wrist'] = np.array(True)
            for side, name in enumerate(('right','left')):
                member = prefix+'/'+name+'_hand.pkl'
                if member not in names:
                    continue
                raw = CPUReader(io.BytesIO(self.taco.read(member))).load()
                lookup = {int(k):v for k,v in raw.items()}
                shape_member = prefix+'/'+name+'_hand_shape.pkl'
                if shape_member not in names:
                    raise ValueError('missing original TACO shape; no zero-shape fallback')
                shape = CPUReader(io.BytesIO(self.taco.read(shape_member))).load()['hand_shape']
                out['betas'][:,side] = np.asarray(shape).reshape(10)
                for i,fid in enumerate(ids):
                    r = lookup[int(fid)]
                    out['pose'][i,side] = np.asarray(r['hand_pose']).reshape(48)
                    out['transl'][i,side] = np.asarray(r['hand_trans']).reshape(3)
                    out['parameter_valid'][i,side] = True
        elif sid==6:
            stem = Path(job['source_sequence']).stem
            family, identity = stem.split('-',1)
            file = PREP/'hot3d-native-full-20261014-r1/native'/family/identity/'mano_hand_pose_trajectory.jsonl'
            ids = base['native_timestamp_ns']
            wanted = set(ids.tolist())
            raw = {}
            with file.open() as f:
                for line in f:
                    r = json.loads(line)
                    if r['timestamp_ns'] in wanted:
                        raw[r['timestamp_ns']] = r['hand_poses']
            out['pca'] = np.array(15)
            out['pca_pose'] = np.zeros((n,2,15),np.float32)
            out['flat_mean'] = np.array(False)
            out['left_shape_fix'] = np.array(True)
            out['tip_convention'] = np.array('smplx-cache')
            for i,t in enumerate(ids):
                for side in (0,1):
                    r = raw.get(int(t),{}).get(str(1-side))
                    if r is None or not base['hand_valid'][i,side].all():
                        continue
                    q = r['wrist_xform']['q_wxyz']
                    out['pose'][i,side,:3] = Rotation.from_quat(q[1:]+q[:1]).as_rotvec()
                    out['pca_pose'][i,side] = r['pose']
                    out['betas'][i,side] = r['betas']
                    out['transl'][i,side] = r['wrist_xform']['t_xyz']
                    out['parameter_valid'][i,side] = True
        elif sid==8:
            import yaml
            matches = list((PREP/'dexycb-labels-full-20261015-r1/raw').glob('*/'+job['source_sequence']))
            if len(matches)!=1:
                raise ValueError('unique DexYCB native folder required')
            rawfolder = matches[0]
            meta = yaml.safe_load((rawfolder/'meta.yml').read_text())
            side = 0 if meta['mano_sides'][0]=='right' else 1
            joints = np.zeros((n,2,21,3),np.float32)
            jv = np.zeros((n,2,21),bool)
            for i,fid in enumerate(base['native_frame_id']):
                path = rawfolder/'836212060125'/f'labels_{int(fid):06d}.npz'
                if not path.exists():
                    continue
                with np.load(path,allow_pickle=False) as z:
                    jp, pose = z['joint_3d'][0],z['pose_m'][0]
                valid = np.isfinite(jp).all(-1) & ~(jp==-1).all(-1)
                valid &= bool(base['hand_valid'][i,side].all())
                joints[i,side] = jp
                jv[i,side] = valid
            out['native_joints'], out['native_joint_valid'] = joints,jv
            # Native joint21 is authoritative. MANO is not admitted until the
            # source's PCA/shape/translation convention is separately checked.
        elif sid==9:
            camera = base['native_camera_to_world']
            jp = base['native_joints_camera']
            out['native_joints'] = np.einsum('nij,nhpj->nhpi',camera[:,:3,:3],jp)+camera[:,None,None,:3,3]
            out['native_joint_valid'] = np.broadcast_to(base['hand_valid'].all(-1)[...,None],(n,2,21)).copy()
            # Preserve native MANO for a later audited surface adapter.
            out['native_mano_camera'] = base['native_mano_camera']
        elif sid==10:
            native = Path(job['source_row']['native_source_path'])
            with np.load(native,allow_pickle=False) as z:
                cams, ids = z['native_camera_to_world'], z['native_frame_id']
            hp = PREP/'hoi4d-hand-native-20261014-r1/sequences'/(job['source_sequence'].replace('/','__')+'.npz')
            with np.load(hp,allow_pickle=False) as z:
                lookup = {int(v):i for i,v in enumerate(z['native_frame_id'])}
                order = [lookup[int(v)] for v in ids]
                params = z['native_mano_camera'][order]
                valid = z['hand_valid'][order]
            out = blank(len(ids))
            out['pose'],out['betas'],out['transl'] = params[...,:48],params[...,48:58],params[...,58:61]
            out['parameter_valid'] = valid.all(-1) & np.isfinite(params).all(-1)
            out['world'],out['resample'] = cams,np.array(True)
        else:
            raise ValueError('unsupported source')
        return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=21600)
    p.add_argument('--sources',default='0,1,2,5,6,8,9,10,4')
    p.add_argument('--limit-per-source',type=int)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    for sub in ('packets','records'):(a.output/sub).mkdir(exist_ok=True)
    plan=json.loads(a.plan.read_text());extractor=Extractor();started=time.monotonic()
    ledger={};counts={};sources=list(map(int,a.sources.split(',')))
    # First publish one packet/source, then stream the full corpus in order.
    jobs=sorted(plan['jobs'],key=lambda j:(sources.index(j['source_id']) if j['source_id'] in sources else 999,j['key']))
    first=[next(j for j in jobs if j['source_id']==sid) for sid in sources if any(j['source_id']==sid for j in jobs)]
    ordered=first+[j for j in jobs if j not in first]
    for job in ordered:
        sid=job['source_id'];key=job['key']
        if sid not in sources or (a.limit_per_source and counts.get(sid,0)>=a.limit_per_source):continue
        if time.monotonic()-started>a.seconds or (a.output/'STOP').exists() or shutil.disk_usage(a.output).free<20*2**30:break
        record=a.output/'records'/(key+'.json')
        if record.exists():continue
        try:
            packet=extractor.packet(job)
            pending=a.output/'packets'/(key+'.part.npz')
            np.savez_compressed(pending,**packet)
            pending.replace(a.output/'packets'/(key+'.npz'))
            row=dict(status='READY',source_id=sid,key=key,packet='packets/'+key+'.npz')
        except Exception as e:
            row=dict(status='UNAVAILABLE',source_id=sid,key=key,reason=type(e).__name__+': '+str(e))
        atomic_json(record,row);ledger[key]=row;counts[sid]=counts.get(sid,0)+1
        if len(ledger)%10==0 or len(ledger)<=9:
            progress=dict(status='RUNNING',pid=os.getpid(),processed=len(ledger),counts=counts,
                ready=sum(r['status']=='READY' for r in ledger.values()),elapsed_s=time.monotonic()-started,
                python=sys.version,device='CPU annotation extraction; no neural inference')
            atomic_json(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    atomic_json(a.output/'export-finished.json',dict(status='FINISHED',processed=len(ledger),counts=counts,elapsed_s=time.monotonic()-started))


if __name__=='__main__':main()
