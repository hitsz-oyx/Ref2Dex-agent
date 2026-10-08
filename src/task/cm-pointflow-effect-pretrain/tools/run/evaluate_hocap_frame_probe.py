"""Frozen HOCap frame-index Probe; never a verified-clock WM30 training pack."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    part = path.with_suffix('.partial')
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def validate_frame_manifest(meta, allow):
    if (not allow or meta.get('schema') != 'ref2dex.hocap-frame-probe.v1'
            or meta.get('training_allowed') is not False or meta.get('fps') is not None
            or meta.get('source_fps_verified') is not False
            or meta.get('nominal_feature_fps') != 30):
        raise ValueError('requires explicitly acknowledged test-only unverified-clock pack')


def pick_panel(rows, sequences, seed):
    """At most two moving and one remaining natural row per sequence, no replacement."""
    import numpy as np
    rng = np.random.default_rng(seed)
    moving, natural = [], []
    for sequence in range(sequences):
        pool = np.flatnonzero(rows[:, 0] == sequence)
        movers = pool[rows[pool, 3] == 0]
        selected = rng.choice(movers, min(2, len(movers)), replace=False)
        moving.extend(selected.tolist())
        remaining = np.setdiff1d(pool, selected)
        if len(remaining): natural.append(int(rng.choice(remaining)))
    return np.array(moving, dtype=np.int64), np.array(natural, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='download run with dataset/')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manopth', type=Path, required=True, help='pinned HOCap author fork')
    parser.add_argument('--mano-root', type=Path, default=ROOT/'data/raw_data/ARCTIC/arctic/data/body_models/mano')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-unverified-clock', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=228)
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    if not args.allow_unverified_clock or not 100 <= args.seed < 300 or not 1 <= args.seconds <= 1800:
        parser.error('explicit clock acknowledgement and bounded Probe required')
    out = args.output.resolve()
    if ROOT/'outputs/cm-pointflow-effect-pretrain' not in out.parents or out.exists():
        parser.error('fresh task-owned output required')
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied: raise RuntimeError('GPU occupied: '+occupied)
    out.mkdir(parents=True)
    scratch = ROOT/'tmp/hocap-frame-probe'; scratch.mkdir(exist_ok=True, parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.manopth.resolve()))
    sys.path.insert(0, str(TASK/'src'))
    sys.path.insert(0, str(TASK/'tools/run/ref5_data_expansion'))
    started = time.time(); stop = [False]; deadline = started + args.seconds
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stop.__setitem__(0, True))
    manifest = dict(status='RUNNING', git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    started_at=started, deadline=deadline, pid=os.getpid(), gpu=args.gpu,
                    training_allowed=False, test_ready=False, source_fps_verified=False,
                    source_fps=None, nominal_feature_fps=30, horizon_unit='original array frame index',
                    seed=args.seed, checkpoint=str(args.checkpoint.resolve()), frozen_inputs={})
    signatures = {}
    def pin(path):
        path = Path(path).resolve(); s = path.stat()
        manifest['frozen_inputs'][str(path)] = sha(path)
        signatures[str(path)] = (s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    def guard(full=False):
        if stop[0] or time.time() >= deadline: raise TimeoutError('original Probe deadline/stop')
        if shutil.disk_usage(out).free < 20*2**30: raise RuntimeError('20GiB disk reserve')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 2**30: raise RuntimeError('1GiB output cap')
        for path, signature in signatures.items():
            s = Path(path).stat()
            if (s.st_size,s.st_mtime_ns,s.st_ctime_ns) != signature or (full and sha(path) != manifest['frozen_inputs'][path]):
                raise ValueError('frozen input drift: '+path)
        pids = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
        if any(int(p) != os.getpid() for p in pids if p.strip()): raise RuntimeError('foreign GPU process appeared')
    try:
        import numpy as np
        for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
            if name not in np.__dict__: setattr(np,name,value)
        import torch
        import yaml
        import trimesh
        from scipy.spatial import cKDTree
        from scipy.spatial.transform import Rotation
        from manopth.manolayer import ManoLayer
        from oakink_wm.data import Windows, local_objects
        from oakink_wm.pointworld_temporal import model_from_config, VENDOR
        from oakink_wm.pointworld_performance import install_fused_hilbert
        from prepare_native import sample_surface
        import train_oakink2_pointworld_temporal as base
        base.configure_numerics(); install_fused_hilbert()
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        raw = args.source.resolve()/'dataset'
        pin(args.checkpoint);pin(Path(__file__))
        for path in raw.rglob('*'):
            if path.is_file():pin(path)
        for directory in (args.manopth, TASK/'src/oakink_wm', VENDOR/'ptv3'):
            for path in Path(directory).rglob('*.py'):pin(path)
        for side in ('LEFT','RIGHT'):pin(args.mano_root/('MANO_'+side+'.pkl'))
        pin(TASK/'tools/run/ref5_data_expansion/prepare_native.py');pin(Path(base.__file__))
        state = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        for key,value in state['identity']['implementation_sources'].items():
            if sha(TASK/key) != value:raise ValueError('checkpoint implementation drift: '+key)
        for key,value in state['identity']['vendor_sources'].items():
            if sha(VENDOR/key) != value:raise ValueError('checkpoint vendor drift: '+key)
        manifest.update(checkpoint_step=state['step'], parent_git_commit=state['identity']['git_commit'],
                        stats_sha256=state['identity']['stats_sha256'], weights_only_inference=True,
                        manopth_revision=subprocess.check_output(['git','-C',str(args.manopth),'rev-parse','HEAD'],text=True).strip())
        pack = out/'processed';pack.mkdir();(pack/'canonical').mkdir();(pack/'sequences').mkdir()
        models = {side:ManoLayer(side=side,mano_root=str(args.mano_root),flat_hand_mean=False,ncomps=45,use_pca=True).cuda() for side in ('right','left')}
        canonical = {}; records = []; rows = []; semantics = [0,1,5,9,13,17,4,8,12,16,20]
        for index,seq in enumerate(sorted(raw.glob('subject_*/*'))):
            guard();m=yaml.safe_load((seq/'meta.yaml').read_text());param=np.load(seq/'poses_m.npy');o=np.load(seq/'poses_o.npy');n=m['num_frames']
            hv=(np.isfinite(param).all(-1)&~np.all(param==-1,axis=-1)).T
            hand=np.zeros((n,2,11,3),dtype=np.float32); joints21=np.zeros((n,2,21,3),dtype=np.float32)
            beta=yaml.safe_load((raw/'calibration/mano'/(m['subject_id']+'.yaml')).read_text())['betas']
            for side_id,side in enumerate(('right','left')):
                ids=np.flatnonzero(hv[:,side_id])
                with torch.no_grad():
                    for begin in range(0,len(ids),128):
                        chosen=ids[begin:begin+128];p=torch.tensor(param[side_id,chosen],device='cuda');b=torch.tensor(beta,dtype=torch.float32,device='cuda').expand(len(chosen),-1)
                        j=models[side](p[:,:48],b,p[:,48:])[1].cpu().numpy()/1000
                        joints21[chosen,side_id]=j;hand[chosen,side_id]=j[:,semantics]
            if index==0:
                ext=yaml.safe_load((raw/'calibration/extrinsics'/m['extrinsics']).read_text())['extrinsics']
                def mat(v):return np.array([v[:4],v[4:8],v[8:12],[0,0,0,1]])
                labels=sorted(seq.glob('*/label_*.npz'));errors=[]
                for path in labels:
                    c2w=np.linalg.inv(mat(ext['tag_1']))@mat(ext[path.parent.name]);t=int(path.stem.split('_')[-1])
                    with np.load(path,allow_pickle=False) as z:
                        source=z['hand_joints_3d'];world=source@c2w[:3,:3].T+c2w[:3,3]
                        valid=np.isfinite(source).all(-1)&~np.all(source==-1,axis=-1)
                        errors.extend(np.linalg.norm(joints21[t]-world,axis=-1)[valid].tolist())
                if not errors or max(errors)>1e-5:raise ValueError('MANO/camera labels disagree')
                manifest['mano_label_check']=dict(frames=len(labels),max_error_m=max(errors),mean_error_m=float(np.mean(errors)))
            pv=(np.isfinite(o).all(-1)&~np.all(o==-1,axis=-1)).T
            poses=np.broadcast_to(np.eye(4),(n,len(o),4,4)).copy();safe=o.copy();safe[~pv.T]=[0,0,0,1,0,0,0]
            poses[:,:,:3,:3]=Rotation.from_quat(safe[...,:4].reshape(-1,4)).as_matrix().reshape(len(o),n,3,3).transpose(1,0,2,3)
            poses[:,:,:3,3]=safe[...,4:].transpose(1,0,2);poses=poses.astype(np.float32)
            objects=['hocap_'+name for name in m['object_ids']];centers=[]
            for name,key in zip(m['object_ids'],objects):
                if key not in canonical:
                    mesh=trimesh.load(raw/'models'/name/'cleaned_mesh_10000.obj',process=False);canonical[key]=sample_surface(mesh,key);np.savez_compressed(pack/'canonical'/(key+'.npz'),**canonical[key])
                centers.append(canonical[key]['center'])
            centers=np.asarray(centers,np.float32);name=seq.parent.name+'__'+seq.name;dest=pack/'sequences'/name;dest.mkdir()
            for key,value in dict(hand=hand,hand_valid=hv,poses=poses,pose_valid=pv,centers=centers,
                                  frame_ids=np.arange(n,dtype=np.int64),program=np.zeros((n,len(o)),bool),near=np.zeros((n,len(o)),bool)).items():np.save(dest/(key+'.npy'),value)
            write(dest/'meta.json',dict(objects=objects,frames=n,source_sequence=seq.relative_to(raw).as_posix(),frame_ids_kind='array_index_only'))
            hand_bad=((np.linalg.norm(np.diff(hand,axis=0),axis=-1)>.15)&hv[1:,:,None]&hv[:-1,:,None]).any((1,2))
            delta=np.linalg.norm(np.diff(poses[:,:,:3,3],axis=0),axis=-1);relative=poses[1:,:,:3,:3]@poses[:-1,:,:3,:3].swapaxes(-1,-2)
            pbad=(delta>.15)|(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1))>.6)
            count=0
            for t in range(3,n-24,8):
                if not hv[t].any() or not (hv[t-3:t+25]==hv[t]).all() or hand_bad[t-3:t+24].any():continue
                for anchor in np.flatnonzero(pv[t]):
                    cloud=canonical[objects[anchor]]['points']@poses[t,anchor,:3,:3].T+poses[t,anchor,:3,3]
                    distances=cKDTree(cloud).query(hand[t,hv[t]].reshape(-1,3))[0]
                    if distances.min()>=.05:continue
                    selected=local_objects(poses[t],pv[t],anchor,centers)
                    if not pv[t-3:t+25,selected].all() or pbad[t-3:t+24,selected].any():continue
                    future=poses[t+1:t+25,anchor];rotation=future[:,:3,:3]@poses[t,anchor,:3,:3].T
                    moving=(np.linalg.norm(future[:,:3,3]-poses[t,anchor,:3,3],axis=-1)>.002).any() or (np.arccos(np.clip((np.trace(rotation,axis1=-2,axis2=-1)-1)/2,-1,1))>.02).any()
                    rows.append([index,anchor,t,0 if moving else 1]);count+=1
            records.append(dict(sequence=name,source_sequence=seq.relative_to(raw).as_posix(),task=m['task_id'],frames=n,candidates=count))
            if index%8==0:print(json.dumps(dict(stage='prepare',sequences=index+1,candidates=len(rows),elapsed=time.time()-started)),flush=True)
        del models;torch.cuda.empty_cache();guard()
        rows=np.asarray(rows,dtype=np.int64).reshape(-1,4);np.save(pack/'index_test.npy',rows)
        meta=dict(schema='ref2dex.hocap-frame-probe.v1',status='COMPLETED',fps=None,nominal_feature_fps=30,
                  source_fps_verified=False,training_allowed=False,sequences=[r['sequence'] for r in records],records=records)
        write(pack/'manifest.json',meta);validate_frame_manifest(meta,args.allow_unverified_clock)
        moving,natural=pick_panel(rows,len(records),args.seed)
        if not len(moving) or not len(natural):raise ValueError('insufficient frozen panels')
        np.save(out/'moving_panel.npy',moving);np.save(out/'natural_panel.npy',natural)
        for path in pack.rglob('*'):
            if path.is_file():pin(path)
        pin(out/'moving_panel.npy');pin(out/'natural_panel.npy')
        class FrameWindows(Windows):
            def __init__(self):
                self.root=out;self.meta=meta;self.rows=rows;self.sequences=meta['sequences'];self.groups=[np.flatnonzero(rows[:,3]==k) for k in range(3)];self.calls=0
            def __getitem__(self,item):
                self.calls+=1
                if self.calls%16==1:guard()
                return super().__getitem__(item)
            def sequence(self,seq):
                value=dict(super().sequence(seq))
                # Internal OakInk120Hz index assertion only; not timestamps or source FPS.
                value['frame_ids']=4*value['frame_ids'];return value
        dataset=FrameWindows();model=model_from_config(state['identity']['stats'],state['config']).cuda();model.load_state_dict(state['model'],strict=True)
        del state;torch.cuda.reset_peak_memory_stats();results={}
        for name,panel in [('moving',moving),('natural',natural)]:
            guard();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
            values=base.evaluate(model,dataset,panel,'action',2,True);write(out/(name+'_metrics.json'),values)
            selected=rows[panel];summary={}
            for h in (1,4,8,12,24):
                key='anchor/cat-1/h'+str(h)+'/point_epe'
                summary['h'+str(h)]=dict(model_mm=values['model/'+key]*1000,static_mm=values['static/'+key]*1000)
            summary.update(windows=len(panel),sequences=len(np.unique(selected[:,0])),subjects=len(set(records[s]['source_sequence'].split('/')[0] for s in selected[:,0])))
            results[name]=summary;print(json.dumps(dict(stage='evaluate',panel=name,summary=summary,elapsed=time.time()-started)),flush=True)
        guard(full=True);manifest.update(status='COMPLETED',elapsed_seconds=time.time()-started,panels=results,
                                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,verdict='UNCLEAR',
                                       limitation='unverified source clock; nominal30Hz feature scaling; released human future hand conditioning; no fitting, checkpoint selection or robot utility')
        write(out/'result.json',manifest)
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error),elapsed_seconds=time.time()-started);raise
    finally:write(out/'input_manifest.json',manifest)


if __name__=='__main__':main()
