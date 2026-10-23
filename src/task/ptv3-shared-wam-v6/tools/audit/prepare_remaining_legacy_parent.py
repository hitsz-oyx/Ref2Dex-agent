"""Prepare the sole local non-TEST GRAB parent absent from the frozen plan.

No old IDs/splits are rewritten. The appended parent still passes the normal
rigid admission, contact and event conversion; raw existence is not admission.
"""
import argparse
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_full_rigid_data as prep


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();run=a.run.resolve()
    tools=prep.TASK.parent/'cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion'
    sys.path[:0]=[str(tools),str(prep.TASK.parent/'cm-pointflow-effect-pretrain/src')]
    import prepare_native as native
    from oakink_wm.native_splits import grab_split
    from export_hand_packets import blank
    import torch
    torch.set_num_threads(1)
    root=Path('/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/dexplore_grab/sequences')
    original=prep.read(prep.PLAN)['jobs'];known={r['source_sequence'] for r in original if r['source_id']==1}
    unplanned=[f for f in sorted(root.iterdir()) if f.is_dir() and 'grab_'+f.name not in known and grab_split(f.name.split('_',1)[1])!='test']
    if len(unplanned)!=1:raise ValueError('expected the single audited unplanned TRAIN parent')
    folder=unplanned[0];subject,name=folder.name.split('_',1)
    raw=Path('/home2/wyy/oyx_ws/Ref2Dex/data/raw_data/GRAB/grab')/subject/(name+'.npz')
    options=SimpleNamespace(grab_root=raw.parents[2],mano_root=Path('/home2/wyy/oyx_ws/Ref2Dex/dataset/arctic/data/body_models/mano/models'),device='cuda:7')
    hand,poses,ids,clock,meshes,dependencies=native.grab(raw,options)
    with np.load(raw,allow_pickle=True) as z:r={k:z[k].item() if z[k].shape==() else z[k] for k in z.files}
    valid=np.isfinite(hand).all(axis=(-1,-2));hand[~valid]=0;n=len(ids)
    with np.load(folder/'motion.npz',allow_pickle=False) as z:
        if int(z['n_frames'])!=n:raise ValueError('legacy/native parent clock mismatch')
    packet=blank(n);templates=[]
    import trimesh
    for side,key in enumerate(['rhand','lhand']):
        params=r[key]['params'];packet['pose'][:,side]=np.concatenate([params['global_orient'][ids],params['fullpose'][ids]],axis=-1)
        packet['transl'][:,side]=params['transl'][ids];packet['parameter_valid'][:,side]=True
        templates.append(np.asarray(trimesh.load(options.grab_root/r[key]['vtemp'],process=False).vertices,np.float32))
    packet['templates']=np.stack(templates);packet['tip_convention']=np.array('smplx-cache')
    data=dict(hand=hand.astype(np.float32),hand_valid=valid,poses=poses.astype(np.float32),pose_valid=native.rigid_valid(poses),source_frame_ids=ids)
    target=run/'remaining-legacy-parent';key='s1-local-extra-00000';packet_path=target/'packet.npz'
    windows=np.column_stack([np.zeros(max(n-27,0),np.int32),np.arange(3,n-24,dtype=np.int32)])
    buffers={}
    for k,v in data.items():
        stream=io.BytesIO();np.save(stream,v);buffers[k+'.npy']=stream.getvalue()
    stream=io.BytesIO();np.savez_compressed(stream,**packet);buffers['packet.npz']=stream.getvalue()
    stream=io.BytesIO();np.save(stream,windows);buffers['candidate-train.npy']=stream.getvalue()
    byte_count=sum(map(len,buffers.values()));conversion=prep.fixed_conversion_sizes(n,1)
    source_sequence='grab_'+folder.name
    budget=dict(parent=source_sequence,frames=n,exact_native_input_bytes=byte_count,exact_conversion_bytes=conversion,
        extra_candidate_windows=len(windows),free_before_bytes=prep.shutil.disk_usage(run).free,storage_floor_bytes=prep.FLOOR,
        split='train',official_split_unchanged=True,training_allowed=False,source_raw_sha256=prep.digest(raw),
        git_commit=prep.subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.version)
    prep.write(run/'remaining-legacy-budget.json',budget);print(json.dumps(budget),flush=True)
    if not a.execute:return
    prep.guard(run,byte_count+conversion+prep.GIB)
    if target.exists():raise FileExistsError('preserve previous preparation')
    target.mkdir()
    for name,value in buffers.items():(target/name).write_bytes(value)
    job=dict(key=key,source_id=1,sequence_id=len(original),source_sequence=source_sequence,sequence_folder=str(target),
        raw_path=str(raw),objects=[x[0] for x in meshes],windows=dict(train=str(target/'candidate-train.npy')),window_counts=dict(train=len(windows)))
    meta=dict(key=key,sequence_id=len(original),source_id=1,source_sequence=source_sequence,split='train',status='CONVERT',frames=n,
        objects=1,source=str(target/'hand.npy'),packet=str(packet_path),conversion_bytes=conversion,scope='all local non-TEST mother census; appended without rewriting frozen39526 IDs')
    prep.write(run/'additional-local-jobs.json',dict(status='COMPLETED',plan_sha256=prep.digest(prep.PLAN),rows=[dict(job=job,inventory=meta)],source_raw_sha256=prep.digest(raw)))
    print(json.dumps(dict(status='PREPARED_FOR_SAME_ADMISSION',parent=source_sequence,sequence_id=len(original))),flush=True)


if __name__=='__main__':
    sys.path.insert(0,str(prep.TASK.parent/'ptv3-shared-wam-v5/tools/run'))
    main()
