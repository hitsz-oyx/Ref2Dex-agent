"""Reuse downloaded official assets; process TRAIN meshes in isolated GPU batches."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[5]
OLD = REPO/'src/task/video-data-preparation'
spec = importlib.util.spec_from_file_location('official_assets',OLD/'tools/run/prepare_unidexgrasp_assets.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path, value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def prepare(a):
    root=a.output; shared=root/'shared-assets';inputs=a.inputs
    shared.mkdir(parents=True,exist_ok=True)
    train,_=module.split_file(inputs,'train_set.yaml',time.monotonic()+60)
    seen,_=module.split_file(inputs,'test_set_seen_cat.yaml',time.monotonic()+60)
    unseen,_=module.split_file(inputs,'test_set_unseen_cat.yaml',time.monotonic()+60)
    existing=json.loads(a.existing_manifest.read_text())
    previous={r['object_code']:r['split'] for r in existing['objects']}
    selected=[]
    for code in sorted(set(train)-set(seen)-set(unseen)-set(previous)):
        selected.append(dict(object_code=code,scale=float(train[code][0]),official_split='TRAIN',
            split='dev' if (len(selected)+1)%5==0 else 'train'))
    if a.limit:
        selected=selected[:a.limit]
    if (root/'selection.json').exists():
        old=json.loads((root/'selection.json').read_text())
        if old['objects']!=selected:raise ValueError('resume selection changed')
    else:
        write(root/'selection.json',dict(objects=selected,selection='sorted official TRAIN; TEST and existing20 excluded; no result selection',
            total_eligible=len(set(train)-set(seen)-set(unseen)),existing_meshes=len(previous)))
    for folder in ('mjcf','textures'):
        if not (shared/folder).exists():
            shutil.copytree(a.vendor/'assets'/folder,shared/folder)
    chosen={r['object_code'] for r in selected}
    if not (shared/'mesh-extraction.json').exists():
        extracted=0
        with tarfile.open(inputs/'meshdatav3_scaled.tar.xz','r|xz') as archive:
            for entry in archive:
                parts=module.safe_name(entry.name).parts
                if not entry.isfile() or len(parts)<4 or parts[0]!='meshdatav3_scaled' or '/'.join(parts[1:3]) not in chosen:
                    continue
                if shutil.disk_usage(root).free<20*2**30:raise RuntimeError('physical reserve reached')
                dest=shared/entry.name;dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():
                    if dest.stat().st_size!=entry.size:raise ValueError('native mesh length differs')
                    continue
                partial=dest.with_suffix(dest.suffix+'.partial')
                with archive.extractfile(entry) as source,partial.open('wb') as target:
                    shutil.copyfileobj(source,target,1024*1024)
                partial.replace(dest);extracted+=entry.size
        write(shared/'mesh-extraction.json',dict(new_bytes=extracted,meshes=len(selected)))
    all_priors=np.load(inputs/'datasetv4.1_posedata.npy',allow_pickle=True).item()
    widths=dict(target_qpos=22,target_hand_pos=3,target_hand_rot=4,object_euler_xy=2,object_init_z=1)
    batches=[]
    with zipfile.ZipFile(inputs/'meshdatav3_pc_feat.zip') as archive:
        for offset in range(0,len(selected),a.batch_size):
            rows=selected[offset:offset+a.batch_size];batch=root/f'batch{offset//a.batch_size:03d}'
            batch.mkdir(exist_ok=True);assets=batch/'assets';assets.mkdir(exist_ok=True);priors={}
            if (batch/'manifest.json').exists():
                batches.append(batch);continue
            for folder in ('mjcf','textures','meshdatav3_scaled'):
                if not (assets/folder).exists():(assets/folder).symlink_to(shared/folder,target_is_directory=True)
            for row in rows:
                code,scale=row['object_code'],row['scale'];prior={k:np.asarray(all_priors[code][scale][k],np.float32) for k in widths}
                count=len(prior['object_init_z'])
                if not count or any(v.shape!=(count,widths[k]) or not np.isfinite(v).all() for k,v in prior.items()):
                    raise ValueError('official initialization mismatch')
                for name in ('coacd_%03d.urdf','decomposed_%03d.obj'):
                    if not (shared/'meshdatav3_scaled'/code/'coacd'/(name%round(scale*100))).is_file():
                        raise ValueError('selected official mesh missing')
                feature='meshdatav3_pc_feat/'+code+'/pc_feat_%03d.npy'%round(scale*100)
                target=assets/feature;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(feature))
                row['initialization_priors']=count;priors[code]={scale:prior}
            np.save(assets/'datasetv4.1_posedata.npy',priors)
            write(batch/'manifest.json',dict(status='COMPLETE',upstream_commit=module.PIN,objects=rows,asset_root=str(assets.resolve()),
                source_inputs=str(inputs),training_allowed=False))
            batches.append(batch)
    return batches


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--vendor',type=Path,required=True);p.add_argument('--existing-manifest',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=4);p.add_argument('--batch-size',type=int,default=25)
    p.add_argument('--limit',type=int);p.add_argument('--seconds',type=int,default=86400)
    a=p.parse_args()
    if not 1<=a.batch_size<=32 or not 60<=a.seconds<=86400:p.error('bounded batch/deadline')
    for field in ('output','inputs','vendor','existing_manifest'):setattr(a,field,getattr(a,field).resolve())
    a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();status=dict(status='PREPARING',pid=os.getpid(),gpu=a.gpu,
        checked_at=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        completed_batches=0,failed_batches=[],training_allowed=False)
    write(a.output/'progress.json',status)
    batches=prepare(a);status.update(status='COLLECTING',total_batches=len(batches));write(a.output/'progress.json',status)
    for batch in batches:
        if (a.output/'STOP').exists() or time.monotonic()-start>a.seconds:
            status['status']='STOPPED';break
        if shutil.disk_usage(a.output).free<20*2**30:
            status['status']='PHYSICAL_DISK_STOP';break
        run=batch/'replay'
        if (run/'trajectory-manifest.json').exists():
            status['completed_batches']+=1;continue
        rows=json.loads((batch/'manifest.json').read_text())['objects']
        if not (run/'result.json').exists():
            if run.exists():
                status['failed_batches'].append(dict(batch=batch.name,reason='incomplete prior run retained'));continue
            memory=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if memory>64:
                status['status']='GPU_BUSY';break
            command=[sys.executable,str(OLD/'tools/run/replay_unidexgrasp.py'),'--subset',str(REPO/'outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1'),
                '--vendor',str(a.vendor),'--object-manifest',str(batch/'manifest.json'),'--output',str(run),'--gpu',str(a.gpu),
                '--envs',str(len(rows)*8),'--episodes','2','--seconds','600','--gpu-seconds','300']
            status.update(active_batch=batch.name,checked_at=time.time());write(a.output/'progress.json',status)
            with (batch/'collector.log').open('ab') as out:
                result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,timeout=650,cwd=REPO)
            if result.returncode:
                status['failed_batches'].append(dict(batch=batch.name,reason='collector failed'));continue
        command=[sys.executable,str(OLD/'tools/audit/audit_unidexgrasp_replay.py'),'--run',str(run),'--render-objects','0','--no-window-export']
        with (batch/'audit.log').open('ab') as out:
            result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,timeout=600,cwd=REPO)
        if result.returncode:
            status['failed_batches'].append(dict(batch=batch.name,reason='native audit failed'))
        else:
            status['completed_batches']+=1
        status.update(checked_at=time.time(),elapsed_s=time.monotonic()-start);write(a.output/'progress.json',status)
        print(batch.name,status['completed_batches'],'/',len(batches),flush=True)
    else:
        status['status']='COMPLETE' if not status['failed_batches'] else 'PARTIAL'
    status.update(checked_at=time.time(),elapsed_s=time.monotonic()-start);write(a.output/'progress.json',status)


if __name__=='__main__':main()
