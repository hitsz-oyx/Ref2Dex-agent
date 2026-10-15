"""Fetch pinned image-free DexYCB archives; extract one camera's original labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import requests

REVISION='18376c6f7ab2f1cda6f5d316326a67f886171514'
MIRROR='Louis0411/dexycb_mesh'
CAMERA='836212060125'


def atomic(path,value):
    part=path.with_suffix('.json.part');part.write_text(json.dumps(value,indent=2)+'\n');part.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,11)))
    p.add_argument('--seconds',type=int,default=14400)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    assert 60<=a.seconds<=86400 and set(a.subjects)<=set(range(1,11))
    archives=root/'archives';raw=root/'raw';archives.mkdir(exist_ok=True);raw.mkdir(exist_ok=True)
    rows=[x for x in json.loads(a.catalog.read_text()) if any(x['path'].endswith(f'subject-{s:02d}.tar.gz') for s in a.subjects)]
    assert len(rows)==len(set(a.subjects)) and sum(x['size'] for x in rows)<2_000_000_000
    started=time.monotonic()
    def guard():
        if (root/'STOP').exists() or time.monotonic()-started>a.seconds or shutil.disk_usage(root).free<20*2**30:
            raise RuntimeError('STOP/deadline/20GiB free boundary')
    def work(row):
        name=Path(row['path']).name;subject=name[:-7];final=archives/name;part=archives/(name+'.part')
        receipt=root/(subject+'.json')
        if receipt.exists() and json.loads(receipt.read_text())['status']=='COMPLETED':return json.loads(receipt.read_text())
        state=dict(status='DOWNLOADING',subject=subject,bytes=0,expected_bytes=row['size'],revision=REVISION)
        for attempt in range(8):
            guard()
            if final.exists() and final.stat().st_size==row['size']:break
            host='https://hf-mirror.com' if attempt<2 else 'https://huggingface.co'
            url=f'{host}/datasets/{MIRROR}/resolve/{REVISION}/{row["path"]}?download=true&task_time={int(time.time())}'
            count=part.stat().st_size if part.exists() else 0
            if count>row['size']:raise ValueError('oversized partial')
            try:
                with requests.get(url,headers={'Range':f'bytes={count}-'} if count else {},stream=True,timeout=(20,40)) as response:
                    response.raise_for_status()
                    if count and response.status_code!=206:count=0
                    if response.status_code==206 and not response.headers.get('Content-Range','').startswith(f'bytes {count}-'):
                        raise ValueError('unexpected range response')
                    with part.open('ab' if count else 'wb') as handle:
                        last=0
                        for block in response.iter_content(1024*1024):
                            guard();count+=len(block)
                            if count>row['size']:raise ValueError('download exceeds catalog size')
                            handle.write(block)
                            if time.monotonic()-last>5:
                                state.update(bytes=count,elapsed_s=time.monotonic()-started);atomic(receipt,state);last=time.monotonic()
                    if count!=row['size']:raise ValueError('incomplete archive')
                    part.replace(final)
                    break
            except (requests.RequestException,ValueError) as e:
                state.update(last_error=type(e).__name__+': '+str(e).split(' for url:')[0],attempt=attempt+1)
                atomic(receipt,state)
                if attempt==7:raise
                time.sleep(2)
        else:raise RuntimeError('download exhausted')
        guard();state['status']='EXTRACTING';atomic(receipt,state);files=0;total=0
        with tarfile.open(final,'r:gz') as tar:
            for member in tar:
                guard();parts=Path(member.name).parts
                if member.isdir():continue
                if not member.isfile() or member.name.startswith('/') or '..' in parts:raise ValueError('unsafe archive member')
                if len(parts)<2 or parts[0]!=subject:raise ValueError('unexpected archive root')
                selected=(len(parts)==3 and parts[-1]=='meta.yml') or (len(parts)==4 and parts[-2]==CAMERA and parts[-1].startswith('labels_') and parts[-1].endswith('.npz'))
                if not selected:continue
                if member.size>2_000_000:raise ValueError('unexpected label size')
                dest=raw/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
                data=tar.extractfile(member).read();assert len(data)==member.size
                if dest.exists():
                    if dest.read_bytes()!=data:raise ValueError('existing extracted label differs')
                else:dest.write_bytes(data)
                files+=1;total+=member.size
                if total>150_000_000:raise ValueError('subject extracted budget exceeded')
        folders=sorted((raw/subject).iterdir());assert len(folders)==100
        state.update(status='COMPLETED',bytes=final.stat().st_size,files=files,extracted_bytes=total,sessions=len(folders),elapsed_s=time.monotonic()-started,camera=CAMERA,source=f'https://huggingface.co/datasets/{MIRROR}',upstream='https://dex-ycb.github.io/')
        atomic(receipt,state);print(json.dumps(state),flush=True);return state
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(work,rows))
        atomic(root/'progress.json',dict(status='COMPLETED',subjects=result,pid=os.getpid(),elapsed_s=time.monotonic()-started))
    except BaseException as e:
        atomic(root/'progress.json',dict(status='FAILED',reason=str(e).split(' for url:')[0],pid=os.getpid(),elapsed_s=time.monotonic()-started));raise


if __name__=='__main__':main()
