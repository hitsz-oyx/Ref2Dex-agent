#!/usr/bin/env python3
"""Bounded authorized pose/CAD download; never select video/MPS groups."""
import argparse
import json
from pathlib import Path
import time
import zipfile

import requests


def download(url, path, expected, deadline):
    if path.exists():
        if path.stat().st_size!=expected:raise ValueError('existing input length differs')
        return
    error=None
    for attempt in range(3):
        if time.monotonic()>deadline:raise TimeoutError('download deadline')
        session=requests.Session();session.trust_env=False
        if attempt:session.proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}
        try:
            with session.get(url,stream=True,timeout=(15,30)) as response:
                if response.status_code!=200:raise RuntimeError('HTTP'+str(response.status_code))
                with path.with_suffix(path.suffix+'.partial').open('wb') as stream:
                    size=0
                    for part in response.iter_content(1024*1024):
                        if time.monotonic()>deadline:raise TimeoutError('download deadline')
                        size+=len(part)
                        if size>expected:raise ValueError('download length exceeds manifest')
                        stream.write(part)
                if size!=expected:raise ValueError('incomplete input length')
            path.with_suffix(path.suffix+'.partial').replace(path)
            return
        except requests.RequestException:
            error='network request failed; signed URL intentionally omitted'
        finally:session.close()
    raise RuntimeError(error or 'download failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aria-manifest',type=Path,required=True)
    p.add_argument('--quest-manifest',type=Path,required=True)
    p.add_argument('--assets-manifest',type=Path,required=True)
    p.add_argument('--aria-sequence',default='P0001_10a27bf7')
    p.add_argument('--quest-sequence',default='P0002_1464cbdc')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-mib',type=int,default=512)
    p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--resume',action='store_true',help='resume incomplete input download without replacing files')
    a=p.parse_args();root=a.output.resolve()
    if root.exists() and (not a.resume or (root/'download-summary.json').exists()):
        raise FileExistsError('fresh input scope or explicit incomplete --resume required')
    selected=[];summary={}
    for kind,path,sequence in [('aria',a.aria_manifest,a.aria_sequence),('quest',a.quest_manifest,a.quest_sequence),
                              ('assets',a.assets_manifest,'assets')]:
        data=json.loads(path.read_text())
        groups=('assets',) if kind=='assets' else ('ground_truth','hand_data')
        summary[kind]={group:sum(row[group]['file_size_bytes'] for row in data['sequences'].values()
                               if group in row) for group in groups}
        for group in groups:
            entry=data['sequences'][sequence][group]
            selected.append((kind,sequence,group,entry))
    total=sum(entry['file_size_bytes'] for *_,entry in selected)
    if not 1<=a.max_mib<=1024 or total>a.max_mib*2**20 or not 30<=a.seconds<=1800:
        raise ValueError('bounded geometry-only input budget')
    root.mkdir(parents=True,exist_ok=a.resume);(root/'archives').mkdir(exist_ok=a.resume)
    deadline=time.monotonic()+a.seconds;began=time.monotonic();files=[];expanded=0
    for kind,sequence,group,entry in selected:
        path=root/'archives'/entry['filename']
        download(entry['download_url'],path,entry['file_size_bytes'],deadline)
        dest=root/'native'/kind/sequence;dest.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():continue
                name=Path(info.filename)
                if name.is_absolute() or '..' in name.parts:raise ValueError('unsafe archive member')
                expanded+=info.file_size
                if expanded>1024*2**20:raise ValueError('1GiB extraction budget')
                target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    # Both hand_data and ground_truth contain the same license.
                    # Resume retains every published native file byte-for-byte.
                    if target.read_bytes()!=archive.read(info):
                        raise ValueError('existing native member differs; refuse overwrite')
                    continue
                with target.open('xb') as out,archive.open(info) as source:
                    while True:
                        chunk=source.read(1024*1024)
                        if not chunk:break
                        out.write(chunk)
        files.append(dict(kind=kind,sequence=sequence,group=group,filename=path.name,bytes=path.stat().st_size))
        print(kind,sequence,group,'downloaded',path.stat().st_size,flush=True)
        (root/'progress.json').write_text(json.dumps(dict(files=files,elapsed_s=time.monotonic()-began)))
    (root/'download-summary.json').write_text(json.dumps(dict(status='COMPLETE',files=files,
        full_geometry_group_bytes=summary,total_downloaded_bytes=total,total_extracted_bytes=expanded,
        elapsed_s=time.monotonic()-began,training_allowed=False,
        user_authorized_manifests=[str(x) for x in (a.aria_manifest,a.quest_manifest,a.assets_manifest)]),indent=2)+'\n')
    print('Complete',total,'bytes, elapsed',time.monotonic()-began,flush=True)


if __name__=='__main__':main()
