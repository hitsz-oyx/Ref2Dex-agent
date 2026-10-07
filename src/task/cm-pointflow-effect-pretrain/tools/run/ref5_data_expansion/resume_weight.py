"""Resume a pinned official model with bounded parallel HTTP byte ranges.

The existing prefix must belong to the stopped downloader. Append only whole
ordered batches, keep a contiguous resumable prefix, and verify official LFS
SHA256 or a stable strong content ETag before publishing the checkpoint.
Record local SHA256 in both cases; an ETag is not a provider SHA256.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import uuid

import requests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True)
    identity=p.add_mutually_exclusive_group(required=True)
    identity.add_argument('--sha256',help='official full-file LFS SHA256')
    identity.add_argument('--etag',help='strong official HTTP ETag when no provider SHA256 is published')
    p.add_argument('--size',type=int,required=True)
    p.add_argument('--partial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seconds',type=int,default=1800)
    a=p.parse_args()
    if not 1<=a.workers<=4 or not 0<a.size<=6*1024**3:
        p.error('workers=1..4 and model size<=6 GiB required')
    if (a.sha256 and len(a.sha256)!=64) or a.output.exists() or (a.etag and a.etag.startswith('W/')):
        p.error('require a stable official identity and a fresh final checkpoint path')
    begin=time.monotonic();prefix=a.partial.stat().st_size
    if not 0<=prefix<=a.size:raise ValueError('invalid partial prefix size')
    part=a.output.with_suffix(a.output.suffix+'.part')
    if a.partial.resolve()!=part.resolve():
        if part.exists():raise FileExistsError('another partial download exists')
        a.partial.rename(part)
    chunk=16*1024**2
    def fetch(bounds):
        lo,hi=bounds
        if time.monotonic()-begin>a.seconds:raise TimeoutError('download deadline')
        join='&' if '?' in a.url else '?'
        url=a.url+join+'ref5_range='+uuid.uuid4().hex
        for attempt in range(3):
            try:
                headers={'Range':f'bytes={lo}-{hi}'}
                if a.etag:headers['If-Range']=a.etag
                with requests.get(url,headers=headers,stream=True,
                                  timeout=(20,90)) as r:
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{a.size}':
                        raise ValueError(f'wrong HTTP range: {r.status_code} {r.headers.get("Content-Range")}')
                    if a.etag and r.headers.get('ETag')!=a.etag:
                        raise ValueError('official content ETag changed')
                    parts=[]
                    for payload in r.iter_content(1024**2):
                        if time.monotonic()-begin>a.seconds:raise TimeoutError('download deadline')
                        parts.append(payload)
                    data=b''.join(parts)
                    if len(data)!=hi-lo+1:raise ValueError('short byte range')
                    return data
            except requests.RequestException:
                if attempt==2:raise
    # Verify a small part of the pre-existing prefix against the same source.
    if prefix:
        for lo in [0,max(0,prefix-1024)]:
            hi=min(prefix-1,lo+1023)
            with part.open('rb') as f:f.seek(lo);old=f.read(hi-lo+1)
            if old!=fetch((lo,hi)):raise ValueError('existing prefix differs from pinned model')
    print(json.dumps(dict(status='RESUMING',prefix=prefix,size=a.size)),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool,part.open('ab') as f:
        while prefix<a.size:
            jobs=[(lo,min(lo+chunk-1,a.size-1)) for lo in
                  range(prefix,min(prefix+a.workers*chunk,a.size),chunk)]
            data=list(pool.map(fetch,jobs))
            for payload in data:f.write(payload)
            f.flush();prefix=part.stat().st_size
            print(json.dumps(dict(bytes=prefix,total=a.size,elapsed_s=time.monotonic()-begin)),flush=True)
    h=hashlib.sha256()
    with part.open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    if a.sha256 and h.hexdigest()!=a.sha256:raise ValueError('official model SHA256 mismatch; partial retained')
    part.rename(a.output)
    report=dict(status='COMPLETED',url=a.url,bytes=prefix,sha256=h.hexdigest(),
                official_sha256=a.sha256,official_etag=a.etag,
                workers=a.workers,elapsed_s=time.monotonic()-begin)
    a.output.with_suffix(a.output.suffix+'.download.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
