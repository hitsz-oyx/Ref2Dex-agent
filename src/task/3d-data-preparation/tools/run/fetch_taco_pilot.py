#!/usr/bin/env python3
"""Extract eight public official ZIP members by exact HTTP ranges."""
import argparse
import json
from pathlib import Path
import struct
import time
import zlib
import requests

BASE='https://www.dropbox.com/scl/fo/6wux06w26exuqt004eg1a/'
SUFFIX='?rlkey=e76q06hyj9yqbahhipmf5ij1o&dl=1'
ARCHIVES={
    'hand':('ABNVTswzgHl6iczNcZW033Y/Hand_Poses.zip',15231790511),
    'pose':('AMtatHMOYfCbwEYQbu-Vfl4/Object_Poses.zip',34340303),
    'mesh':('AJLl79Ae4czOVbdaOlISXhU/Object_Models.zip',424209827)}
MEMBERS=[('hand',260,9543900,10091010,'left_hand.pkl'),
    ('hand',9544275,9521699,10091010,'right_hand.pkl'),
    ('hand',19066090,325,445,'right_hand_shape.pkl'),
    ('hand',19066537,326,445,'left_hand_shape.pkl'),
    ('pose',266,10040,14272,'tool_164.npy'),('pose',10422,9356,14272,'target_057.npy'),
    ('mesh',305349901,2395985,7049105,'164_cm.obj'),('mesh',370463868,2546786,7092197,'057_cm.obj')]


def get_range(url,lo,hi,total,deadline):
    for attempt in range(3):
        if time.monotonic()>deadline:raise TimeoutError('range deadline')
        session=requests.Session();session.trust_env=False
        if attempt:session.proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}
        try:
            with session.get(url,headers={'Range':'bytes=%d-%d'%(lo,hi)},stream=True,timeout=(15,30)) as response:
                if response.status_code!=206 or response.headers.get('Content-Range')!='bytes %d-%d/%d'%(lo,hi,total):
                    raise ValueError('official ZIP changed or server ignored exact range')
                chunks=[];received=0
                for chunk in response.iter_content(1024*1024):
                    if time.monotonic()>deadline:raise TimeoutError('range deadline')
                    received+=len(chunk)
                    if received>hi-lo+1:raise ValueError('range length budget')
                    chunks.append(chunk)
                if received!=hi-lo+1:raise IOError('incomplete range')
                return b''.join(chunks)
        except requests.RequestException:
            if attempt==2:raise RuntimeError('TACO range network request failed') from None
        finally:session.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--resume',action='store_true',help='reuse only same-size members of an incomplete pilot')
    a=p.parse_args();root=a.output.resolve()
    if root.exists() and (not a.resume or (root/'download-summary.json').exists()):
        raise FileExistsError('fresh directory or explicitly resumed incomplete pilot required')
    if not 30<=a.seconds<=1200:raise ValueError('bounded download deadline')
    root.mkdir(parents=True,exist_ok=a.resume);began=time.monotonic();deadline=began+a.seconds;rows=[]
    previous=json.loads((root/'progress.json').read_text())['files'] if a.resume and (root/'progress.json').exists() else []
    for key,offset,compressed,expanded,name in MEMBERS:
        reused=next((r for r in previous if r['filename']==name and r['expanded_bytes']==expanded),None)
        if reused and (root/name).stat().st_size==expanded:
            rows.append(reused);print(name,'reused',expanded,flush=True);continue
        url,total=ARCHIVES[key];url=BASE+url+SUFFIX
        header=get_range(url,offset,offset+29,total,deadline)
        fields=struct.unpack('<4s5H3I2H',header)
        if fields[0]!=b'PK\x03\x04' or fields[3]!=8:raise ValueError('expected official DEFLATE local header')
        name_size,extra_size=fields[-2:];start=offset+30+name_size+extra_size
        filename=get_range(url,offset+30,offset+29+name_size,total,deadline).decode()
        if Path(filename).name!=name:raise ValueError('official member identity changed')
        payload=get_range(url,start,start+compressed-1,total,deadline)
        raw=zlib.decompress(payload,wbits=-15)
        if len(raw)!=expanded:raise ValueError('official member expanded length changed')
        (root/name).write_bytes(raw)
        rows.append(dict(filename=name,zip_member=filename,archive=key,compressed_bytes=compressed,expanded_bytes=expanded))
        print(name,'fetched',compressed,'expanded',expanded,flush=True)
        (root/'progress.json').write_text(json.dumps(dict(files=rows,elapsed_s=time.monotonic()-began)))
    (root/'download-summary.json').write_text(json.dumps(dict(status='COMPLETE',triplet='(dust, roller, pan)',
        sequence='20230927_032',files=rows,total_payload_bytes=sum(x['compressed_bytes'] for x in rows),
        expanded_bytes=sum(x['expanded_bytes'] for x in rows),elapsed_s=time.monotonic()-began,
        source='official TACO V1 Dropbox backup; exact byte range and member lengths',training_allowed=False),indent=2)+'\n')


if __name__=='__main__':main()
