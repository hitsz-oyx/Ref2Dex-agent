#!/usr/bin/env python3
"""Pinned OakInk2 annotation-only acquisition for a fixed 100-sequence audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--sequences',type=int,default=100)
    p.add_argument('--seconds',type=int,default=1800)
    p.add_argument('--max-bytes',type=int,default=12*1024**3)
    a=p.parse_args();root=a.run_dir.resolve();start=time.monotonic()
    revision=json.loads((root/'dataset_info.json').read_text())['sha']
    ann=sorted(json.loads((root/'anno_tree.json').read_text()),key=lambda x:x['path'])
    if not 1<=a.sequences<=len(ann):raise ValueError('wrong sequence count')
    select=[ann[round(i*(len(ann)-1)/(a.sequences-1))] for i in range(a.sequences)] if a.sequences>1 else ann[:1]
    names={'object_raw.tar','object_repair.tar','object_affordance.tar','program.tar','program_extension.tar'}
    assets=[x for x in json.loads((root/'root_tree.json').read_text()) if x['path'] in names]
    jobs=assets+select;expected=sum(x['size'] for x in jobs)
    if expected>a.max_bytes or shutil.disk_usage(root).free<expected+10*1024**3:
        raise ValueError('storage budget/reserve violated')
    out=root/'download';out.mkdir(exist_ok=True)
    manifest=root/'download_manifest.json'
    if manifest.exists():raise FileExistsError('use a fresh manifest, preserve earlier run')
    record=dict(status='RUNNING',revision=revision,selection=[x['path'] for x in select],
                files={},expected_bytes=expected,budget_seconds=a.seconds,byte_cap=a.max_bytes,
                script_sha256=digest(Path(__file__)),git_status='UNAVAILABLE: missing original git common directory')
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    def fetch(item):
        name=item['path'];path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        sha=item['lfs']['oid']
        if path.exists():
            if path.stat().st_size!=item['size'] or digest(path)!=sha:raise ValueError('existing bytes mismatch: '+name)
        else:
            part=path.with_name(path.name+'.part')
            timeout=min(300,max(1,a.seconds-(time.monotonic()-start)))
            url='https://huggingface.co/datasets/kelvin34501/OakInk-v2/resolve/'+revision+'/'+name
            subprocess.run(['curl','--fail','--silent','--show-error','--location','--retry','2',
                            '--connect-timeout','15','--max-time','240','--max-filesize',str(item['size']),
                            '--output',str(part),url],check=True,timeout=timeout)
            if part.stat().st_size!=item['size'] or digest(part)!=sha:raise ValueError('download checksum mismatch: '+name)
            part.rename(path)
        return name,dict(size=path.stat().st_size,sha256=sha)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending=[pool.submit(fetch,x) for x in jobs]
            for future in as_completed(pending):
                name,info=future.result();record['files'][name]=info
                record['elapsed_seconds']=time.monotonic()-start
                manifest.write_text(json.dumps(record,indent=2)+'\n')
                print(json.dumps(dict(completed=len(record['files']),total=len(jobs),
                                      bytes=sum(x['size'] for x in record['files'].values()),
                                      elapsed_seconds=record['elapsed_seconds'])),flush=True)
        record['status']='COMPLETED'
    except BaseException as exc:record.update(status='FAILED',error=str(exc));raise
    finally:
        record['elapsed_seconds']=time.monotonic()-start;manifest.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
