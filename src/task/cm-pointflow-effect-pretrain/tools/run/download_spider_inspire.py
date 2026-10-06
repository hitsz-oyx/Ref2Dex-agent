#!/usr/bin/env python3
"""Fetch a pinned, checksummed Inspire subset and its exact XML dependencies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import time
import subprocess
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

SOURCES = ('dexycb_lifted', 'hot3d_v2', 'hrdexdb_24f', 'oakink_lifted')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--per-source', type=int, default=1, help='0 selects all Inspire')
    p.add_argument('--max-bytes', type=int, default=2 * 1024**3)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--transport', choices=('urllib','curl'), default='curl')
    a = p.parse_args()
    root = a.run_dir.resolve(); start = time.monotonic()
    revision = json.loads((root/'dataset_info.json').read_text())['sha']
    checks = json.loads((root/'checksums.json').read_text())
    out = root/'dataset'; out.mkdir(exist_ok=True)
    records = {}; total = 0
    local_by_hash = {}
    for existing in out.rglob('*'):
        if existing.is_file():
            rel = existing.relative_to(out).as_posix()
            if rel in checks and hashlib.sha256(existing.read_bytes()).hexdigest() == checks[rel]:
                local_by_hash[checks[rel]] = existing
    def fetch(rel):
        nonlocal total
        rel = posixpath.normpath(rel)
        if rel not in checks or rel.startswith('../') or PurePosixPath(rel).is_absolute():
            raise ValueError('Unlisted or unsafe dependency: '+rel)
        if time.monotonic()-start > a.seconds:
            raise TimeoutError('download deadline')
        target = out/rel; target.parent.mkdir(parents=True, exist_ok=True)
        # A content identity is mandatory on resume, too.
        if target.exists():
            data = target.read_bytes()
        elif checks[rel] in local_by_hash:
            data = local_by_hash[checks[rel]].read_bytes()
        else:
            url = 'https://huggingface.co/datasets/retarget/retarget_full/resolve/'+revision+'/'+rel
            if a.transport == 'curl':
                transfer = target.with_name(target.name+'.download')
                subprocess.run(['curl','--silent','--show-error','--fail','--location',
                                '--retry','2','--connect-timeout','15','--max-time','60',
                                '--max-filesize',str(a.max_bytes),'--output',str(transfer),url],
                               check=True,timeout=min(200,max(1,a.seconds-(time.monotonic()-start))))
                data = transfer.read_bytes()
                transfer.unlink()
            else:
                data = fetch_urllib(url, rel)
            if len(data) > a.max_bytes:
                raise ValueError('individual file exceeds cap')
        digest = hashlib.sha256(data).hexdigest()
        if digest != checks[rel]: raise ValueError('checksum mismatch: '+rel)
        return rel, data, digest
    def fetch_urllib(url, rel):
        for attempt in range(3):
            try:
                request_url = url+'?download=true&attempt='+str(time.time_ns())
                with urllib.request.urlopen(request_url, timeout=45) as response:
                    data = response.read(a.max_bytes+1)
                break
            except (urllib.error.URLError, TimeoutError) as error:
                print(json.dumps(dict(path=rel,attempt=attempt+1,error=str(error))),flush=True)
                if attempt == 2 or time.monotonic()-start > a.seconds: raise
                time.sleep(1+attempt)
        return data
    def fetch_batch(paths):
        nonlocal total
        paths = sorted(set(paths)-records.keys())
        with ThreadPoolExecutor(max_workers=8) as pool:
            for rel, data, digest in pool.map(fetch, paths):
                total += len(data)
                if total > a.max_bytes: raise ValueError('subset storage cap')
                target = out/rel
                if not target.exists(): target.write_bytes(data)
                records[rel] = dict(bytes=len(data),sha256=digest)
        print(json.dumps(dict(files=len(records),bytes=total,elapsed_seconds=time.monotonic()-start)),flush=True)
    selected = {}
    for source in SOURCES:
        tasks = sorted(p.rsplit('/0/',1)[0] for p in checks
                       if p.startswith('processed/'+source+'/inspire/right/') and p.endswith('/0/trajectory_mjwp.npz'))
        selected[source] = tasks[:a.per_source] if a.per_source else tasks
    ordinary = [p for p in checks if any(p.startswith(t+'/') for tasks in selected.values() for t in tasks)
                and not p.endswith(('.mp4','.webm'))]
    record = dict(status='RUNNING',revision=revision,selection=selected,per_source=a.per_source,transport=a.transport,
                  max_bytes=a.max_bytes,budget_seconds=a.seconds)
    record['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    record['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stem = 'download_samples' if a.per_source else 'download_inspire'
    manifest = root/(stem+'.json'); attempt_id = 1
    while manifest.exists():
        attempt_id += 1; manifest = root/(stem+'-r'+str(attempt_id)+'.json')
    try:
        fetch_batch(ordinary)
        xml_done = set()
        while True:
            pending = [p for p in records if p.endswith('.xml') and p not in xml_done]
            if not pending: break
            dependencies = []
            for rel in pending:
                xml_done.add(rel); tree = ET.parse(out/rel)
                compiler = tree.getroot().find('compiler')
                for element in tree.iter():
                    file = element.get('file')
                    if not file: continue
                    directory = ''
                    if compiler is not None:
                        if element.tag == 'mesh': directory = compiler.get('meshdir','')
                        elif element.tag == 'texture': directory = compiler.get('texturedir','')
                    dependencies.append(posixpath.normpath(posixpath.join(posixpath.dirname(rel),directory,file)))
            fetch_batch(dependencies)
        record['status']='COMPLETED'
    except BaseException as exc:
        record.update(status='FAILED',error=str(exc));raise
    finally:
        record.update(files=records,total_bytes=total,elapsed_seconds=time.monotonic()-start)
        manifest.write_text(json.dumps(record,indent=2)+'\n')

if __name__ == '__main__': main()
