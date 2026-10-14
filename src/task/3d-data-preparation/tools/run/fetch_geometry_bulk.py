"""Resumable authorized geometry acquisition; signed URLs never leave memory."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time
import zipfile
from urllib.parse import urlsplit
import requests

STOP = threading.Event()
LOCK = threading.Lock()
EXTRACT_LOCK = threading.Lock()
NETWORK_BYTES = 0


def write_json(path, data):
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(data, indent=2)+'\n'); temp.replace(path)


def safe_filename(name):
    p = Path(name)
    if p.name != name or name in ('', '.', '..'):
        raise ValueError('unsafe input filename')
    return name


def fetch(entry, root, deadline, progress, auth=None):
    global NETWORK_BYTES
    path = root/'archives'/safe_filename(entry['filename'])
    expected = entry['bytes']
    if path.exists():
        if path.stat().st_size != expected:
            raise ValueError('published input length changed')
        return path
    partial = path.with_suffix(path.suffix+'.partial')
    for attempt in range(8):
        if STOP.is_set() or time.monotonic() >= deadline:
            raise TimeoutError('stopped/deadline; partial input retained')
        if shutil.disk_usage(root).free < 20*2**30:
            STOP.set(); raise RuntimeError('physical disk reserve20GiB reached')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise ValueError('partial exceeds declared source size')
        if offset == expected:
            partial.replace(path); return path
        session = requests.Session(); session.trust_env = False
        if auth is not None:
            parsed = urlsplit(entry['url'])
            if parsed.scheme != 'https' or parsed.hostname != 'h2odataset.ethz.ch':
                raise ValueError('Basic credentials restricted to HTTPS H2O origin')
            session.auth = auth
        if attempt % 2:
            session.proxies = {'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}
        try:
            headers = {'Range':f'bytes={offset}-'} if offset else {}
            with session.get(entry['url'], headers=headers, stream=True, timeout=(15,30)) as response:
                if response.status_code in (401,403):
                    raise RuntimeError('authorization/link failed HTTP'+str(response.status_code))
                if offset:
                    wanted = f'bytes {offset}-{expected-1}/{expected}'
                    if response.status_code != 206 or response.headers.get('Content-Range') != wanted:
                        raise ValueError('server did not honor resumed range')
                elif response.status_code != 200:
                    raise IOError('HTTP'+str(response.status_code))
                size = offset
                with partial.open('ab') as stream:
                    for chunk in response.iter_content(1024*1024):
                        if STOP.is_set() or time.monotonic() >= deadline:
                            raise TimeoutError('stopped/deadline')
                        size += len(chunk)
                        if size > expected:
                            raise ValueError('received more than declared source size')
                        stream.write(chunk)
                        with LOCK:
                            progress[entry['filename']] = size
                            NETWORK_BYTES += len(chunk)
                        if shutil.disk_usage(root).free < 20*2**30:
                            STOP.set(); raise RuntimeError('physical disk reserve20GiB reached')
                if size != expected:
                    raise IOError('incomplete input')
            partial.replace(path); return path
        except (requests.RequestException, IOError):
            if attempt == 7:
                raise RuntimeError('network failed after8 attempts; partial retained; URL omitted') from None
        finally:
            session.close()


def extract(path, destination):
    total = 0
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = Path(member.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in member.filename:
                raise ValueError('unsafe native archive member')
            if member.is_dir():
                continue
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            total += member.file_size
            if target.exists():
                if target.stat().st_size != member.file_size or target.read_bytes() != archive.read(member):
                    raise ValueError('existing native file differs; never overwrite')
                continue
            temporary = target.with_suffix(target.suffix+'.extracting')
            with temporary.open('wb') as out, archive.open(member) as source:
                while True:
                    if STOP.is_set() or shutil.disk_usage(destination).free < 20*2**30:
                        raise RuntimeError('native extraction stopped; partial retained')
                    chunk = source.read(1024*1024)
                    if not chunk:
                        break
                    out.write(chunk)
            temporary.replace(target)
    return total


def entries(args):
    if args.dataset == 'h2o':
        preflight = json.loads((args.output/'access-preflight.json').read_text())['files']
        expected = {'object.zip', 'label_split.zip'} | {f'subject{i}_pose_v1_1.tar.gz' for i in range(1,5)}
        if {r['filename'] for r in preflight} != expected or len(preflight) != 6:
            raise ValueError('six authorized H2O geometry archives required')
        if any(r['http_status'] != 200 or r['bytes'] <= 0 for r in preflight):
            raise ValueError('H2O access preflight failed')
        return [dict(family='h2o',sequence='all',group='geometry',filename=r['filename'],
            bytes=r['bytes'],url='https://h2odataset.ethz.ch/data/dataset/'+safe_filename(r['filename'])) for r in preflight]
    if args.dataset == 'hot3d':
        result = []
        for family, filename in [('aria','Hot3DAria_download_urls.json'),
                                 ('quest','Hot3DQuest_download_urls.json'),('assets','Hot3DAssets_download_urls.json')]:
            data = json.loads((args.repo/filename).read_text())
            for sequence, groups in data['sequences'].items():
                for group in (['assets'] if family == 'assets' else ['ground_truth','hand_data']):
                    if group in groups:
                        r = groups[group]
                        result.append(dict(family=family, sequence=sequence, group=group,
                            filename=r['filename'], bytes=r['file_size_bytes'], url=r['download_url']))
        return result
    from fetch_taco_pilot import BASE, SUFFIX, ARCHIVES
    return [dict(family='taco', sequence='all', group=key, filename=Path(path).name,
                 bytes=size, url=BASE+path+SUFFIX) for key,(path,size) in ARCHIVES.items()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['hot3d','taco','h2o'], required=True)
    p.add_argument('--credential-file', type=Path)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse', type=Path)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seconds', type=int, default=86400)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    auth = None
    if a.dataset == 'h2o' and not a.plan_only:
        if a.credential_file is None or a.credential_file.stat().st_mode & 0o077:
            p.error('H2O requires private mode0600 credential file')
        credential = json.loads(a.credential_file.read_text())
        auth = (credential['username'], credential['password'])
    elif a.credential_file is not None:
        p.error('credentials are only supported for H2O downloads')
    if not 1 <= a.workers <= 8 or not 60 <= a.seconds <= 86400:
        p.error('bounded workers/deadline required')
    if a.plan_only:
        selected = entries(a)
        print(json.dumps(dict(dataset=a.dataset, files=len(selected),
                             bytes=sum(r['bytes'] for r in selected)), indent=2))
        return
    root = a.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    (root/'archives').mkdir(exist_ok=True)
    selected = entries(a)
    public = [{k:v for k,v in row.items() if k != 'url'} for row in selected]
    if (root/'input-manifest.json').exists():
        if json.loads((root/'input-manifest.json').read_text()) != public:
            raise ValueError('resume source identity changed')
    else:
        write_json(root/'input-manifest.json',public)
    if a.reuse:
        for r in selected:
            source = a.reuse/'archives'/r['filename']; target = root/'archives'/r['filename']
            if source.is_file() and source.stat().st_size == r['bytes'] and not target.exists():
                target.symlink_to(source.resolve())
    signal.signal(signal.SIGUSR1,lambda *_:STOP.set())
    began = time.monotonic(); deadline = began+a.seconds; done=[]; errors=[]; progress={}
    total = sum(r['bytes'] for r in selected)
    def publish(status):
        with LOCK:
            received = sum(progress.values())
        elapsed = time.monotonic()-began
        byte_done = sum(r['bytes'] for r in done)
        write_json(root/'acquisition-status.json',dict(status=status,pid=os.getpid(),dataset=a.dataset,
            total_files=len(selected),completed_files=len(done),completed_bytes=byte_done,
            active_received_bytes=received,total_bytes=total,elapsed_s=elapsed,
            network_bytes=NETWORK_BYTES,payload_bytes_per_s=NETWORK_BYTES/max(elapsed,1),errors=errors,
            checked_at=time.time(),training_allowed=False))
    def one(row):
        path = fetch(row,root,deadline,progress,auth=auth)
        expanded = 0
        if a.dataset == 'hot3d':
            with EXTRACT_LOCK:
                expanded = extract(path,root/'native'/row['family']/row['sequence'])
        return {**{k:v for k,v in row.items() if k != 'url'},'expanded_bytes':expanded}
    publish('RUNNING')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(one,r):r for r in selected}
        pending=set(futures)
        from concurrent.futures import wait, FIRST_COMPLETED
        while pending:
            completed,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for f in completed:
                row=futures[f]
                try:
                    result=f.result();done.append(result)
                    print('COMPLETE',row['filename'],row['bytes'],flush=True)
                except Exception as error:
                    # Never serialize the exception text: requests may include signed URLs.
                    errors.append(dict(filename=row['filename'],error_type=type(error).__name__))
                    print('FAILED',row['filename'],type(error).__name__,flush=True)
                with LOCK:
                    progress.pop(row['filename'],None)
            publish('STOPPING' if STOP.is_set() else 'RUNNING')
    status='STOPPED' if STOP.is_set() else ('PARTIAL' if errors else 'COMPLETE')
    publish(status);write_json(root/'download-summary.json',dict(status=status,files=done,errors=errors,
        total_downloaded_bytes=sum(r['bytes'] for r in done),total_extracted_bytes=sum(r['expanded_bytes'] for r in done),
        elapsed_s=time.monotonic()-began,training_allowed=False))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
