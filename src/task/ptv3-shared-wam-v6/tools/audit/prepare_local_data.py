"""Bounded local data inventory/preparation, without training or admission.

Original inputs stay read-only. Engineering windows are explicitly distinct
from the unavailable remote frozen window filter and normalization.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import zipfile

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
GIB = 2**30


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def capacity(root, peak_bytes, floor_bytes=20*GIB):
    free = shutil.disk_usage(root).free
    if free-peak_bytes < floor_bytes:
        raise RuntimeError('Insufficient headroom: free=%d peak=%d floor=%d' % (free,peak_bytes,floor_bytes))
    return free


def npy_header(stream):
    magic = stream.read(8)
    if magic[:6] != b'\x93NUMPY': raise ValueError('invalid NPY magic')
    width = 2 if magic[6] == 1 else 4
    length = struct.unpack('<H' if width == 2 else '<I', stream.read(width))[0]
    return ast.literal_eval(stream.read(length).decode('latin1'))


def local_job(job):
    job = dict(job)
    if job['source_id'] < 3:
        job['sequence_folder'] = str(ROOT/'outputs'/job['sequence_folder'].split('/outputs/',1)[1])
    else:
        job['payload'] = job['source_row']['source_path']
    return job


def inspect_job(job):
    job = local_job(job)
    if job['source_id'] < 3:
        folder = Path(job['sequence_folder'])
        with (folder/'hand.npy').open('rb') as stream: shape = npy_header(stream)['shape']
        with (folder/'poses.npy').open('rb') as stream: poses = npy_header(stream)['shape']
        n, objects = shape[0], poses[1]
        source = folder/'hand.npy'
    else:
        source = Path(job['payload'])
        with zipfile.ZipFile(source) as archive:
            with archive.open('hand.npy') as stream: shape = npy_header(stream)['shape']
            n = shape[0]
            if job['source_id'] == 4:
                objects = 1
                for name, expected in [('q.npy',(n,22)),('hand_root.npy',(n,13)),('object.npy',(n,13))]:
                    with archive.open(name) as stream: actual = npy_header(stream)['shape']
                    if actual != expected: raise ValueError('robot source shape changed: '+name)
            else:
                with archive.open('object_poses.npy') as stream: poses = npy_header(stream)['shape']
                if poses[0] != n: raise ValueError('hand/object frame mismatch')
                objects = poses[1]
    if objects != len(job['objects']) or n < 28: raise ValueError('source length/object identity mismatch')
    packet = None
    if job['source_id'] != 4:
        base = ROOT/'outputs/ptv3-shared-wam-v5/hand-packets-20261019-r1'
        record = json.loads((base/'records'/(job['key']+'.json')).read_text())
        if record['status'] != 'READY': raise ValueError('annotation not ready')
        packet = str(base/record['packet'])
        if not Path(packet).is_file(): raise FileNotFoundError(packet)
    # Includes headers/block padding and one atomic staging directory; rename
    # publishes it without making a second copy. Contact/event arrays follow.
    estimate = n*(2*778*3*2+2*21*3*4+2*21+5)+n*objects*(2048*3+65+24)+256*1024
    return dict(key=job['key'], source_id=job['source_id'], frames=n, objects=objects,
                source=str(source), packet=packet, estimated_bytes=estimate,
                split=sorted(job['windows']), source_sequence=job['source_sequence'])


def quota_snapshot():
    class D(ctypes.Structure):
        _fields_ = [(k,ctypes.c_uint64) for k in ('bhard','bsoft','space','ihard','isoft','inodes','btime','itime')]+[('valid',ctypes.c_uint32)]
    library=ctypes.CDLL(None,use_errno=True); results=[]
    for name,identifier,kind in [('user',os.getuid(),0),('group',os.getgid(),1),('project',0,2)]:
        value=D();ctypes.set_errno(0)
        result=library.quotactl(ctypes.c_int((0x800007<<8)|kind),ctypes.c_char_p(b'/dev/md0'),ctypes.c_int(identifier),ctypes.byref(value))
        error=ctypes.get_errno()
        results.append(dict(kind=name,id=identifier,result=result,errno=error,
            interpretation='no active quota' if result and error==3 else 'unverified: permission denied' if result and error==1 else os.strerror(error) if result else 'queried',
            values={k:getattr(value,k) for k,_ in value._fields_} if not result else None))
    return results


def inventory(args):
    out=args.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v6' not in out.parents:
        raise ValueError('fresh task-local output required')
    capacity(ROOT,256*1024**2);out.mkdir(parents=True)
    plan=json.loads(args.plan.read_text());start=time.monotonic();rows=[];failures=[]
    def checked(job):
        try:return inspect_job(job),None
        except Exception as exc:return None,dict(key=job['key'],source_id=job['source_id'],reason=repr(exc))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(row,failure) in enumerate(pool.map(checked,plan['jobs'])):
            if row:rows.append(row)
            else:failures.append(failure)
            if i%2000==0:
                save(out/'inventory-progress.json',dict(checked=i+1,total=len(plan['jobs']),failures=len(failures),elapsed_s=time.monotonic()-start))
                print(json.dumps(dict(checked=i+1,total=len(plan['jobs']),failures=len(failures))),flush=True)
            if time.monotonic()-start>900:raise TimeoutError('15-minute inventory bound')
    storage=shutil.disk_usage(ROOT);vfs=os.statvfs(ROOT)
    result=dict(status='COMPLETED' if not failures else 'INCOMPLETE',host=os.uname().nodename,
        target=str(out),filesystem=subprocess.check_output(['findmnt','-T',str(out),'-J'],text=True),
        uid=os.getuid(),free_bytes=storage.free,free_gib=storage.free/GIB,free_inodes=vfs.f_favail,
        quota=quota_snapshot(),checked_sequences=len(rows),failed_sequences=failures,
        estimated_full_cache_bytes=sum(r['estimated_bytes'] for r in rows),
        estimate='Fresh source NPY/ZIP headers for every planned sequence, plus writer dtypes and conservative per-sequence metadata padding; not yesterday census.',
        scope='Header/count/path inventory only; full CRC/content/geometry verification belongs to prepared sequences.',
        plan=str(args.plan.resolve()),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        elapsed_s=time.monotonic()-start,rows=rows)
    save(out/'inventory.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','filesystem')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('inventory','prepare'),required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'tmp/wam-v5/plan-compact.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='inventory':inventory(args)
    else:prepare(args)


if __name__=='__main__':main()
