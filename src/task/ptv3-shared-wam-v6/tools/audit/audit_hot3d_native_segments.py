#!/usr/bin/env python3
"""Check every admitted HOT3D window against independently read NAS segments.

Read-only on both machines. NPZ byte identities can differ because the local
pack omitted segment boundaries; measure whether an admitted H4/K24 actually
crosses one before calling it a supervision failure. Never edit either input.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def audit(manifest, output):
    rows = {}
    with (manifest.parent/'sequences.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['source_id'] == 6:
                rows[row['key']] = row
    assert len(rows) == 287
    local = []
    for key, row in rows.items():
        with np.load(row['native_evidence']['uri'],allow_pickle=False) as z:
            local.append(dict(key=key,frames=len(z['time']),
                segment_edges=np.flatnonzero(np.diff(z['segment'])).tolist(),
                time_bad_edges=np.flatnonzero(~np.isclose(np.diff(z['time']),1/30,rtol=0,atol=1e-5)).tolist()))
    # No data path is interpolated into remote source code. Inputs cross stdin
    # as JSON; the remote job manifest resolves its own immutable NAS paths.
    code = '''import sys,json,numpy as np,pathlib
local={r['key']:r for r in json.load(sys.stdin)}
root=pathlib.Path('/mnt/ugreen_nas/storage/oyx_storage/remote-rigid-throughput-20261023-r1')
result=[]
for line in (root/'input-rows.jsonl').open():
 r=json.loads(line)
 if r['key'] not in local: continue
 path=r['job']['payload']
 with np.load(path,allow_pickle=False) as z:
  time=z['time']; segment=z['segment']
  assert len(time)==local[r['key']]['frames'] and np.isfinite(time).all() and (np.diff(time)>0).all()
  result.append(dict(key=r['key'],path=path,frames=len(time),
   segment_edges=np.flatnonzero(np.diff(segment)).tolist(),
   time_bad_edges=np.flatnonzero(~np.isclose(np.diff(time),1/30,rtol=0,atol=1e-5)).tolist()))
print(json.dumps(result))'''
    command = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
        '-o','UserKnownHostsFile='+str(ROOT/'tmp/ssh-4-4090/known_hosts'),
        '-i','/home2/wyy/.ssh/id_ed25519','-p','22023','wbcd@127.0.0.1',
        '/home/wbcd/miniconda3/envs/graspenv/bin/python -c '+shlex.quote(code)]
    result = subprocess.run(command,input=json.dumps(local).encode(),stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,timeout=180,cwd=ROOT)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    remote = json.loads(result.stdout)
    assert len(remote) == len(rows) and {r['key'] for r in remote} == set(rows)
    native = {r['key']:r for r in local}
    details = []
    for measured in remote:
        key = measured['key']; row = rows[key]
        with np.load(row['window_shard'],allow_pickle=False) as z:
            tuples = z['tuples']
        bad = np.zeros(measured['frames']-1,bool)
        bad[measured['segment_edges']] = True
        bad[measured['time_bad_edges']] = True
        prefix = np.r_[0,np.cumsum(bad)]
        crossing = prefix[tuples[:,2]+24] != prefix[tuples[:,2]-3]
        details.append(dict(key=key,split=row['split'],frames=measured['frames'],
            admitted_windows=len(tuples),crossing_windows=int(crossing.sum()),
            local_segment_edges=native[key]['segment_edges'],remote_segment_edges=measured['segment_edges'],
            segment_boundaries_equal=measured['segment_edges']==native[key]['segment_edges'],
            time_bad_edges_equal=measured['time_bad_edges']==native[key]['time_bad_edges'],
            crossing_examples=tuples[crossing][:20].tolist(),remote_path=measured['path']))
    crossings = sum(r['crossing_windows'] for r in details)
    record = dict(status='PASS' if crossings==0 else 'FAIL_CONTENT',sequences=len(rows),
        windows=sum(r['admitted_windows'] for r in details),crossing_windows=crossings,
        differing_segment_boundaries=sum(not r['segment_boundaries_equal'] for r in details),
        time_bad_edge_mismatches=sum(not r['time_bad_edges_equal'] for r in details),rows=details,
        scope='every local admitted HOT3D H4/K24 tested against actual remote native segment/time arrays',
        native_payload_identity_equal=False,training_allowed=False,python=sys.version)
    with output.open('x') as stream:
        json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in record.items() if k!='rows'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.manifest.resolve(),args.output.resolve())
