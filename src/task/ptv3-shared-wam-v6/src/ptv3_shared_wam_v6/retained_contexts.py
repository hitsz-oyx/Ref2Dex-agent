"""Verify only this failed run's exiting GPU contexts before bounded sharing."""
import os
from pathlib import Path
import re
import subprocess
from ptv3_shared_wam_v5.storage import read_json


def validate_owner(pid, uid, started, cmdline, parent, pending, expected_start, current_uid):
    if (uid != current_uid or abs(started-expected_start)>120 or cmdline or parent!=1
            or not pending & ((1 << (15-1)) | (1 << (9-1)))):
        raise ValueError('retained process identity is not a confirmed exiting owned worker: %s'%pid)


def verify_retained_contexts(gpus, rows, failed_run):
    failed_run=Path(failed_run)
    failed=read_json(failed_run/'process.json')
    if (failed.get('status')!='FAILED' or failed.get('run_id')!=failed_run.name
            or not failed.get('work_version','').startswith('v6.')):
        raise ValueError('confirmed failed v6 supervisor record required')
    log=failed_run/'train.log'
    try:content=log.read_text()
    except PermissionError:log.chmod(0o664);content=log.read_text()
    owned=set(map(int,re.findall(r'Sending process (\d+) closing signal SIGTERM',content)))
    if not owned:raise ValueError('failed run worker identities unavailable')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total','--format=csv,noheader,nounits'],text=True,timeout=20)
    devices={int(f[0]):dict(uuid=f[1].strip(),total_mib=int(f[2]))
             for f in (line.split(',') for line in raw.splitlines()) if int(f[0]) in gpus}
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=20)
    selected={d['uuid'] for d in devices.values()}
    apps=[(int(f[0]),f[1].strip()) for f in (line.split(',') for line in raw.splitlines()) if f[1].strip() in selected]
    boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
    verified=[]
    for pid,uuid in apps:
        if pid not in owned:raise ValueError('unrelated GPU process present: %s'%pid)
        folder=Path('/proc')/str(pid);status=(folder/'status').read_text()
        fields={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in status.splitlines() if ':' in x}
        stat=(folder/'stat').read_text().split(') ',1)[1].split()
        started=boot+int(stat[19])/os.sysconf('SC_CLK_TCK')
        validate_owner(pid,int(fields['Uid'].split()[0]),started,(folder/'cmdline').read_bytes(),
                       int(fields['PPid']),int(fields['ShdPnd'],16),failed['started_at'],os.getuid())
        verified.append(dict(pid=pid,gpu_uuid=uuid,started_at=started))
    free={}
    for g in gpus:
        row=rows[g]
        if row['requires_reset'] or row['utilization']!=0 or row['memory_mib'] is None:
            raise ValueError('selected recovery GPU must be healthy and idle')
        free[g]=devices[g]['total_mib']-row['memory_mib']
        if free[g]<20*1024:raise ValueError('at least20GiB free capacity required per selected GPU')
    return dict(failed_run=str(failed_run),verified_exiting_workers=verified,free_mib=free,
                policy='User-authorized three-GPU recovery; batch<=64; no other active process allowed')
