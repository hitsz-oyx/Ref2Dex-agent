"""Restore exact original input files from a read-only NAS backup; no new data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_full_rigid_data as prep


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    audit=prep.read(run/'source0-remote-inventory.json');budget=prep.read(run/'storage-budget-v4.json')
    if audit['status']!='COMPLETED' or not budget['fits']:raise ValueError('fresh exact remote inventory and fitting full budget required')
    prep.guard(run,audit['total_copy_bytes']+prep.GIB)
    destination=run/'restored-source0-inputs';destination.mkdir(exist_ok=True)
    filelist=run/'source0-copy-files.txt'
    filelist.write_text(''.join(r['source_sequence']+'/'+f['name']+'\n' for r in audit['rows'] for f in r['files']))
    ssh='ssh -p 22023 -i /home2/wyy/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=tmp/ssh-4-4090/known_hosts'
    origin='wbcd@127.0.0.1:/mnt/ugreen_nas/storage/oyx_storage/goal-point-wam-remote-20261012-r1/repo/outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006/processed/sequences/'
    command=['rsync','-rtz','--compress-level=1','--no-owner','--no-group','--partial-dir=.rsync-partial','--stats','--files-from='+str(filelist),'-e',ssh,origin,str(destination)+'/']
    start_receipt=run/('source0-copy-receipt-start.json' if not (run/'source0-copy-receipt-start.json').exists() else 'source0-copy-receipt-start-%d.json'%time.time_ns())
    prep.write(start_receipt,dict(source_script_sha256=prep.digest(__file__),status='STARTED',command=command,source_read_only=True,exact_input_bytes=audit['total_copy_bytes'],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.version))
    subprocess.run(command,check=True,cwd=prep.ROOT)
    plan={r['key']:r for r in prep.read(prep.PLAN)['jobs']};inventory=prep.read(run/'inventory.json');inv={r['key']:r for r in inventory['rows']};mappings=[]
    for r in audit['rows']:
        folder=destination/r['source_sequence'];job=plan[r['key']]
        for f in r['files']:
            if (folder/f['name']).stat().st_size!=f['bytes']:raise ValueError('restored file size mismatch')
        meta=prep.read(folder/'meta.json')
        if meta['objects']!=job['objects'] or meta['frames']!=r['frames']:raise ValueError('restored original native metadata mismatch')
        newjob=dict(job,sequence_folder=str(folder));inspected=prep.inspect_job(newjob)
        if inspected['frames']!=r['frames'] or inspected['objects']!=r['objects']:raise ValueError('fresh source shapes changed')
        mappings.append(dict(key=r['key'],job=newjob,inventory_updates=dict(inspected,status='CONVERT',conversion_bytes=prep.fixed_conversion_sizes(r['frames'],r['objects']),sequence_id=inv[r['key']]['sequence_id'])))
    old=prep.read(run/'recovered-inputs.json');archive=run/'recovered-inputs-source2.json'
    if not archive.exists():prep.shutil.copyfile(run/'recovered-inputs.json',archive)
    previous=[r for r in old['rows'] if r['key'] not in {x['key'] for x in mappings}]
    prep.write(run/'recovered-inputs.json',dict(status='COMPLETED',parity_checks=old['parity_checks'],plan_sha256=inventory['plan_sha256'],rows=previous+mappings,
        source0_backup=origin,source0_original_native_metadata_verified=True,restored_source0=len(mappings),reconstructed_source2=len(previous)))
    prep.write(run/'source0-copy-receipt-completed.json',dict(status='COMPLETED',restored_sequences=len(mappings),input_bytes=audit['total_copy_bytes'],
        exact_new_conversion_bytes=sum(r['inventory_updates']['conversion_bytes'] for r in mappings),command=command,source_read_only=True,files_from=str(filelist),recovered_inputs_sha256=prep.digest(run/'recovered-inputs.json')))
    print(json.dumps(dict(status='COMPLETED',restored_sequences=len(mappings))),flush=True)


if __name__=='__main__':main()
