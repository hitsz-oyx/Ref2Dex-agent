"""Inherit immutable completed fits from failed audit; no repeat model work."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();out=a.output.resolve();assert ROOT in source.parents and ROOT in out.parents and not out.exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    old=json.loads((source/'run_manifest.json').read_text());assert old['run_status']=='FAILED'
    assert [phase['run_status'] for phase in old['phases']]==['COMPLETED','COMPLETED','COMPLETED','FAILED']
    assert 'could not be broadcast together with shapes (5,16,3) (16,5,3)' in (source/'audit.log').read_text()
    assert json.loads((source/'fit/results.json').read_text())['actual_optimizer_updates']==4800
    for pid in [old['pid']]+[phase['pid'] for phase in old['phases']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise RuntimeError('ancestor PID still present')
    changed=ROOT/'scripts/audit_surface_calibration.py';assert str(changed) in old['input_sha256']
    committed=subprocess.check_output(['git','show',old['git_commit']+':scripts/audit_surface_calibration.py'],cwd=ROOT)
    import hashlib
    assert hashlib.sha256(committed).hexdigest()==old['input_sha256'][str(changed)]
    old_line="        sdk_error=float(np.abs(measured[:,[NAMES.index(n) for n in contact],:3,3]-train['sdk_next_positions'][select]).max());sdk_max=max(sdk_max,sdk_error);assert sdk_error<2e-4\n"
    new_lines="        measured_contact=measured[:,[NAMES.index(n) for n in contact]][:,:,:3,3]\n        assert measured_contact.shape==train['sdk_next_positions'][select].shape\n        sdk_error=float(np.abs(measured_contact-train['sdk_next_positions'][select]).max());sdk_max=max(sdk_max,sdk_error);assert sdk_error<2e-4\n"
    assert committed.decode().replace(old_line,new_lines)==changed.read_text(), 'only declared audit indexing fix allowed'
    hashes={}
    for filename,digest in old['input_sha256'].items():
        if filename==str(changed):continue
        assert sha(Path(filename))==digest,'ancestor protected drift '+filename
        hashes[filename]=digest
    for path in source.rglob('*'):
        if path.is_file():hashes[str(path.resolve())]=sha(path)
    for rel in subprocess.check_output(['git','ls-files','*.py'],cwd=ROOT,text=True).splitlines():hashes[str(ROOT/rel)]=sha(ROOT/rel)
    card=ROOT/'docs/experiments/probes/P-20261003-surface-calibration.md';hashes[str(card)]=sha(card)
    recovery=ROOT/'docs/activities/20261003-surface-calibration-audit-recovery.md';hashes[str(recovery)]=sha(recovery)
    def verify():
        for filename,digest in hashes.items():assert sha(Path(filename))==digest,'protected drift '+filename
    verify();out.mkdir();os.symlink(os.path.relpath(source/'data',out),out/'data');os.symlink(os.path.relpath(source/'fit',out),out/'fit')
    begin=time.monotonic();m=dict(experiment_id=old['experiment_id'],run_id=out.name,run_status='RUNNING',pid=os.getpid(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_run=str(source),
        source_run_commit=old['git_commit'],input_sha256=hashes,phases=[],new_optimizer_updates=0,
        inherited_optimizer_updates=4800,new_native_ticks=0,wall_limit_seconds=900,storage_limit_bytes=1<<30)
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    command=[PYTHON,'-u',str(changed),'--root',str(out),'--execution-source',old['source_execution'],'--native-source',old['source_native'],'--prior-source',old['source_prior']]
    phase=dict(name='audit',command=command,run_status='RUNNING',execution_device='cpu',reason='independent file/geometry/statistical audit only');m['phases'].append(phase);save()
    def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
    def guard():
        if time.monotonic()-begin>900 or bytes_in(out)>1<<30:raise RuntimeError('audit recovery budget')
    try:
        print(json.dumps(dict(phase='audit',status='STARTED',new_optimizer_updates=0)),flush=True)
        run_owned_child(command,ROOT,env,out/'audit.log',guard,900,spawned);verify()
        result=json.loads((source/'fit/results.json').read_text());audit=json.loads((out/'audit.json').read_text());assert audit['label']==result['label']
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');phase['run_status']='COMPLETED'
        m.update(run_status='COMPLETED',label=result['label'],actual_optimizer_updates=4800,inputs_unchanged=True,bytes=bytes_in(out))
        print(json.dumps(dict(label=result['label'],gates=result['gates'],new_optimizer_updates=0)),flush=True)
    except BaseException as error:phase.update(run_status='FAILED',error=repr(error));m.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
