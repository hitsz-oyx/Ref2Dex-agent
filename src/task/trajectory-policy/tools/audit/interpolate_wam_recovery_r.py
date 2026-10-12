"""One predeclared alpha=.5 R.net interpolation; no optimization or grid search."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2]; ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run')]
from fit_wam_recovery_retargeter import BASE_SHA, PANEL, sha, write, recovery_split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); parent=a.parent.resolve(); out=a.output.resolve()
    owner=ROOT/'outputs/trajectory-policy'
    if parent.parent!=owner or out.parent!=owner or out.exists() or not out.name.startswith('wam-ref2-'):
        raise ValueError('owned parent and fresh bounded output required')
    started=time.monotonic()
    if subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('GPU2 occupied')
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    import torch
    from trajectory_policy.pd_retargeter import load_model, windows, decode_target, command
    torch.set_num_threads(2)
    pm=json.loads((parent/'manifest.json').read_text()); pr=json.loads((parent/'result.json').read_text())
    if (pm['status']!='COMPLETED' or pr['status']!='COMPLETED' or pm['updates']!=1000
            or pm['base_R_sha256']!=BASE_SHA or sha(parent/'tau.pt')!=pm['checkpoint_sha256']
            or sha(parent/'tau.pt')!=pr['checkpoint_sha256']
            or sha(parent/'result.json')!=pm['result_sha256']
            or sha(parent/'predictions.npz')!=pm['predictions_sha256']
            or sha(parent/'scoped_admission.json')!=pm['admission_sha256']):
        raise ValueError('authenticated fixed-final1000 parent required')
    evidence=dict(pm['input_sha256'])
    for path in [Path(__file__),*[parent/k for k in ('manifest.json','result.json','predictions.npz','tau.pt','scoped_admission.json')]]:
        evidence[str(path)]=sha(path)
    def guard():
        if time.monotonic()-started>60:
            raise TimeoutError('60s bounded interpolation')
        if any(sha(path)!=digest for path,digest in evidence.items()):
            raise ValueError('consumed input drift')
        pids=set(map(int,subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()))
        if pids-{os.getpid()}:
            raise RuntimeError('foreign GPU2 process')
        paths=[v for v in owner.glob('wam-ref2-*') if v.is_dir()]
        paths += [v for v in (ROOT/'tmp').rglob('wam-*') if v.is_dir() and not any(q.name.startswith('wam-') for q in v.parents if q!=ROOT/'tmp')]
        size=sum(int(subprocess.check_output(['du','-sb',str(v)],text=True).split()[0]) for v in paths)
        if size>=100_000_000:
            raise RuntimeError('ref2 artifact bound100MB')
        return size
    guard()
    base_path=owner/'competent-pd-fit-20261011-r1/tau.pt'
    if sha(base_path)!=BASE_SHA:
        raise ValueError('base R identity drift')
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    packet=torch.load(parent/'tau.pt',map_location='cpu',weights_only=False)
    if (packet['base_R_sha256']!=BASE_SHA or packet['width']!=base['width']
            or packet['use_tau']!=base['use_tau'] or set(packet['model'])!=set(base['model'])):
        raise ValueError('base/parent architecture differs')
    for name,value in packet['model'].items():
        if name.startswith('net.'):
            packet['model'][name]=base['model'][name]+.5*(value-base['model'][name])
        elif not torch.equal(value,base['model'][name]):
            raise ValueError('normalization buffer differs')
    admission=json.loads((parent/'scoped_admission.json').read_text())
    nominal_path=Path(admission['nominal_trajectory_path'])
    native_path=Path(admission['native_source_path'])
    def read(path,limit=None):
        with np.load(path,allow_pickle=False) as f:
            return {k:f[k][:,:limit].copy() if limit else f[k].copy() for k in ('q','dq','hand','obj','velocity','pd_targets')}
    nominal=read(nominal_path,28); live=read(native_path)
    nm=json.loads(Path(admission['native_manifest_path']).read_text())
    offset,scale=[torch.as_tensor(nm['native_controller'][key],device='cuda') for key in ('offset','scale')]
    nr=windows(nominal,list(range(24,28))); lr=windows(live,[3]); train,held=recovery_split()
    panels=dict(nominal_report=(nr,np.isin(nr[3],PANEL)),nominal_startup=(nr,nr[3]==0),
        recovery_train=(lr,np.isin(lr[3],train)),recovery_held=(lr,np.isin(lr[3],held)),
        recovery_earlyheld=(lr,(lr[3]>=40)&(lr[3]<48)))
    raw={}; final={}; model=load_model(packet,'cuda')
    for name,(data,mask) in panels.items():
        state,tau,target=[torch.as_tensor(v[mask],device='cuda') for v in data[:3]]
        with torch.inference_mode():
            q=state[:,:18]; pd=decode_target(model(state,tau),q); truth=decode_target(target,q)
            _,action,clipped=command(pd,q,offset,scale)
        delta=(pd-truth).cpu().numpy(); xyz=np.linalg.norm(delta[:,:3],axis=-1)*1000
        final[name]=dict(count=len(xyz),xyz_p50_mm=float(np.percentile(xyz,50)),
            xyz_p95_mm=float(np.percentile(xyz,95)),rotation_p95_rad=float(np.percentile(np.abs(delta[:,3:6]).max(-1),95)),
            finger_p95_rad=float(np.percentile(np.abs(delta[:,[6,8,10,12,14,15]]).max(-1),95)),
            clipping_rate=float(clipped.float().mean()))
        with np.load(parent/'predictions.npz',allow_pickle=False) as f:
            row_ids=f['initial_'+name+'_row_ids'].copy()
        raw.update({f'final_{name}_pd':pd.cpu().numpy(),f'final_{name}_true_PD':truth.cpu().numpy(),
            f'final_{name}_action':action.cpu().numpy(),f'final_{name}_row_ids':row_ids})
    initial=pr['initial']
    gates=dict(recovery_held_XYZ20mm=final['recovery_held']['xyz_p95_mm']<=20,
        nominal_XYZ20mm=final['nominal_report']['xyz_p95_mm']<=20,
        startup_XYZ10mm=final['nominal_startup']['xyz_p95_mm']<=10,
        rotation_finger_retained=all(final[k][m]<=1.5*initial[k][m] for k in
            ('recovery_held','recovery_earlyheld','nominal_report','nominal_startup')
            for m in ('rotation_p95_rad','finger_p95_rad')))
    if any(not np.isfinite(v).all() for v in raw.values()):
        raise ValueError('nonfinite interpolation outputs')
    artifact_bytes=guard(); out.mkdir()
    (out/'scoped_admission.json').write_bytes((parent/'scoped_admission.json').read_bytes())
    manifest=dict(pm,status='RUNNING',run_id=out.name,input_sha256=evidence,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        admission_path=str(out/'scoped_admission.json'),admission_sha256=sha(out/'scoped_admission.json'),
        updates=0,parent_updates=1000,optimization_updates=0,net_interpolation_alpha=.5,
        parent_fit_path=str(parent),parent_checkpoint_sha256=pm['checkpoint_sha256'])
    packet.update(input_sha256=evidence,net_interpolation_alpha=.5,parent_fit_sha256=pm['checkpoint_sha256'],
        scoped_admission_sha256=manifest['admission_sha256'])
    torch.save(packet,out/'tau.pt')
    if (out/'tau.pt').stat().st_size>=2_000_000:
        raise ValueError('checkpoint bound2MB')
    result=dict(pr,initial=initial,final=final,gates=gates,curve=[],offline_screen_pass=all(gates.values()),
        local_GT_physical_calibration_allowed=all(gates.values()),net_interpolation_alpha=.5,
        parent_fit_path=str(parent),parent_checkpoint_sha256=pm['checkpoint_sha256'],optimization_updates=0,
        initial_net_identical_base=False,parent_initial_net_identical_base=pr['initial_net_identical_base'],net_gradient_sum=None,
        conclusion='PROMISING' if all(gates.values()) else 'UNCLEAR',
        elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        admission_path=manifest['admission_path'],admission_sha256=manifest['admission_sha256'],
        checkpoint_sha256=sha(out/'tau.pt'),artifact_bytes=artifact_bytes)
    np.savez_compressed(out/'predictions.npz',**raw)
    write(out/'result.json',result)
    manifest.update(status='COMPLETED',checkpoint_sha256=result['checkpoint_sha256'],
        result_sha256=sha(out/'result.json'),predictions_sha256=sha(out/'predictions.npz'))
    write(out/'manifest.json',manifest)
    guard()
    print(json.dumps(dict(final=final,gates=gates,offline_screen_pass=result['offline_screen_pass']),indent=2))


if __name__=='__main__': main()
