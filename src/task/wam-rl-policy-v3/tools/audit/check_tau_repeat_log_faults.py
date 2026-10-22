"""Inject four log faults into private copies; never alter real traces or physics."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    original=a.run.resolve();manifest=json.loads((original/'manifest.json').read_text());fake=out/'fault-injected-run';fake.mkdir();manifest['audit_only_fault_injection']=True
    (fake/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(fake/'runtime-code').symlink_to(original/'runtime-code',target_is_directory=True)
    mutations={'block0-baseline-gt':'pd','block0-calibrated-gt':'episode','block0-baseline-wam':'packed-tau','block1-baseline-gt':'gt-value'}
    for row in manifest['runs']:
        name=row['label'];destination=fake/name
        if name not in mutations:destination.symlink_to(original/name,target_is_directory=True);continue
        destination.mkdir()
        with np.load(original/name/'trajectory.npz') as z:data={k:z[k].copy() for k in z.files}
        kind=mutations[name]
        if kind=='pd':data['pd_targets_actual'][0,0]+=.02
        elif kind=='episode':data['episodes_input'][0]+=1000
        elif kind=='packed-tau':data['observation'][3,421]+=.02
        elif kind=='gt-value':
            data['tau'][3,0,0]+=.02;data['observation'][3,421]+=.02
        np.savez_compressed(destination/'trajectory.npz',**data)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(ROOT/'tmp'))
    result=subprocess.run([sys.executable,str(TASK/'tools/audit/summarize_frozen_tau_repeats.py'),'--run',str(fake),'--output',str(out/'scoring')],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    (out/'scoring.log').write_text(result.stdout)
    if not (out/'scoring/summary.json').is_file():raise RuntimeError('Scoring crashed; not a caught log fault')
    summary=json.loads((out/'scoring/summary.json').read_text());rows={r['label']:r for r in summary['records']}
    caught=dict(pd=rows['block0-baseline-gt']['actual_pd_reconstruction_max_error']>2e-5,
        episode=not rows['block0-calibrated-gt']['episode_alignment_ok'],
        packed_tau=rows['block0-baseline-wam']['packed_tau_max_error']>0,
        gt_value=bool(rows['block1-baseline-gt']['gt_contract_error_ticks']))
    passed=result.returncode==1 and summary['status']=='FAIL' and all(caught.values())
    report=dict(status='PASS' if passed else 'FAIL',caught=caught,scorer_exit_code=result.returncode,real_physics_controls=0)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return int(not passed)


if __name__=='__main__':raise SystemExit(main())
