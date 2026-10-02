"""Serialize the frozen failed analysis without repeating native or learning phases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,PYTHON
from scripts.resume_continuous_critic_policy import bytes_in


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    begin=time.monotonic();old=json.loads((source/'run_manifest.json').read_text())
    assert old['experiment_id']=='P-20261002-option-model-policy' and old['run_status']=='FAILED'
    assert old['phases'][-1]['name']=='policy_result' and old['phases'][-1]['run_status']=='FAILED'
    assert all(phase['run_status']=='COMPLETED' for phase in old['phases'][:-1])
    assert 'Object of type bool_ is not JSON serializable' in (source/'policy_result.log').read_text()
    hashes=dict(old['input_sha256'])
    for path in source.rglob('*'):
        if path.is_file():hashes[str(path.resolve())]=sha(path)
    hashes[str(Path(__file__).resolve())]=sha(Path(__file__))
    def verify():
        for path,digest in hashes.items():assert sha(Path(path))==digest,path
    verify();out.mkdir()
    manifest=dict(experiment_id=old['experiment_id'],run_id=out.name,run_status='RUNNING',source_run=str(source),source_status_preserved='FAILED',input_sha256=hashes,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),pid=os.getpid(),new_native_trajectories=0,new_optimizer_steps=0)
    def save():
        manifest['wall_seconds']=time.monotonic()-begin
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        original=ROOT/'scripts/analyze_option_model_policy.py'
        text=original.read_text()
        marker='a=p.parse_args();root=a.directory.resolve()'
        assert text.count(marker)==1
        text=text.replace(marker,"p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();destination=a.output.resolve()")
        assert text.count("root/'results.json'")==2
        text=text.replace("root/'results.json'","destination/'results.json'")
        # Only the JSON representation changes; every count and gate executes
        # the exact original frozen source pinned by the failed parent.
        assert text.count('json.dumps(result')==2
        text=text.replace('json.dumps(result,indent=2)','json.dumps(result,indent=2,default=lambda value:value.item())')
        text=text.replace('json.dumps(result)','json.dumps(result,default=lambda value:value.item())')
        generated=out/'serialization_only_analysis.py';generated.write_text(text)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (out/'analysis.log').open('w') as log:
            subprocess.run([PYTHON,'-u',str(generated),'--directory',str(source),'--output',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
        result=json.loads((out/'results.json').read_text());assert result['run_status']=='COMPLETED'
        # A separate Python integer reconstruction verifies the saved outcome
        # counts and all three original decisions without NumPy scalar types.
        panels=[]
        for seed in (611,612):
            rows=json.loads((source/f's{seed}/rows.json').read_text())
            counts=[[sum(int(row['physical105']) for row in rows if row['motion']==motion and row['arm']==arm) for arm in range(4)] for motion in range(3)]
            assert all(sum(row['motion']==motion and row['arm']==arm for row in rows)==64 for motion in range(3) for arm in range(4))
            panels.append(counts)
        total=[sum(panel[motion][arm] for panel in panels for motion in range(3)) for arm in range(4)]
        gates=dict(gain5pp_over_all_three=all(total[1]/384>=total[k]/384+.05 for k in (0,2,3)),each_seed_noninferiority_all=all(sum(panel[m][1] for m in range(3))>=sum(panel[m][k] for m in range(3)) for panel in panels for k in (0,2,3)),motion1_loss_at_most5pp_vs_p0=sum(panel[1][1] for panel in panels)/128>=sum(panel[1][0] for panel in panels)/128-.05)
        assert gates==result['gates'] and dict(zip(('p0','cm','dynamics_off','direct_q'),total))==result['physical105_counts_per384']
        assert result['label']==('PROMISING' if all(gates.values()) else 'UNPROMISING')
        verify()
        combined_time=old['wall_seconds']+time.monotonic()-begin;combined_bytes=bytes_in(source)+bytes_in(out)
        assert combined_time<=1200 and combined_bytes<=2<<30
        manifest.update(run_status='COMPLETED',label=result['label'],inputs_unchanged=True,all_frozen_counts_and_gates_independently_rebuilt=True,result_sha256=sha(out/'results.json'),combined_wall_seconds=combined_time,combined_bytes=combined_bytes,retained_model_optimizer_steps=6000,retained_actor_optimizer_steps=3000,retained_pretraining_trajectories=1536,retained_evaluation_trajectories=1536)
        print(json.dumps(result),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
