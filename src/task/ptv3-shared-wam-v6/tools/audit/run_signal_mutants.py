"""Isolated production-source regressions for the v2 audit; no live mutation."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
TASK=ROOT/'src/task/ptv3-shared-wam-v6'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    samples=TASK/'tools/run/samples_efficiency_env.py';data=TASK/'src/ptv3_shared_wam_v6/data.py'
    test=TASK/'tests/test_samples_execution_signals.py'
    mutants=[
        ('encoder-terminates',samples,'done[i]=decision.terminated or self.physical_abort[i]','done[i]=decision.terminated or self.physical_abort[i] or self.workspace[i]',test,'test_encoder_failure_masks_tau_and_preserves_physical_episode'),
        ('encoder-vetoes-success',samples,'success=bool(decision.task_complete and not self.physical_abort[i] and not forced)','success=bool(decision.task_complete and not self.physical_abort[i] and not forced and not self.workspace[i])',test,'test_success_survives_same_tick_encoder_failure'),
        ('physical-safety-disabled',samples,"self.physical_abort[i]=self.safety[i]['abort']",'self.physical_abort[i]=False',test,'test_real_table_penetration_aborts_even_with_encodable_input'),
        ('invalid-tau-not-zeroed',samples,'tau=np.zeros((len(ids),8,11,3),np.float32)','tau=np.full((len(ids),8,11,3),.01,np.float32)',test,'test_encoder_failure_masks_tau_and_preserves_physical_episode'),
        ('dropout-becomes-raw-invalid',data,'event_raw_invalid=np.bool_(not raw_valid)','event_raw_invalid=np.bool_(not valid)',TASK/'tests/test_v6_training.py','test_raw_event_invalidity_and_active_dropout_are_separate'),
        ('wrong-sparse21-map',TASK/'src/ptv3_shared_wam_v6/online_runtime.py','ids=np.asarray(SPARSE11_IN_21);samples=[];rotations=[]','ids=np.arange(11);samples=[];rotations=[]',TASK/'tests/test_ppo_model_identity.py','test_actual_v6_predict_maps_full21_output_and_consumes_event_goal'),
        ('event-not-injected',TASK/'src/ptv3_shared_wam_v6/online_runtime.py','if event:','if False:',TASK/'tests/test_ppo_model_identity.py','test_actual_v6_predict_maps_full21_output_and_consumes_event_goal'),
        ('recoverable-encoder-error-bubbles',samples,'except EncodingUnavailable as error:','except ZeroDivisionError as error:',test,'test_recoverable_encoder_error_continues_with_masked_state_only_mode'),
        ('thirty-states-become-one-second',TASK/'src/ptv3_shared_wam_v6/execution_contract.py','now-self.since>=1.-1e-10','now-self.since>=29/30',test,'test_narrow_witness_uses_elapsed_seconds_and_keeps_ever_after_loss'),
        ('ever-success-overwritten-on-loss',TASK/'src/ptv3_shared_wam_v6/execution_contract.py','self.ever_success|=self.final_success','self.ever_success=self.final_success',test,'test_narrow_witness_uses_elapsed_seconds_and_keeps_ever_after_loss'),
    ]
    reports=[]
    for name,source,old,new,target,case in mutants:
        run=out/name;run.mkdir();text=source.read_text()
        if text.count(old)!=1:raise ValueError('mutation site changed: '+name)
        modified=run/source.name;modified.write_text(text.replace(old,new,1))
        if source==samples:
            code="""import importlib.util,sys
original=importlib.util.spec_from_file_location
def redirect(name,path,*args,**kwargs):
    if str(path).endswith('/tools/run/samples_efficiency_env.py'):path=sys.argv[1]
    return original(name,path,*args,**kwargs)
importlib.util.spec_from_file_location=redirect
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
"""
        else:
            module_name='ptv3_shared_wam_v6.'+source.stem
            code="import sys;import "+module_name+" as module;exec(compile(open(sys.argv[1]).read(),module.__file__,'exec'),module.__dict__);import pytest;raise SystemExit(pytest.main(sys.argv[2:]))"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(ROOT/'tmp'),
            PYTHONPATH=':'.join(map(str,[TASK/'src',ROOT/'src/task/wam-rl-policy-v3/src',ROOT/'src/task/trajectory-policy/src'])))
        command=[sys.executable,'-c',code,str(modified),'-q','-p','no:cacheprovider',str(target)+'::'+case,'--basetemp='+str(run/'pytest')]
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
        (run/'pytest.log').write_text(result.stdout);killed=result.returncode==1 and 'failed' in result.stdout and 'ERROR collecting' not in result.stdout
        reports.append(dict(name=name,file=str(source.relative_to(ROOT)),old=old,new=new,test=str(target.relative_to(ROOT))+'::'+case,killed=killed,returncode=result.returncode))
    report=dict(status='PASS' if all(r['killed'] for r in reports) else 'FAIL',mutants=reports)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],killed=sum(r['killed'] for r in reports),total=len(reports))))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
