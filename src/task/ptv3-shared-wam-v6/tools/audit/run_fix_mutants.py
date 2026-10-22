"""Inject production-code regressions into isolated copies; require pytest to reject.

Never edits live sources or data. Each mutant is a separate CPU process, with
the mutated package imported before pytest/conftest can alter search paths.
"""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
PACKAGE=ROOT/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3'
TESTS=ROOT/'src/task/wam-rl-policy-v3/tests'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    mutants=[
        ('prefix-terminal','task_annotations.py','return declared_task,availability',
         "return __import__('dataclasses').replace(declared_task,terminal_mode=TerminalMode.UNKNOWN,events=()),availability",
         TESTS/'test_prefix_task_semantics.py','test_safe_prefix_keeps_full_terminal_endpoint_and_original_event_frames'),
        ('unknown-precondition','task_scheduler.py','spec.require_success_path()','pass',
         TESTS/'test_executable_task_admission.py','test_scheduler_refuses_unknown_before_any_transition'),
        ('stale-prefix-admission','task_admission.py','if spec.packet()!=declared.packet():','if False:',
         TESTS/'test_executable_task_admission.py','test_prefix_created_unknown_is_not_silently_dropped_before_migration'),
        ('online-voxel-terminal','measured_official_env.py','done[i]=decision.terminated','done[i]=decision.terminated or self.workspace[i]',
         TESTS/'test_model_domain_admission.py','test_later_voxel_violation_masks_tau_without_terminating_task'),
        ('voxel-admission-bypass','measured_official_env.py',"if not self.runtime.workspace_valid([admission])[0]:",'if False:',
         TESTS/'test_model_domain_admission.py','test_initial_model_domain_failure_is_explicit_admission_error'),
        ('release-after-endpoint','task_scheduler.py','def _terminal_release_ready(self,actual):',
         'def _terminal_release_ready(self,actual):\n        return False',
         TESTS/'test_terminal_release_timing.py','test_release_at_declared_frame_does_not_require_extra_held_dwell'),
        ('reward-component-omitted','measured_official_env.py',
         'self.reward_components[i]+=component;self.reward_totals+=component',
         'component[0]=0.;self.reward_components[i]+=component;self.reward_totals+=component',
         TESTS/'test_measured_episode_logging.py','test_actual_step_logs_reward_phase_and_interaction_conservation'),
        ('continuous-wrist-unwrapped','continuous_wrist.py',
         'result[:,3:6]=torch.remainder(result[:,3:6]+math.pi,2*math.pi)-math.pi',
         'result[:,3:6]=result[:,3:6]',
         TESTS/'test_continuous_wrist.py','test_only_continuous_rotation_coordinates_change_and_fk_is_equivalent'),
    ]
    reports=[]
    for name,filename,old,new,test,case in mutants:
        run=out/name;overlay=run/'src';shutil.copytree(PACKAGE,overlay/'wam_rl_policy_v3',ignore=shutil.ignore_patterns('__pycache__'))
        target=overlay/'wam_rl_policy_v3'/filename;source=target.read_text()
        if source.count(old)!=1:raise ValueError(f'{name}: expected one mutation site, got {source.count(old)}')
        target.write_text(source.replace(old,new,1))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',REF2DEX_RUN_ROOT=str(ROOT),
            PYTHONPATH=':'.join(map(str,[overlay,ROOT/'src/task/trajectory-policy/src',ROOT/'src/task/ptv3-shared-wam-v6/src'])),
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(ROOT/'tmp'))
        code="import sys;sys.path.insert(0,sys.argv[1]);import wam_rl_policy_v3."+filename[:-3]+";import pytest;raise SystemExit(pytest.main(sys.argv[2:]))"
        command=[sys.executable,'-c',code,str(overlay),'-q','-p','no:cacheprovider',str(test)+'::'+case,'--basetemp='+str(run/'pytest-tmp')]
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
        (run/'pytest.log').write_text(result.stdout)
        # Collection/import errors are not a killed mutant: require test failures.
        killed=result.returncode==1 and 'failed' in result.stdout and 'ERROR collecting' not in result.stdout
        reports.append(dict(mutant=name,file=filename,old=old,new=new,test=str(test.relative_to(ROOT))+'::'+case,
            returncode=result.returncode,killed=killed,log=str(run/'pytest.log')))
    report=dict(status='PASS' if all(r['killed'] for r in reports) else 'FAIL',mutants=reports,
        scope='Isolated source mutation tests; acceptance tests are also run separately against unmodified sources.')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],killed=sum(r['killed'] for r in reports),total=len(reports),output=str(out/'summary.json'))))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
