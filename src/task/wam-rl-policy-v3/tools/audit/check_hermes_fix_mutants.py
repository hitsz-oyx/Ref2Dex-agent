"""Isolated regressions for Hermes recheck, with actual pytest assertion failures."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cases=[
        ('legacy-raw-equality','tools/audit/repair_prefix_task_semantics.py',"if task.packet()!=raw.packet():raise ValueError('prefix changed complete declared task')","if task.packet()!=raw_packet:raise ValueError('wrong legacy equality')",'test_prefix_repair_cli.py'),
        ('contact-gap-undoes-observation','src/wam_rl_policy_v3/task_scheduler.py','self.path.settle_since=None;self.path.last_pose=None','self.path.pointer=self.confirmed_pointer;self.path.settle_since=None;self.path.last_pose=None','test_task_scheduler.py::test_one_frame_contact_gap_keeps_observed_geometry_without_certifying_grasp'),
        ('gap-supplies-crossing','src/wam_rl_policy_v3/task_scheduler.py','if not path_ready:self.path.last_pose=None','if False:self.path.last_pose=None','test_task_scheduler.py::test_contact_gap_does_not_create_an_unobserved_milestone_crossing'),
        ('state-occupancy-is-not-duration','tools/audit/score_source_pd_positive_control.py','longest=max(longest,float(now-start))','longest=max(longest,float(now-start)+1/30)','test_source_pd.py::test_positive_control_duration_uses_seconds_not_thirty_state_occupancy'),
        ('continuous-target-not-unwrapped','src/wam_rl_policy_v3/source_pd.py','target[...,3:6]+=2*np.pi*np.round((q[...,3:6]-target[...,3:6])/(2*np.pi))','target[...,3:6]=target[...,3:6]','test_source_pd.py::test_source_pd_inverse_uses_nearest_equivalent_continuous_target'),
        ('tau-includes-current','src/wam_rl_policy_v3/tau_replay_contract.py','np.arange(1,horizon+1)','np.arange(horizon)','test_tau_replay_contract.py::test_reference_delta_is_first_future_and_does_not_reanchor_on_live_hand'),
        ('shifted-clock-accepted','src/wam_rl_policy_v3/tau_replay_contract.py','not np.array_equal(tick,np.arange(controls+1))','False','test_tau_replay_contract.py::test_saved_clock_rejects_shifted_post_action_state'),
        ('pd-coupling-ratio-lost','src/wam_rl_policy_v3/tau_replay_contract.py','target[...,dst]=target[...,src]*ratio','target[...,dst]=target[...,src]','test_tau_replay_contract.py::test_pd_reconstruction_matches_independent_torch_native_contract'),
    ]
    results=[]
    for name,relative,old,new,test in cases:
        folder=out/name;copy=folder/'task';shutil.copytree(TASK,copy,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        target=copy/relative;text=target.read_text()
        if text.count(old)!=1:raise ValueError('mutation site changed: '+name)
        target.write_text(text.replace(old,new,1))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(copy/'src')+':'+str(ROOT/'src/task/trajectory-policy/src'),TMPDIR=str(ROOT/'tmp'))
        command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(copy/'tests'/test),'--basetemp='+str(folder/'pytest')]
        run=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        (folder/'pytest.log').write_text(run.stdout)
        results.append(dict(name=name,returncode=run.returncode,killed=run.returncode==1 and 'failed' in run.stdout and 'ERROR collecting' not in run.stdout))
    report=dict(status='PASS' if all(r['killed'] for r in results) else 'FAIL',mutants=results)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
