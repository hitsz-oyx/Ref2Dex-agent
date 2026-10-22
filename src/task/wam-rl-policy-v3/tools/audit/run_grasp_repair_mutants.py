"""Require tests to reject reintroduced control/ablation defects in isolated copies."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    mutations=[
        ('zero-content-leak','if zero else','if False else','test_clean_ablation_changes_only_future_content_and_preserves_mask'),
        ('changed-valid-mask','return result,bool(valid)','return result,False','test_clean_ablation_changes_only_future_content_and_preserves_mask'),
        ('old-weak-servo','return command\n','return np.clip(command,-.0035,.0035)\n','test_goal_servo_overcomes_saved_v9_saturated_drift_and_never_uses_reported_velocity'),
        ('missing-integral','self.integral+np.where(free,.30*error*dt,0.)','self.integral','test_goal_servo_overcomes_saved_v9_saturated_drift_and_never_uses_reported_velocity'),
        ('always-filter-R','if not risk:\n','risk=True\n        if not risk:\n','test_healthy_R_PD_proposal_is_not_mistaken_for_instantaneous_kinematic_motion'),
        ('object-center-ignored','center=world.mean(0)','center=np.zeros(3)','test_geometric_candidate_follows_object_and_is_not_selected_by_object_identity')]
    reports=[]
    for name,old,new,case in mutations:
        folder=out/name;package=folder/'src/wam_rl_policy_v3';shutil.copytree(TASK/'src/wam_rl_policy_v3',package,ignore=shutil.ignore_patterns('__pycache__'))
        target=package/'grasp_repair.py';s=target.read_text()
        if s.count(old)!=1:raise ValueError('mutation site not unique: '+name)
        target.write_text(s.replace(old,new,1))
        code="import sys;sys.path.insert(0,sys.argv[1]);import wam_rl_policy_v3.grasp_repair;import pytest;raise SystemExit(pytest.main(sys.argv[2:]))"
        cmd=[sys.executable,'-c',code,str(folder/'src'),'-q','-p','no:cacheprovider',str(TASK/'tests/test_grasp_repair.py')+'::'+case,'--basetemp='+str(folder/'pytest-tmp')]
        result=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp')),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
        (folder/'test.log').write_text(result.stdout);killed=result.returncode==1 and 'failed' in result.stdout and 'ERROR collecting' not in result.stdout
        reports.append(dict(name=name,killed=killed,returncode=result.returncode,case=case,log=str(folder/'test.log')))
    report=dict(status='PASS' if all(r['killed'] for r in reports) else 'FAIL',killed=sum(r['killed'] for r in reports),total=len(reports),mutations=reports)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='mutations'}));return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
