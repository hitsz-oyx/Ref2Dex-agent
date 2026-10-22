"""Production-source negative controls for geometry, SE3, limits and residual PPO."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    mutations=[
        ('stale-grasp-offset','grasp_geometry.py','desired=effective@measured_relative','desired=effective@self.grasp_relative','test_relative_servo_still_corrects_measured_object_slip_after_wrist_arrived'),
        ('fixed15mm','grasp_geometry.py','tol=.10*width','tol=.015','test_geometry_thresholds_scale_and_are_rigid_transform_invariant'),
        ('all-five-tip-mean','grasp_geometry.py','float(np.partition(gaps,1)[1])','float(gaps.mean())','test_two_finger_measurement_ignores_three_distant_nonparticipating_fingers'),
        ('drop-grasp-lever-arm','grasp_geometry.py','desired=effective@measured_relative','desired=effective.copy()','test_relative_transform_includes_rotational_lever_arm_and_norm_rate_limits'),
        ('ignore-orientation','grasp_geometry.py','angular=.5*rotvec','angular=0.*rotvec','test_postgrasp_pose_command_contains_object_orientation_error'),
        ('unlimited-translation-norm','grasp_geometry.py','delta*=min(1.,self.linear_rate*dt/max(np.linalg.norm(delta),1e-12))','delta*=1.','test_relative_transform_includes_rotational_lever_arm_and_norm_rate_limits'),
        ('ignore-passive-limit','grasp_geometry.py',"self.offset=np.asarray(offset);self.scale=np.asarray(scale)","self.offset=np.asarray(offset);self.scale=np.asarray(scale);self.mask[7]=False",'test_passive_joint_limit_rejects_candidate_before_ranking'),
        ('oversized-residual','intent_residual.py','correction=np.tanh(np.asarray(latent))*RESIDUAL_SCALE','correction=np.tanh(np.asarray(latent))*RESIDUAL_SCALE*10','test_residual_is_bounded_in_intent_and_does_not_rotate_translation_about_object'),
        ('drop-time-limit-bootstrap','intent_residual.py','val=np.r_[values,bootstrap]','val=np.r_[values,0.]','test_actual_residual_ppo_uses_time_limit_bootstrap_and_has_bounded_epochs')]
    records=[]
    for name,file,old,new,case in mutations:
        folder=out/name;package=folder/'src/wam_rl_policy_v3';shutil.copytree(TASK/'src/wam_rl_policy_v3',package,ignore=shutil.ignore_patterns('__pycache__'))
        path=package/file;source=path.read_text()
        if source.count(old)!=1:raise ValueError('nonunique mutation site '+name)
        path.write_text(source.replace(old,new))
        code="import sys;sys.path.insert(0,sys.argv[1]);import wam_rl_policy_v3.grasp_geometry;import pytest;raise SystemExit(pytest.main(sys.argv[2:]))"
        command=[sys.executable,'-c',code,str(folder/'src'),'-q','-p','no:cacheprovider',str(TASK/'tests/test_grasp_geometry_se3.py')+'::'+case,'--basetemp='+str(folder/'pytest-tmp')]
        result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp')),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
        (folder/'test.log').write_text(result.stdout);killed=result.returncode==1 and 'failed' in result.stdout and 'ERROR collecting' not in result.stdout
        records.append(dict(name=name,killed=killed,returncode=result.returncode,case=case))
    report=dict(status='PASS' if all(r['killed'] for r in records) else 'FAIL',killed=sum(r['killed'] for r in records),total=len(records),records=records)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return int(report['status']!='PASS')

if __name__=='__main__':raise SystemExit(main())
