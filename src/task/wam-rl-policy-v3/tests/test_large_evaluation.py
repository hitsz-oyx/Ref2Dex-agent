from pathlib import Path
import sys
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.binary_statistics import wilson,compare
from wam_rl_policy_v3.stage_evaluation import PalmSurface


def test_known_small_n_statistics_and_frozen_joint_success_rule():
    np.testing.assert_allclose(wilson(5,5),[.5655175352,1],atol=1e-9)
    assert compare(2,5,5,5)['fisher_two_sided_p']==pytest.approx(1/6)
    assert compare(3,5,5,5)['fisher_two_sided_p']==pytest.approx(4/9)
    assert not compare(5,5,2,5)['registered_a_superior']
    assert compare(36,40,4,40)['registered_a_superior']
    assert wilson(0,40)[1]>.08


def test_base_origin_is_not_the_physical_palm_surface_and_rigid_motion_is_equivariant():
    # The bug is at the real field seam: the base origin is 160mm from the
    # object, although an actual collision-mesh point is touching it.
    palm=PalmSurface(np.array([[0,0,-.16],[.02,0,-.10]]));identity=np.eye(4)
    obj=identity.copy();obj[2,3]=-.16;points=np.zeros((1,3))
    assert np.linalg.norm(obj[:3,3])>.1 and palm.gap(identity,obj,points)==0
    global_pose=identity.copy();global_pose[:3,:3]=Rotation.from_euler('xyz',[.3,-.4,.2]).as_matrix();global_pose[:3,3]=[.2,.4,.7]
    assert palm.gap(global_pose,global_pose@obj,points)<1e-15


def test_actual_urdf_palm_mesh_has_a_nonzero_offset_from_its_wrist_origin():
    root=Path(__file__).resolve().parents[4];path=root/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    palm=PalmSurface.from_urdf(path)
    assert palm.vertices[:,2].min()<-.15 and abs(palm.vertices[:,2].max())<1e-4


def test_unchanged_s2c_can_reject_the_deliberately_wrong_wrist_gap():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ptv3-shared-wam-v6/src'))
    from ptv3_shared_wam_v6.staged_progress import StagedProgress
    from wam_rl_policy_v3.task_reward import RewardFeatures
    feedback=SimpleNamespace(workspace_ok=True,force_valid=True,contact_observed=True,grasp_established=True,grasp_lost=False,supported=False,released=False)
    forces=np.ones((5,3));features=RewardFeatures(.16,.01,0.,0.,0.,1.,.01)
    good=StagedProgress(0);bad=StagedProgress(0)
    palm=PalmSurface([[0,0,-.16]]);pose=np.eye(4);obj=np.eye(4);obj[2,3]=-.16
    corrected=palm.features(features,pose,obj,np.zeros((1,3)))
    for t in range(40):
        z=0 if t==0 else .06
        good.update(z,t/30,corrected,feedback,forces)
        bad.update(z,t/30,features,feedback,forces)
    assert good.snapshot()['current']['hold'] and not bad.snapshot()['achieved']['approach']


def test_early_hold_target_error_or_later_drop_is_not_terminal_success():
    from wam_rl_policy_v3.stage_evaluation import terminal_success
    held={'current':{'hold':True}};dropped={'current':{'hold':False}}
    assert not terminal_success(held,.01,100,600,False)
    assert not terminal_success(held,.016,600,600,False)
    assert not terminal_success(dropped,.01,600,600,False)
    assert not terminal_success(held,.01,600,600,True)
    assert terminal_success(held,.015,600,600,False)


def test_registration_has_40_per_arm_per_subset_and_no_validation_seeds(tmp_path):
    import json,subprocess
    root=Path(__file__).resolve().parents[4]
    panel=tmp_path/'panel.json';out=tmp_path/'registration.json'
    panel.write_text(json.dumps(dict(status='READY',unseen=[dict(sequence='new%d'%i,object='object%d'%i,reference='unused-path') for i in range(8)])))
    subprocess.run([sys.executable,str(root/'src/task/wam-rl-policy-v3/tools/run/register_large_grasp.py'),'--panel',str(panel),'--output',str(out)],check=True)
    reg=json.loads(out.read_text());jobs=reg['jobs']
    assert len(jobs)==80 and len({j['native_seed'] for j in jobs})==80
    assert all(100<=j['native_seed']<=179 and 180<=j['model_seed']<=259 for j in jobs)
    for subset in ('seen','unseen'):
        trials=[j for j in jobs if j['subset']==subset]
        assert len(trials)==40
        assert all(sorted(j['order'])==reg['arms'] for j in trials)
        assert all(sum(j['order'][position]==arm for j in trials)==10 for position in range(4) for arm in reg['arms'])


def test_incomplete_n5_never_publishes_even_a_significant_contribution(tmp_path):
    import importlib.util,json
    script=Path(__file__).resolve().parents[1]/'tools/run/summarize_large_grasp.py'
    spec=importlib.util.spec_from_file_location('large_summary_under_test',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reg=dict(protocol='test',subsets=['seen','unseen'],arms=['A1','A2','A3','A4'],horizon_controls=600,primary='test endpoint')
    rows=[]
    for subset in reg['subsets']:
        for arm in reg['arms']:
            for i in range(5):
                passed=arm=='A1';stage=dict.fromkeys(['approach','contact','lift','hold','release'],passed)
                rows.append(dict(id=subset+arm+str(i),trial=subset+str(i),subset=subset,arm=arm,sequence=subset,final_success=passed,ever_success=passed,final_hold=passed,goal_within15mm=passed,
                                 final_goal_error_m=0.01 if passed else .1,controls=600,physical_abort=None,stage_achieved=stage,stage_final=stage,longest_hold_s=1. if passed else 0.,adapter_clipped_controls=0,final_clipped_controls=0))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(protocol='test',runs=rows,status='RUNNING',environment_controls=24000)))
    result=module.summarize(reg,[manifest]);assert result['status']=='INCOMPLETE' and not result['robust_tau_contribution']
    for subset in reg['subsets']:
        comparison=result['comparisons'][subset]['A1_vs_A2']
        assert comparison['fisher_two_sided_p']<.05 and not comparison['claim_eligible'] and not comparison['registered_a_superior']
