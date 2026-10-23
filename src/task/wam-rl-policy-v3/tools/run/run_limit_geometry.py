"""Five frozen S2e geometry/R episodes with shared admission, no new baseline."""
from pathlib import Path
import difflib
ROOT=Path(__file__).resolve().parents[5]
LEGACY=ROOT/'tmp/s2e-geometry-20261022/outputs/wam-rl-policy-v3/s2e-geometry-20261022-r1'
source=(LEGACY/'runner.py').read_text()
changes=[
 ('import argparse,json,os,shutil,socket,subprocess,sys,time','import argparse,json,os,shutil,socket,subprocess,sys,time,types\nfrom dataclasses import replace'),
 ('out.mkdir(parents=True);frozen=freeze(out,oldrun)',"out.mkdir(parents=True);frozen=out/'runtime-code';shutil.copytree(Path("+repr(str(LEGACY/'runtime-code'))+"),frozen,symlinks=True)\n    shutil.copyfile(TASK/'src/wam_rl_policy_v3/reference_admission.py',frozen/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/reference_admission.py')\n    (out/'runner.patch').write_text(APPLIED_DIFF)"),
 ("    shutil.copyfile(TASK/'src/wam_rl_policy_v3/pose_consumer.py',frozen/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/pose_consumer.py')\n",''),
 ("    shutil.copyfile(ROOT/'src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/staged_progress.py',frozen/'src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/staged_progress.py')\n",''),
 ("    shutil.copyfile(TASK/'src/wam_rl_policy_v3/geometry_tau.py',frozen/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/geometry_tau.py')\n",''),
 ("    wam=OnlineV6Runtime(ASSETS/'outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt',ASSETS/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json',ASSETS/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry',device='cuda:0',forecast_only=True)","    wam=types.SimpleNamespace(step=11000,config={},seconds=0.) # unused in geometry arm; no WAM inference"),
 ("keys=['wam','geometry'];order=[keys[b%2:]+keys[:b%2] for b in range(5)]","keys=['geometry'];order=[keys[:] for b in range(5)]"),
 ("from wam_rl_policy_v3.tau_replay_contract import pd_targets","from wam_rl_policy_v3.tau_replay_contract import pd_targets\n    from wam_rl_policy_v3.reference_admission import JointEnvelope,ProgressWatchdog,UnreachableReference"),
 ("        for block,chains in enumerate(order):","        envelope=JointEnvelope(hello['audit']['physical_joint_limits'],offset,scale)\n        for block,chains in enumerate(order):"),
 ("for t in range(301):","watchdog=ProgressWatchdog();protection_stop=None\n                for t in range(601):"),
 ("if witness.final_success or safe['abort'] or t==300:break","if safe['abort'] or t==600:break"),
 ("                    evidence=context.observer.last_evidence","                    lower,upper=envelope.finger_intervals()\n                    original_fingers=np.r_[intent.preshape_q,intent.closure_q].copy()\n                    intent=replace(intent,preshape_q=np.clip(intent.preshape_q,lower,upper),closure_q=np.clip(intent.closure_q,lower,upper))\n                    try:envelope.pose_configuration(pose@intent.approach_pose_object,np.asarray(state['q'])[0],pose7(np.asarray(state['palm_pose'])[0]),intent.preshape_q)\n                    except UnreachableReference as error:protection_stop='REFERENCE_REJECTED: '+str(error);break\n                    evidence=context.observer.last_evidence"),
 ("                    raw_action=raw[None]","                    stalled=watchdog.update(t/30,pose7(np.asarray(state['palm_pose'])[0]),np.asarray(info['wrist_target']))\n                    if stalled:protection_stop=stalled;break\n                    raw,guard=envelope.protect(raw,np.asarray(state['q'])[0],np.asarray(state['dq'])[0])\n                    info['admission']=guard;info['reference_finger_alternative']=bool(np.max(np.abs(original_fingers-np.r_[intent.preshape_q,intent.closure_q]))>1e-6)\n                    raw_action=raw[None]"),
 ("record=dict(label=label,chain=chain,block=block,","record=dict(protective_stop=protection_stop,label=label,chain=chain,block=block,"),
]
original=source
for old,new in changes:
 if source.count(old)!=1:raise ValueError('frozen runner seam changed: '+old[:90])
 source=source.replace(old,new)
diff=''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='frozen-s2e/runner.py',tofile='admitted-geometry/runner.py'))
if __name__=='__main__':
 namespace=dict(__name__='limit_geometry_frozen',__file__=str(Path(__file__).resolve()),APPLIED_DIFF=diff)
 exec(compile(source,str(Path(__file__).resolve()),'exec'),namespace)
 namespace['main']()
