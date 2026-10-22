"""CPU boundary invariants with explicit mutant rejection, not physics certification."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from . import paths


def equal(actual, expected, atol=1e-6):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and set(actual)==set(expected) and all(equal(actual[k],v,atol) for k,v in expected.items())
    if isinstance(expected,(str,bool,type(None))):return actual==expected
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape:return False
    if a.dtype.kind in 'OUS' or b.dtype.kind in 'OUS':return bool(np.array_equal(a,b))
    return bool(np.allclose(a,b,atol=atol,rtol=1e-6,equal_nan=False))


def verified(name, actual, expected, mutant, mutation):
    positive=equal(actual,expected);caught=not equal(mutant,expected)
    return dict(name=name,status='PASS' if positive and caught else 'FAIL',positive_control=positive,
                mutation_detected=caught,mutation=mutation)


def independent_reward(previous,current,before,after,action,paid=()):
    def phi(d,f):
        if int(d.phase)==0:return -min(f.palm_gap_m,.5)-min(f.mean_tip_gap_m,.5)
        if int(d.phase)==1:return d.path_progress-min(f.tracking_error_m,.2)
        return -min(f.terminal_error_m,.2)-min(f.object_speed_mps,1.)-.01*min(f.contact_force_n,20.)+min(f.hand_clearance_m,.05)
    events=.2*sum(e not in set(paid) and e in ('grasp_acquired','path_arrived') for e in set(current.transitions))
    return dict(shaping=.96*(0. if current.terminated else phi(current,after))-phi(previous,before),event=events,
                sparse=1. if current.task_complete else -1. if current.failure_reason else 0.,
                time=-.01/30,effort=-.01*float(np.mean(np.square(action))))


def evaluation_identity_matches(left,right):
    def exact(a,b):
        if isinstance(b,dict):
            return isinstance(a,dict) and set(a)==set(b) and all(exact(a[k],v) for k,v in b.items())
        return bool(np.array_equal(np.asarray(a),np.asarray(b)))
    required=('seed','envs','hz','controls','state')
    return all(k in left and k in right and exact(left[k],right[k]) for k in required)


def legacy_qualification_timing():
    """Keep old control-count proxy, and separately report elapsed state timestamps."""
    source=paths.ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/summary.json'
    old=json.loads(source.read_text());arms={}
    for name,row in old['arms'].items():
        counts=[int(c['longest_hold_controls']) for c in row['cases']]
        elapsed=[max(0,n-1)/30 for n in counts]
        arms[name]=dict(samples=len(counts),legacy_30_sample_successes=sum(n>=30 for n in counts),
            elapsed_1s_successes=sum(s>=1 for s in elapsed),longest_hold_samples=counts,
            longest_hold_elapsed_s=elapsed,mean_longest_elapsed_s=float(np.mean(elapsed)))
    check=verified('held duration uses elapsed endpoints, not state occupancy',
        max(arms['flow']['longest_hold_elapsed_s']),29/30,1.,'report thirty held states as one elapsed second')
    return dict(source=str(source),scope='same existing P0 summary; 30 Hz contiguous post-control states; original proxy retained',
        rule='max(0, held_state_samples - 1)/30 seconds; >=31 states for elapsed>=1s',arms=arms),check


def voxel_first_failure(sample,config,world_from_anchor):
    """Exactly the V4Runtime gate's float32 grid, with the first invalid coordinate."""
    mask=sample['point_valid'] & (sample['features'][:,14]<.5)
    points=[sample['xyz'][mask]];identities=[('scene',int(i)) for i in np.flatnonzero(mask)]
    if config.get('hand_in_ptv3',True):
        valid=sample['start_hand_valid'];points.append(sample['start_hand'][valid])
        identities += [('hand',int(slot),int(joint)) for slot,joint in np.argwhere(valid)]
    values=torch.as_tensor(np.concatenate(points),dtype=torch.float32)
    origin=torch.tensor(config['voxel_origin_m'],dtype=torch.float32)
    grid=torch.floor((values-origin)/config['voxel_m'])
    bad=~torch.isfinite(values) | (grid<0) | (grid>65535)
    entries=bad.nonzero()
    if not len(entries):return None
    index,axis=map(int,entries[0]);local=values[index].numpy();pose=np.asarray(world_from_anchor)
    identity=identities[index];surface=np.asarray(sample['xyz'][mask],dtype=float)
    finite_surface=surface[np.isfinite(surface).all(-1)]
    nearest=None if not len(finite_surface) or not np.isfinite(local).all() else float(np.linalg.norm(finite_surface-local,axis=-1).min())
    geometry=dict(object_center_world_xyz=pose[:3,3].tolist(),
        object_anchor_distance_m=float(np.linalg.norm(local)) if np.isfinite(local).all() else None,
        nearest_included_scene_surface_m=nearest,
        surface_scope='included scene points; object surface only for target-only online samples')
    hand_identity=None
    if identity[0]=='hand':
        slot,joint=identity[1:];names=['wrist','thumb_base','index_base','middle_base','ring_base','little_base',
            'thumb_tip','index_tip','middle_tip','ring_tip','little_tip']
        wrist=np.asarray(sample['start_hand'][slot,0],dtype=float)
        wrist_valid=bool(sample['start_hand_valid'][slot,0])
        tip_ids=np.flatnonzero(sample['start_hand_valid'][slot,6:])+6
        tips=np.asarray(sample['start_hand'][slot,tip_ids],dtype=float)
        hand_identity=dict(slot=slot,slot_name=('right','left')[slot] if slot in (0,1) else 'UNKNOWN',
            canonical_joint_index=joint,canonical_joint_name=names[joint] if joint<len(names) else 'UNKNOWN',
            schema='WAM canonical11: wrist, five bases, five tips')
        geometry.update(same_slot_wrist_anchor_xyz=wrist.tolist() if wrist_valid else None,
            same_slot_wrist_world_xyz=(wrist@pose[:3,:3].T+pose[:3,3]).tolist() if wrist_valid else None,
            distance_to_same_slot_wrist_m=float(np.linalg.norm(local-wrist)) if wrist_valid else None,
            nearest_valid_same_slot_fingertip_m=float(np.linalg.norm(tips-local,axis=-1).min()) if len(tips) else None)
    return dict(point=identity,axis='xyz'[axis],anchor_xyz=local.tolist(),hand_identity=hand_identity,geometry_relation=geometry,
                world_xyz=(local@pose[:3,:3].T+pose[:3,3]).tolist(),
                voxel_origin=origin.tolist(),voxel_size_m=float(config['voxel_m']),voxel_index=grid[index].tolist(),
                violated_bound='nonfinite' if not np.isfinite(local[axis]) else 'lower' if grid[index,axis]<0 else 'upper',
                lower_boundary_clearance_m=float(local[axis]-origin[axis]),
                invalid_point_count=int(bad.any(-1).sum()),
                point_index_in_gate=index,scope='representation voxel domain, not a mechanical reachability certificate')


def basic_checks():
    from wam_rl_policy_v3.tau_contract import v4_trajectory_to_student
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM,WAM_TO_NATIVE,expand_action
    from wam_rl_policy_v3.inspire_shadow import tau_world
    from wam_rl_policy_v3.bc_mlp import NormalizedCommandMLP
    from wam_rl_policy_v3.measured_reward_bridge import MeasuredRewardTracker
    from wam_rl_policy_v3.task_reward import RewardFeatures
    from wam_rl_policy_v3.task_spec import Phase
    checks=[]
    trajectory=np.zeros((24,2,11,3));trajectory[:,0,:,0]=(np.arange(1,25)/30)[:,None];trajectory[:,1,:,1]=2*(np.arange(1,25)/30)[:,None]
    t30=v4_trajectory_to_student(trajectory,object_rotation=np.eye(3),native_fps=30)
    t60=v4_trajectory_to_student(trajectory,object_rotation=np.eye(3),native_fps=60)
    oracle30=np.zeros((8,11,3));oracle30[:,:,0]=(np.arange(1,9)/30)[:,None]
    oracle60=oracle30/2
    shifted=v4_trajectory_to_student(np.roll(trajectory,1,axis=0),object_rotation=np.eye(3),native_fps=30)
    checks.append(verified('tau[0] first future; tau(0)=0 only interpolation endpoint',t30,oracle30,shifted,'shift source trajectory one frame'))
    checks.append(verified('Inspire8/30s versus Shadow8/60s',t60,oracle60,t30,'use 30Hz rows without 60Hz interpolation'))
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    rotated=v4_trajectory_to_student(trajectory,object_rotation=rotation,native_fps=30)
    checks.append(verified('object-anchor displacement to world rotation',rotated,oracle30@rotation.T,oracle30,'omit world-from-anchor rotation'))
    selected=v4_trajectory_to_student(trajectory,object_rotation=np.eye(3),native_fps=30,hand_slot='left')
    checks.append(verified('right/left hand slot identity',t30,oracle30,selected,'swap right and left slot'))
    from ptv3_shared_wam_v5.contracts import JOINT_NAMES,SPARSE11_IN_21
    wanted=['wrist']+[finger+'_base' for finger in ['thumb','index','middle','ring','little']]+[finger+'_tip' for finger in ['thumb','index','middle','ring','little']]
    expected=np.array([JOINT_NAMES.index(name) for name in wanted]);actual=np.asarray(SPARSE11_IN_21)
    checks.append(verified('joint21 to canonical11 semantic mapping',actual,expected,np.roll(actual,1),'cyclic joint-index permutation'))
    native=np.arange(11);canonical=native[NATIVE_TO_WAM];back=canonical[WAM_TO_NATIVE]
    checks.append(verified('native11/canonical11 permutation inverse',back,native,canonical,'forget inverse point permutation'))
    from wam_rl_policy_v3.inspire_contract import ACTIVE
    raw_action=np.linspace(-2,2,12,dtype=np.float32)
    expanded=expand_action(raw_action)
    expected_expanded=np.zeros(18,np.float32);expected_expanded[list(ACTIVE)]=np.clip(raw_action,-1,1)
    unclipped=expected_expanded.copy();unclipped[list(ACTIVE)]=raw_action
    checks.append(verified('raw active12 clipped canonical to native18 exact coordinates',expanded,expected_expanded,unclipped,'bypass action clipping'))
    model=NormalizedCommandMLP(686,312,12)
    class Capture(torch.nn.Module):
        def forward(self,x):self.x=x.clone();return x[:,:12]
    capture=Capture();model.net=capture
    model.x_mean.copy_(torch.linspace(-.1,.1,686));model.x_scale.copy_(torch.linspace(.2,2,686))
    obs=torch.linspace(-9,9,686)[None];raw=obs.clone();raw[:,:312].clamp_(-5,5)
    with torch.no_grad():out=model(obs)
    reconstructed=capture.x*model.x_scale+model.x_mean
    checks.append(verified('observation normalization roundtrip and only raw-current prefix clipping',reconstructed.numpy(),raw.numpy(),obs.clamp(-5,5).numpy(),'clip tau/task fields as if raw physical fields'))
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    class Shape(torch.nn.Module):
        def forward(self, p):return torch.full((len(p),64),.64)
    measured_model=MeasuredInspirePolicy(with_tau=True).eval();measured_model.shape=Shape()
    capture2=Capture();measured_model.actor=capture2
    measured=torch.arange(347,dtype=torch.float32)[None]/100
    task=torch.arange(10,dtype=torch.float32)[None]/10
    tau=torch.arange(264,dtype=torch.float32).reshape(1,8,11,3)/1000
    with torch.no_grad():
        measured_model(measured,task,torch.zeros(1,512,3),tau=tau,tau_valid=torch.tensor([True]))
    want=torch.cat((measured[:,:345],torch.full((1,64),.64),measured[:,345:],task,tau.flatten(1),torch.ones(1,1)),-1)
    misplaced=torch.cat((measured,torch.full((1,64),.64),task,tau.flatten(1),torch.ones(1,1)),-1)
    checks.append(verified('measured686 field offsets including shape64 and force-valid flags',capture2.x.numpy(),want.numpy(),misplaced.numpy(),'insert CAD after validity flags rather than before'))
    with torch.no_grad():
        measured_model(measured,task,torch.zeros(1,512,3),tau=tau+100,tau_valid=torch.tensor([False]))
    absent=want.clone();absent[:,421:]=0
    checks.append(verified('tau-invalid observation zeroes future slice and validity',capture2.x.numpy(),absent.numpy(),want.numpy(),'retain stale predicted tau after reset'))
    pre=SimpleNamespace(phase=Phase.ACQUIRE,path_progress=0.,transitions=(),terminated=False,task_complete=False,failure_reason=None)
    post=SimpleNamespace(phase=Phase.TRACK,path_progress=.3,transitions=('grasp_acquired',),terminated=False,task_complete=False,failure_reason=None)
    before=RewardFeatures(.1,.2,.1,.2,.1,2,.02);after=RewardFeatures(.05,.08,.03,.1,.05,1,.03);action=np.full(12,.2)
    tracker=MeasuredRewardTracker();tracker.initialize(pre,before);value,parts=tracker.step(post,after,action)
    expected_parts=independent_reward(pre,post,before,after,action)
    checks.append(verified('reward independent component sum',np.array([value]),[sum(expected_parts.values())],[value-parts['event']],'omit event reward component'))
    checks.append(verified('reward every declared component',parts,expected_parts,dict(parts,time=0.),'drop time penalty'))
    tracker.step(post,after,action);paid=set(tracker.reward.paid)
    tracker.reset();tracker.initialize(pre,before);again,_=tracker.step(post,after,action)
    checks.append(verified('event reward idempotent within episode and payable after reset',dict(paid=sorted(paid),again=again),dict(paid=['grasp_acquired'],again=value),dict(paid=[],again=again-.2),'leak paid-event set across reset'))
    identity=dict(seed=194,envs=2,hz=30,controls=300,state=dict(q=np.zeros((2,18)).tolist(),tick=[0,0]))
    wrong=copy.deepcopy(identity);wrong['controls']=299
    checks.append(verified('evaluation initial state / seed / env count / seconds horizon',evaluation_identity_matches(identity,copy.deepcopy(identity)),True,evaluation_identity_matches(identity,wrong),'change rollout horizon by one control'))
    return checks


def v6_model_checks():
    """Exercise real v6 forward with a CPU geometry backend seam, never future labels."""
    import sys
    from . import paths
    tests=Path(__file__).resolve().parents[3]/'ptv3-shared-wam-v5/tests'
    sys.path.insert(0,str(tests))
    from test_v5_contract import Geometry,config,stats,batch
    from .model import SharedWAMV6
    from .contracts import SCHEMA
    from .events import EVENT_ENCODING
    cfg=dict(config(),schema=SCHEMA,event_conditioning=True,event_encoding=EVENT_ENCODING,event_dropout=.15)
    torch.manual_seed(185);model=SharedWAMV6(stats(),cfg,Geometry(8)).eval();b=batch()
    b.update(event_type=torch.tensor([1]),event_delta_steps=torch.tensor([3.]),event_valid=torch.tensor([True]),event_hz=torch.tensor([30]))
    with torch.no_grad():first=model(b)['trajectory'].numpy()
    changed=copy.deepcopy(b);changed['trajectory_target']+=10;changed['scene_target']-=10;changed['trajectory_valid'].zero_();changed['contact_target'].fill_(float('nan'))
    with torch.no_grad():isolated=model(changed)['trajectory'].numpy()
    old_encode=model.encode
    def leaking(inputs):
        leaked=dict(inputs);leaked['start_hand']=inputs['start_hand']+.01*inputs['trajectory_target'][:,0]
        return old_encode(leaked)
    model.encode=leaking
    with torch.no_grad():bad_a=model(b)['trajectory'].numpy();bad_b=model(changed)['trajectory'].numpy()
    model.encode=old_encode
    checks=[verified('real v6 future-label and future-mask isolation',isolated,first,first+(bad_b-bad_a),'inject future hand label into observed hand before encode')]
    masked=copy.deepcopy(b);masked['start_hand_valid'][:,1]=False;masked['dense_hand_valid'][:,1]=False
    with torch.no_grad():clean=model(masked)['trajectory'].numpy()
    masked['start_hand'][:,1]=float('nan');masked['start_hand_history'][:,1]=float('nan');masked['dense_hand'][:,1]=float('nan');masked['dense_history'][:,1]=float('nan')
    with torch.no_grad():dirty=model(masked)['trajectory'].numpy()
    bad_mask=dirty.copy();bad_mask[:,:,1]=1
    checks.append(verified('invalid padding mask excludes NaN hand and enforces zero output',dirty,clean,bad_mask,'turn invalid left slot into finite nonzero prediction'))
    return checks
