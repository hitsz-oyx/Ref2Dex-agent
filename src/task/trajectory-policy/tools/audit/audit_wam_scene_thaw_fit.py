"""CPU array/checkpoint audit; no model inference or training."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2]
root=TASK.parents[2]
sys.path.insert(0,str(root/'src/task/trajectory-policy/tools/audit'))
from audit_wam_ref2_shadow import sha, write
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fit',type=Path,required=True)
fit=parser.parse_args().fit.resolve()
if root/'outputs/trajectory-policy' not in fit.parents:
    raise ValueError('task-owned fit required')
m=json.loads((fit/'manifest.json').read_text()); r=json.loads((fit/'result.json').read_text())
assert m['status']==r['status']=='COMPLETED'
assert sha(fit/'result.json')==m['result_sha256']
assert sha(fit/'predictions.npz')==m['predictions_sha256']
assert sha(fit/'head.pt')==r['head_sha256']
assert all(sha(p)==h for p,h in m['input_sha256'].items())
with np.load(fit/'predictions.npz',allow_pickle=False) as f:
    a={k:f[k].copy() for k in f.files}
assert all(np.isfinite(v).all() for v in a.values())
rows=a['rows']; recomputed={}; roles={}; errors=[]
for phase in ('initial','final'):
    recomputed[phase]={}
    for panel in ('nominal_report','query_training','query_validation'):
        prefix=phase+'_'+panel+'_'; selected=rows[a[prefix+'row_ids']]
        d=a[prefix+'pd']-a[prefix+'pd_true']; xyz=np.linalg.norm(d[:,:3],axis=1)
        metrics=dict(xyz_p95_mm=float(np.percentile(xyz,95)*1000),
            rotation_p95_rad=float(np.percentile(np.abs(d[:,3:6]).max(-1),95)),
            finger_p95_rad=float(np.percentile(np.abs(d[:,[6,8,10,12,14,15]]).max(-1),95)))
        recomputed[phase][panel]=metrics
        errors.extend(abs(metrics[k]-r[phase][panel][k]) for k in metrics)
        if panel!='nominal_report':
            assert not a[prefix+'tau_label_valid'].any()
            for role in (2,3):
                mask=selected[:,2]==role
                roles[phase+'_'+panel+'_role'+str(role)]=dict(n=int(mask.sum()),
                    ticks=[int(selected[mask,1].min()),int(selected[mask,1].max())],
                    xyz_p95_mm=float(np.percentile(xyz[mask],95)*1000),
                    xyz_median_mm=float(np.median(xyz[mask])*1000))

nominal=root/'outputs/trajectory-policy/competent-pd-teacher-20261011-r2/trajectory.npz'
live=root/'outputs/trajectory-policy/wam-ref2-execution-recovery-20261012-r2/native/trajectory.npz'
with np.load(nominal,allow_pickle=False) as f:
    nominal_pd=f['pd_targets'].copy()
    nominal_hand=f['hand'].copy(); nominal_obj=f['obj'].copy()
with np.load(live,allow_pickle=False) as f: live_pd=f['expert_PD'].copy()
truth_max=0.
for panel in ('nominal_report','query_training','query_validation'):
    pref='final_'+panel+'_'; selected=rows[a[pref+'row_ids']]
    truth=np.stack([nominal_pd[t,e] if kind==0 else live_pd[t,e-2] for kind,t,e in selected])
    truth_max=max(truth_max,float(np.abs(truth-a[pref+'pd_true']).max()))
for phase in ('initial','final'):
    pref=phase+'_nominal_report_'; selected=rows[a[pref+'row_ids']]
    # Direct pure-array displacement labels, independently of the fitter cache.
    tau=np.stack([np.einsum('tpj,jk->tpk',
        nominal_hand[t+np.arange(1,25),e]-nominal_hand[t,e],nominal_obj[t,e,:3,:3])
        for _,t,e in selected]).astype(np.float32)
    predicted=a[pref+'tau'].reshape(-1,24,11,3)
    epe=float(np.linalg.norm(predicted-tau,axis=-1)[:,:8].mean()*1000)
    errors.append(abs(epe-r[phase]['nominal_report']['first8_epe_mm']))
    recomputed[phase]['nominal_report']['first8_epe_mm']=epe
train=rows[a['final_query_training_row_ids']]; val=rows[a['final_query_validation_row_ids']]
assert len(train)==128 and len(val)==32
assert not (set(map(tuple,train))&set(map(tuple,val)))
assert not ((np.r_[train[:,2],val[:,2]]==3)&(np.r_[train[:,1],val[:,1]]<16)).any()
assert (np.r_[train[:,1],val[:,1]]<128).all()

base_path=root/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
base=torch.load(base_path,map_location='cpu',weights_only=False)
packet=torch.load(fit/'head.pt',map_location='cpu',weights_only=False)
assert packet['base_checkpoint_sha256']==sha(base_path)==m['base_checkpoint_sha256']
assert set(packet['head'])=={'trajectory','hand_query'}
prefix='backbone.dec.dec0.'
scene={k[len('scene.'):]:v for k,v in base['model'].items() if k.startswith('scene.')}
expected={k:v for k,v in scene.items() if k.startswith(prefix)}
patch=packet['scene_patch']; assert set(patch)==set(expected)
assert packet['scene_patch_schema']==m['scene_patch_schema']=='ref2dex.wam-scene-dec0.v1'
assert packet['parent_head_sha256']==m['warm_head_sha256']
for k,v in patch.items():
    assert v.shape==expected[k].shape and v.dtype==expected[k].dtype and torch.isfinite(v).all()
for net in packet['head'].values():
    assert all(torch.isfinite(v).all() for v in net.values())
def network_digest(state):
    digest=hashlib.sha256()
    for k,v in state.items():
        digest.update(k.encode()); digest.update(v.detach().float().numpy().tobytes())
    return digest.hexdigest()
assert network_digest(scene)==r['scene_initial_hash']
scene.update(patch); assert network_digest(scene)==r['scene_final_hash']
for name in ('goal','action','world'):
    state={k[len(name)+1:]:v for k,v in base['model'].items() if k.startswith(name+'.')}
    assert network_digest(state)==r['frozen_initial_hash'][name]==r['frozen_final_hash'][name]
assert r['frozen_scene_remainder_initial_hash']==r['frozen_scene_remainder_final_hash']
assert r['frozen_initial_hash']==r['frozen_final_hash']
assert all(v>0 for v in r['live_PD_actor_gradient_sum'].values())
assert max(errors)<1e-5 and truth_max<1e-6
signals=dict(query_PD_halved=recomputed['final']['query_validation']['xyz_p95_mm']<=recomputed['initial']['query_validation']['xyz_p95_mm']*.5,
    nominal_first8_no_degradation_gt25pct=recomputed['final']['nominal_report']['first8_epe_mm']<=r['retention_limits']['first8_epe_mm'],
    nominal_PD_retained=recomputed['final']['nominal_report']['xyz_p95_mm']<=r['retention_limits']['nominal_PD_xyz_p95_mm'])
assert signals==r['adaptation_signal']
report=dict(pass_=True,scope='CPU checkpoint and pure array statistics; no model inference',
    metric_max_difference=max(errors),source_PD_label_max_abs=truth_max,
    signal=signals,role_metrics=roles,scene_patch_keys=len(patch),scene_patch_hash_reconstructed=True,
    inputs={str(p):sha(p) for p in (Path(__file__),fit/'head.pt',fit/'predictions.npz',fit/'manifest.json',fit/'result.json',base_path,nominal,live)})
report['pass']=report.pop('pass_')
write(fit/'root_fit_audit.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2))
