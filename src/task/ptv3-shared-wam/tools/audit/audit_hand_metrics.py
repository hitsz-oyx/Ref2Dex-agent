"""CPU-only aggregation and fixed-validation label audit; no model execution."""
from pathlib import Path
import argparse
import sys, json
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(ROOT/'src/task/ptv3-shared-wam/src'))
from ptv3_shared_wam.data import SharedWindows, shared_collate
from ptv3_shared_wam.objectives import forecast_metrics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, default=ROOT/'outputs/ptv3-shared-wam/pretrain-main3-ddp4-80k-20261013-r2')
parser.add_argument('--validation-step', type=int, default=37000)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
run = args.run.resolve()
output = args.output.resolve()
if output.exists() or ROOT not in output.parents:
    raise ValueError('fresh repository-local JSON output required')
output.parent.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
manifest = json.loads((run/'manifest.json').read_text())
dataset = SharedWindows(manifest['provenance']['data'], 'val')
indices = np.load(run/'validation.npy')
sample = dataset[int(indices[0])]
batch = shared_collate([sample])
pred = {'trajectory': batch['trajectory_target'].clone(), 'scene': batch['scene_target'].clone()}
pred['trajectory'][...,0] += torch.arange(1,25)[None,:,None,None] * .001
metrics = forecast_metrics(pred,batch,manifest['config'])
mean = metrics['hand_all24']['sum_mm']/metrics['hand_all24']['count']
first = metrics['hand_first8']['sum_mm']/metrics['hand_first8']['count']
assert abs(mean-12.5)<1e-4 and abs(first-4.5)<1e-4,(mean,first)
print('PASS actual forecast_metrics: errors 1..24 mm -> all24=%.4f mm, first8=%.4f mm'%(mean,first),flush=True)
acc = {s:dict(displacements=[],steps=[],spans=[],windows=[],velocities=[]) for s in range(3)}
for index in indices:
    s = dataset[int(index)]
    sid = int(s['source_id']); dst = acc[sid]
    target = s['trajectory_target']; valid = s['trajectory_valid'] & s['start_hand_valid'][None]
    assert target.shape == (24,2,11,3)
    # The reported frozen panel has 128*24*22 valid entries per source.
    # Fail rather than silently use unmasked averages on a different panel.
    assert valid.all(), 'this audit requires the fully valid original panel'
    assert np.isfinite(target[valid]).all()
    assert np.allclose(s['action'].reshape(24,2,11,9)[...,:3][valid], (target+s['start_hand'][None])[valid],atol=1e-6)
    magnitude = np.linalg.norm(target,axis=-1)*1000
    dst['displacements'].append(magnitude)
    dst['windows'].append((float(magnitude[valid].mean()),float(magnitude[-1][valid[-1]].mean())))
    velocity = s['start_hand_history'][...,:3]
    constant_velocity = np.arange(1,25)[:,None,None,None]/30*velocity[None]
    dst['velocities'].append(np.linalg.norm(constant_velocity-target,axis=-1)[valid]*1000)
    step = np.diff(np.concatenate([np.zeros_like(target[:1]),target],axis=0),axis=0)
    dst['steps'].append(np.linalg.norm(step,axis=-1)[valid]*1000)
    for h in range(2):
        if s['start_hand_valid'][h].all():
            dst['spans'].append(np.linalg.norm(s['start_hand'][h,1:]-s['start_hand'][h,0],axis=-1)*1000)

curve = next(row for row in json.loads((run/'curve.json').read_text())
             if row['step'] == args.validation_step)
result = dict(schema='ref2dex.hand-metric-audit.v1',run_id=run.name,validation_step=curve['step'],aggregation_check=dict(all24_mm=mean,first8_mm=first),sources={})
for sid,name in enumerate(['OakInk2','GRAB','ARCTIC']):
    dst=acc[sid]; d=np.stack(dst['displacements']); steps=np.concatenate(dst['steps']); windows=np.array(dst['windows'])
    report=dict(windows=len(d),no_motion_baseline_first8_mm=float(d[:,:8].mean()),no_motion_baseline_all24_mm=float(d.mean()),no_motion_baseline_h24_mm=float(d[:,-1].mean()),constant_velocity_all24_mm=float(np.concatenate(dst['velocities']).mean()),step_motion_median_mm=float(np.median(steps)),step_motion_p99_mm=float(np.percentile(steps,99)),hand_span_median_mm=float(np.median(np.concatenate(dst['spans']))),window_motion_mean_quantiles_mm=np.percentile(windows[:,0],[0,50,90,99,100]).tolist(),model_first8_mm=curve['metrics'][f'source{sid}/hand_first8']['mean_mm'],model_all24_mm=curve['metrics'][f'source{sid}/hand_all24']['mean_mm'])
    all24=curve['metrics'][f'source{sid}/hand_all24']
    first8=curve['metrics'][f'source{sid}/hand_first8']
    report['model_later16_mm']=(all24['mean_mm']*all24['count']-first8['mean_mm']*first8['count'])/(all24['count']-first8['count'])
    result['sources'][name]=report
    print(name,json.dumps(report),flush=True)
output.write_text(json.dumps(result,indent=2)+'\n')
print('REPORT',output,flush=True)
