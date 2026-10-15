"""Paired physical-unit readout of v2 main3 and v3 nine-source checkpoints."""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v3.data import NineWindows, collate
from ptv3_shared_wam_v3.model import SharedWAMV3
from ptv3_shared_wam_v3.checkpoint import atomic_json
from ptv3_shared_wam_v2.model import SharedWAMV2
from ptv3_shared_wam.data import SharedWindows
from ptv3_shared_wam.loading import to_device, project_ipc_directory
from ptv3_shared_wam.objectives import forecast_metrics
from oakink_wm.multisource import mixed_indices

CORE=('hand_first8','hand_all24','scene_first8','scene_all24','scene_target_first8','scene_target_all24','scene_moving','scene_static')
ARCH=('point_width','shared_width','shared_heads','shared_layers','head_width','head_heads','ptv3',
      'scene_tokens','patch_size','voxel_m','voxel_origin_m','position_scale_m','velocity_scale_mps',
      'acceleration_scale_mps2','hand_in_ptv3','explicit_hand','object_points','history','horizon','fps',
      'goal_encoding','task_conditioning')


def recording_identity(data,index,sample):
    sid=int(sample['source_id'])
    if sid<3:return f'{sid}:'+str(np.asarray(sample['sample_id'])[0])
    row=data.native.meta['sequences'][int(data.native.index[index-len(data.main)][0])]
    return '9:'+row['source_sequence'] if sid==9 else str((data.native.root/row['path']).resolve())


def aggregate(rows):
    total={}
    for name in rows[0]['metrics']:
        n=sum(r['metrics'][name]['count'] for r in rows)
        value=sum(r['metrics'][name]['sum_mm'] for r in rows)
        total[name]=dict(mean_mm=value/n if n else None,count=n)
    return total


def paired(old,new):
    assert len(old)==len(new)
    groups=defaultdict(list)
    for i,(a,b) in enumerate(zip(old,new)):
        assert (a['index'],a['capture'])==(b['index'],b['capture'])
        groups[a['capture']].append(i)
    ids=list(groups.values());result={};rng=np.random.default_rng(138)
    resamples=rng.integers(len(ids),size=(2000,len(ids)))
    for name in CORE:
        x=np.array([sum(old[i]['metrics'][name]['sum_mm'] for i in g) for g in ids])
        y=np.array([sum(new[i]['metrics'][name]['sum_mm'] for i in g) for g in ids])
        n=np.array([sum(old[i]['metrics'][name]['count'] for i in g) for g in ids])
        assert all(old[i]['metrics'][name]['count']==new[i]['metrics'][name]['count'] for i in range(len(old)))
        if n.sum()==0:continue
        denominator=n[resamples].sum(1);valid=denominator>0
        difference=(y-x)[resamples].sum(1)[valid]/denominator[valid]
        result[name]=dict(old_mm=float(x.sum()/n.sum()),new_mm=float(y.sum()/n.sum()),
            improvement_pct=float(100*(1-y.sum()/x.sum())) if x.sum()>0 else None,
            new_minus_old_mm=float((y-x).sum()/n.sum()),
            capture_bootstrap_ci95_new_minus_old_mm=np.quantile(difference,[.025,.975]).tolist(),
            captures=len(ids),windows=len(old))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-run',type=Path,required=True);p.add_argument('--new-checkpoint',type=Path,required=True)
    p.add_argument('--new-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();assert not out.exists()
    if shutil.disk_usage(out.parent).free<20*2**30:raise RuntimeError('disk floor')
    torch.set_num_threads(2);torch.cuda.set_device(0);free,total=torch.cuda.mem_get_info()
    if free<8*2**30:raise RuntimeError('GPU preflight8GiB')
    torch.cuda.set_per_process_memory_fraction(3*2**30/total)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    project_ipc_directory();out.mkdir();started=time.monotonic();last_parent=0;slow=0
    progress=dict(status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=TASK.parents[2],text=True).strip(),allocator_limit_mib=3072)
    def guard():
        nonlocal last_parent,slow
        free,_=torch.cuda.mem_get_info();elapsed=time.monotonic()-started
        if free<4*2**30 or elapsed>900 or (out/'STOP').exists() or shutil.disk_usage(out).free<20*2**30:
            raise RuntimeError('bounded diagnostic stop condition')
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>100_000_000:
            raise RuntimeError('100MB diagnostic artifact cap')
        if elapsed-last_parent>60 or last_parent==0:
            parent=json.loads((a.new_run/'progress.json').read_text());progress['parent_training']=parent
            slow=slow+1 if parent.get('updates_per_s',1)<.9 else 0;last_parent=elapsed
            if slow>=3:raise RuntimeError('sustained parent throughput interference')
        progress.update(elapsed_s=elapsed,own_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        own_reserved_mib=torch.cuda.memory_reserved()/2**20,gpu_free_mib=free/2**20)
    try:
        recipe=json.loads((a.new_run/'recipe.json').read_text())
        data=NineWindows(recipe['main3_root'],recipe['native_root'],'dev',recipe)
        original=SharedWindows(recipe['main3_root'],'val',32)
        original_config=json.loads((a.old_run/'manifest.json').read_text())['config']
        regenerated=mixed_indices(original.base,original_config['validation_samples'],original_config['validation_seed'],equal_sources=True)
        saved=a.old_run/'validation.npy'
        if saved.exists():assert np.array_equal(np.load(saved,allow_pickle=False),regenerated)
        selected=[int(i) for i in regenerated if not data.blocked[int(i)]]
        excluded=[int(i) for i in regenerated if data.blocked[int(i)]]
        panel=json.loads((a.new_run/'validation-panel.json').read_text())
        for sid,indices in panel.items():
            if int(sid)>=4:selected.extend(map(int,indices))
        assert len(selected)<=544
        atomic_json(out/'panel.json',dict(legacy_seed=original_config['validation_seed'],
            legacy_indices=regenerated.tolist(),excluded=excluded,selected_indices=selected,
            native_panel={k:v for k,v in panel.items() if int(k)>=4},test_used=False))
        arms={'v2-best59k':a.old_run/'step59000.pt','v2-final80k':a.old_run/'step80000.pt','v3-current40k':a.new_checkpoint}
        configs={};arm_meta={};all_rows={};architecture=None
        for arm,path in arms.items():
            guard();packet=torch.load(path,map_location='cpu',weights_only=False)
            config=packet['config'];stats=packet['stats'];step=packet['step'];schema=packet['schema']
            structure={k:config[k] for k in ARCH}
            if architecture is None:architecture=structure
            else:assert architecture==structure,'architecture mismatch'
            if arm.startswith('v2'):
                assert stats['normalization_source_manifest_sha256']==original.base.meta['normalization_source_manifest_sha256']
                model=SharedWAMV2(stats,config)
            else:
                assert stats['recipe']==recipe and step==40000
                model=SharedWAMV3(stats,config)
            model.load_state_dict(packet['model'],strict=True)
            for name in ('hand','scene'):
                for field in ('mean','std'):assert torch.equal(getattr(model.normalizer,name+'_'+field),torch.tensor(stats[name][field],dtype=torch.float32))
            arm_meta[arm]=dict(checkpoint=str(path.resolve()),step=step,schema=schema,seed=config['seed'],
                training_global_batch=config['batch_size']*packet['provenance']['world_size'],normalizer='own frozen TRAIN statistics',
                parameters=model.parameter_counts())
            configs[arm]=config;del packet;gc.collect();model=model.cuda().eval();rows=[]
            for first in range(0,len(selected),16):
                guard();bound=[data[i] for i in selected[first:first+16]]
                batch=to_device(collate(bound)['model_inputs'],torch.device('cuda',0))
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(batch)
                for field in ('trajectory','scene'):
                    if not torch.isfinite(prediction[field]).all():raise ValueError('nonfinite prediction')
                for j,(sample,provenance,_) in enumerate(bound):
                    sid=int(sample['source_id']);index=selected[first+j]
                    metrics=forecast_metrics({k:v[j:j+1] for k,v in prediction.items()},
                        {k:v[j:j+1] for k,v in batch.items()},config)
                    capture=recording_identity(data,index,sample)
                    rows.append(dict(index=index,source_id=sid,capture=capture,metrics=metrics))
                del prediction,batch,bound
                if first%64==0 or first+16>=len(selected):
                    progress.update(arm=arm,processed=len(rows),target=len(selected));guard()
                    atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            all_rows[arm]=rows
            atomic_json(out/(arm+'-windows.json'),rows)
            del model;gc.collect();torch.cuda.empty_cache()
        sources=sorted({r['source_id'] for r in all_rows['v3-current40k']})
        summary=dict(status='COMPLETED',arms=arm_meta,panel_windows=len(selected),excluded_legacy=len(excluded),
            metrics={arm:{str(sid):aggregate([r for r in rows if r['source_id']==sid]) for sid in sources} for arm,rows in all_rows.items()},
            paired={arm:{str(sid):paired([r for r in all_rows[arm] if r['source_id']==sid],
                                       [r for r in all_rows['v3-current40k'] if r['source_id']==sid]) for sid in sources}
                    for arm in ('v2-best59k','v2-final80k')},
            inference=dict(batch_size=16,bf16=True,tf32=True,checkpoint_selection='old best/final and current40k milestone; no TEST',
                           elapsed_s=time.monotonic()-started),limitations='Single trajectories, different training seeds/schedules/normalizers/source exposure; paired DEV comparison is not a causal multisource or robot-benefit validation.')
        # Reproduction control on the old exact panel (batch16, same order).
        if not excluded:
            history=json.loads((a.old_run/'curve.json').read_text());checks={}
            for arm in ('v2-best59k','v2-final80k'):
                old=next(x for x in history if x['step']==arm_meta[arm]['step']);diff=[]
                for sid in range(3):
                    for name in CORE:
                        v=summary['metrics'][arm][str(sid)][name];ref=old['metrics'][f'source{sid}/{name}']
                        assert v['count']==ref['count']
                        if v['mean_mm'] is not None:diff.append(abs(v['mean_mm']-ref['mean_mm']))
                checks[arm]=dict(max_absolute_metric_difference_mm=max(diff),tolerance_mm=.1,passed=max(diff)<.1)
                if not checks[arm]['passed']:raise ValueError('legacy physical metric reproduction exceeds0.1mm')
            summary['legacy_reproduction']=checks
        atomic_json(out/'summary.json',summary);progress.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
        atomic_json(out/'progress.json',progress);print(json.dumps(dict(status='COMPLETED',windows=len(selected),output=str(out))),flush=True)
    except BaseException as error:
        progress.update(status='FAILED',error=repr(error));atomic_json(out/'progress.json',progress);raise


if __name__=='__main__':main()
