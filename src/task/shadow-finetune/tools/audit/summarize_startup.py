"""GPU readout and matching audit for the bounded startup-sampling intervention."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))


def implementation_check(old_root,new_root):
    checked=[]
    for folder in ['ptv3-shared-wam-v2','ptv3-shared-wam','cm-pointflow-effect-pretrain','goal-point-wam','trajectory-policy']:
        prefix=Path('src/task')/folder/'src'
        for old in sorted((old_root/prefix).rglob('*.py')):
            relative=old.relative_to(old_root);new=new_root/relative
            if not new.is_file() or old.read_bytes()!=new.read_bytes():
                raise ValueError('forecast implementation changed: '+str(relative))
            checked.append(str(relative))
    for name in ['model.py','execution.py','paths.py']:
        relative=Path('src/task/shadow-finetune/src/shadow_finetune')/name
        if (old_root/relative).read_bytes()!=(new_root/relative).read_bytes():
            raise ValueError('Shadow model/execution changed: '+name)
        checked.append(str(relative))
    relative=Path('src/task/shadow-finetune/src/shadow_finetune/data.py')
    a=ast.parse((old_root/relative).read_text());b=ast.parse((new_root/relative).read_text())
    old_nodes={n.name:n for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new_nodes={n.name:n for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name,node in old_nodes.items():
        other=new_nodes[name]
        if isinstance(node,ast.ClassDef):
            methods={n.name:n for n in other.body if isinstance(n,ast.FunctionDef)}
            for method in node.body:
                if isinstance(method,ast.FunctionDef) and ast.dump(method)!=ast.dump(methods[method.name]):
                    raise ValueError('existing data method changed: '+method.name)
        elif ast.dump(node)!=ast.dump(other):raise ValueError('existing data function changed: '+name)
    return dict(status='PASS',identical_model_files=checked,existing_data_functions_and_methods='identical AST')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=6);a=p.parse_args();run=a.run.resolve()
    c=json.loads((run/'controls.json').read_text());reference=Path(c['reference_run'])
    def gpu_pids(gpu):
        s=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        return {int(x) for x in s.split() if x.isdigit()}
    if gpu_pids(a.gpu):raise RuntimeError('selected GPU occupied')
    if a.gpu==6 and gpu_pids(7):raise RuntimeError('temporary test requires reserve GPU7')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    from shadow_finetune import paths
    from shadow_finetune.data import ShadowWindows,collate
    from shadow_finetune.model import ShadowPolicy,loss
    from shadow_finetune.scale_controls import load_controls,window_key
    from shadow_finetune.metrics import native_report
    from ptv3_shared_wam.loading import to_device
    import torch
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    started=time.monotonic();m={};out=run/'summary';out.mkdir(exist_ok=False)
    for name,path in [('uniform',reference/'train-full'),('startup',run/'train-startup')]:
        m[name]=json.loads((path/'manifest.json').read_text())
        if m[name]['status']!='COMPLETED':raise ValueError('incomplete arm: '+name)
    for key in ('parent','parent_step','config','stats','parameters','seed','updates','batch_size','data','normalization_source'):
        if m['uniform'][key]!=m['startup'][key]:raise ValueError('matched settings differ: '+key)
    if m['startup']['comparison']!=c or m['startup']['sampler']['startup_fraction']!=.5:
        raise ValueError('startup intervention mismatch')
    code_check=implementation_check(Path(m['uniform']['code_root']),Path(m['startup']['code_root']))
    with np.load(reference/'train-full/initial-actor.npz') as x,np.load(run/'train-startup/initial-actor.npz') as y:
        if set(x.files)!=set(y.files) or any(not np.array_equal(x[k],y[k]) for k in x.files):
            raise ValueError('actual initial actor tensors differ')
    s=c['settings'];_,dev,validation,_=load_controls(run/'controls.json',m['startup']['data'],s['parent'],s['seed'],s['updates'],s['batch_size'],.5)
    starts=c['startup_validation_ids']
    if [window_key(dev,i) for i in starts]!=c['startup_validation_keys'] or any(dev.keys[i][1]!=0 for i in starts):
        raise ValueError('startup DEV physical identity mismatch')
    panels=dict(common_dev=validation,startup_dev=starts);metrics={};native={};teacher={};exposure={}
    train=ShadowWindows(m['startup']['data'],'train')
    fields=('q','dq','object','hand','hand_root','observation','goal','success','reset')
    for arm,train_path,native_path,fraction in [
        ('uniform',reference/'train-full',reference/'native-final-full',0.),
        ('startup',run/'train-startup',run/'native-final-startup',.5)]:
        if not np.array_equal(np.load(train_path/'validation.npy'),validation):raise ValueError('common DEV differs')
        packet=torch.load(train_path/'step1000.pt',map_location='cpu',weights_only=False)
        rng=np.random.default_rng(s['seed']);counts=dict(samples=0,startup=0,tick0=0)
        for _ in range(s['updates']):
            ids=train.draw_startup(rng,s['batch_size'],fraction)
            counts['samples']+=len(ids);counts['startup']+=sum(train.keys[i][1]<8 for i in ids)
            counts['tick0']+=sum(train.keys[i][1]==0 for i in ids)
        if rng.bit_generator.state!=packet['rng']:raise ValueError('actual sampling RNG differs: '+arm)
        if arm=='startup' and counts!=m[arm]['sampling_counts']:raise ValueError('actual sampling counts differ')
        exposure[arm]=dict(counts,final_rng_exact=True)
        model=ShadowPolicy(packet['stats'],packet['config']).cuda().eval()
        model.load_state_dict(packet['model'],strict=True);config=packet['config'];del packet
        metrics[arm]={}
        for panel,ids in panels.items():
            terms_sum={};predictions=[];targets=[]
            with torch.no_grad():
                for offset in range(0,len(ids),16):
                    selected=ids[offset:offset+16];b=to_device(collate([dev[int(i)] for i in selected]),'cuda')
                    pred=model(b);value,terms=loss(pred,b,config,model)
                    for k,v in dict(total=value,**terms).items():terms_sum[k]=terms_sum.get(k,0.)+float(v)*len(selected)/len(ids)
                    predictions.append(pred['action'][:,0].cpu().numpy());targets.append(b['action_target'][:,0].cpu().numpy())
            pred=np.concatenate(predictions);target=np.concatenate(targets)
            if not np.isfinite(pred).all():raise ValueError('nonfinite readout')
            row=dict(count=len(ids),losses=terms_sum)
            for mode,values in [('raw',pred),('clipped',np.clip(pred,-1,1))]:
                err=np.abs(values-target)
                row[mode]={name:float(err[:,part].mean()) for name,part in
                    [('all',slice(None)),('force',slice(0,3)),('torque',slice(3,6)),('fingers',slice(6,24))]}
            if panel=='common_dev':
                original=json.loads((train_path/'curve.json').read_text())[-1]
                diff={k:abs(v-original[k]) for k,v in terms_sum.items()}
                diff['command_first_mae']=abs(row['raw']['all']-original['command_first_mae'])
                if any(v>(1e-5 if k in ('action','root','finger','command_first_mae') else 1e-4) for k,v in diff.items()):
                    raise ValueError('saved common DEV replay differs: '+str(diff))
                row['saved_metric_abs_differences']=diff
            metrics[arm][panel]=row
        del model;torch.cuda.empty_cache()
        n=json.loads((native_path/'comparison.json').read_text())
        if n['status']!='COMPLETED' or n['seed']!=c['native_reset_seed'] or max(n['initial_state_max_error'].values())>1e-6:
            raise ValueError('native pairing failed')
        with np.load(native_path/'teacher/episode-00.npz') as z:teacher[arm]={k:z[k].copy() for k in fields}
        identities=json.loads((native_path/'teacher/environment-objects.json').read_text())
        if identities!=json.loads((native_path/'student/environment-objects.json').read_text()):raise ValueError('native objects differ')
        with np.load(native_path/'student/episode-00.npz') as z:
            actual=native_report(z,identities)
            for key in ('official_success','final_within5cm','held60_with_lift10cm','by_split'):
                if actual[key]!=n['roles']['student'][key]:raise ValueError('native metric mismatch: '+key)
            clock=float(np.abs(z['observation'][1:,:,167:191]-z['action']).max())
            if clock>1e-6 or not np.isfinite(z['action']).all() or np.abs(z['action']).max()>1.00001:
                raise ValueError('native command clock/range failed')
        native[arm]=dict(report=n,environment_objects=identities,command_clock_max_error=clock)
        (out/'partial-readout.json').write_text(json.dumps(dict(status='RUNNING',metrics=metrics,native=native,
            sampling_exposure=exposure),indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(arm=arm,common_dev=metrics[arm]['common_dev']['raw'],startup_dev=metrics[arm]['startup_dev']['raw'])),flush=True)
        if time.monotonic()-started>600:raise TimeoutError('ten-minute comparison readout')
        if torch.cuda.max_memory_allocated()>4*2**30:raise RuntimeError('4GiB readout budget')
        if gpu_pids(a.gpu)-{os.getpid()} or (a.gpu==6 and gpu_pids(7)):raise RuntimeError('GPU reserve/ownership changed')
    if native['uniform']['environment_objects']!=native['startup']['environment_objects']:raise ValueError('cross-arm native panel differs')
    trace_error={k:float(np.abs(teacher['uniform'][k].astype(float)-teacher['startup'][k]).astype(float).max()) for k in fields}
    if max(trace_error.values())>1e-6:raise ValueError('cross-arm full teacher trace differs')
    gain=lambda panel,key:1-metrics['startup'][panel]['raw'][key]/max(metrics['uniform'][panel]['raw'][key],1e-12)
    fit= gain('startup_dev','all')>=.3 and gain('startup_dev','torque')>=.3 and gain('common_dev','all')>=-.2
    held={arm:v['report']['roles']['student']['held60_with_lift10cm'] for arm,v in native.items()}
    invalid=any(v['report']['roles']['student']['controller_audit']['invalid_envs'] for v in native.values())
    result=dict(status='COMPLETED',matched_checks='PASS',implementation_check=code_check,initial_actor_exact=True,
        metrics=metrics,native=native,sampling_exposure=exposure,teacher_full_trace_max_errors=trace_error,
        startup_mae_gain=gain('startup_dev','all'),startup_torque_gain=gain('startup_dev','torque'),
        common_dev_mae_gain=gain('common_dev','all'),held=held,invalid_inputs=bool(invalid),
        fitting_signal='PROMISING' if fit else 'UNCLEAR',
        signal='PROMISING' if fit and held['startup']-held['uniform']>=2 and not invalid else 'UNCLEAR',
        elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        scope='single-seed startup-sampling intervention using a frozen prior uniform control; not causal proof or Inspire attribution')
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','signal','fitting_signal','startup_mae_gain','startup_torque_gain','common_dev_mae_gain','held')}))


if __name__=='__main__':main()
