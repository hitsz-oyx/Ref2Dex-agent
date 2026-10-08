"""Bounded matched main-only / main+ContactPose history-only auxiliary Probe.

Both arms consume identical main/auxiliary batches and forward/backward calls.
Only the auxiliary loss coefficient differs (0 / predeclared weight).
No future ContactPose hands enter either auxiliary forward pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from oakink_wm.transport_auxiliary import (history_inputs,validate_transport_source,
                                         validate_auxiliary_config,classify_auxiliary,HISTORY_INPUTS)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def write(path,value):
    part = path.with_suffix('.partial')
    part.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    part.replace(path)


def file_signature(path):
    stat = Path(path).stat()
    return (stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-data',type=Path,required=True)
    parser.add_argument('--four-source-data',type=Path,required=True,help='frozen descriptor source; only ContactPose is used')
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--stats',type=Path,required=True)
    parser.add_argument('--loss-stats',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=TASK/'configs/contactpose_transport_auxiliary_probe.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--engineering',action='store_true',help='two real updates with small batches, no Probe verdict')
    args = parser.parse_args()
    out = args.output.resolve()
    if ROOT/'outputs/cm-pointflow-effect-pretrain' not in out.parents or out.exists():
        parser.error('fresh task-owned output required')
    config = json.loads(args.config.read_text());validate_auxiliary_config(config)
    if args.engineering:
        config.update(updates=2,main_batch=2,aux_batch=2,seconds=600,seed=27)
    occupied = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied before model load: '+occupied)
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('free disk below20GiB')
    started = time.time();deadline = started+config['seconds'];stop = [False]
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda signum,frame:stop.__setitem__(0,True))
    scratch = ROOT/'tmp/contactpose-transport-auxiliary';scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),TMPDIR=str(scratch),
                      PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    import numpy as np
    import torch
    from oakink_wm.multisource import MixedWindows,SourceWindows,mixed_indices,MAIN_SOURCE_NAMES
    from oakink_wm.loss_normalization import validate_loss_statistics,physical_loss
    from oakink_wm.pointworld_temporal import model_from_config,VENDOR
    from oakink_wm.pointworld_performance import install_fused_hilbert
    import train_oakink2_pointworld_temporal as base
    base.configure_numerics();install_fused_hilbert()
    out.mkdir(parents=True)
    frozen,signatures = {},{}
    def pin(path,expected=None):
        if stop[0] or time.time() >= deadline:raise TimeoutError('bounded auxiliary preparation deadline')
        path = Path(path).absolute();actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('source/input identity mismatch: '+str(path))
        frozen[str(path)] = actual;signatures[str(path)] = file_signature(path)
    def check(full=False):
        if stop[0] or time.time() >= deadline:
            raise TimeoutError('bounded auxiliary stop/deadline')
        for path,expected in signatures.items():
            if file_signature(path) != expected or (full and sha(path) != frozen[path]):
                raise RuntimeError('source/input drift: '+path)
        pids = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
        if any(int(pid.strip()) != os.getpid() for pid in pids if pid.strip()):
            raise RuntimeError('foreign GPU process; stop only this owned run')
        if shutil.disk_usage(ROOT).free < 20*2**30:
            raise RuntimeError('free disk below20GiB')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 2*2**30:
            raise RuntimeError('auxiliary output exceeded2GiB')
    manifest = dict(schema='pointworld.transport-auxiliary.probe.v1',status='PREPARING',
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    run_id=out.name,config=config,pid=os.getpid(),physical_gpu=args.gpu,
                    deadline=deadline,started_at=started,engineering_only=args.engineering,
                    sources=frozen,arms={},supervision='ContactPose history-only rigid transport auxiliary',
                    limitation='same formerly four-source initialized parent; incremental auxiliary effect only')
    try:
        for path in (args.config,args.parent,args.stats,args.loss_stats,
                     args.main_data/'processed/manifest.json',args.four_source_data/'processed/manifest.json',
                     Path(__file__),*sorted((TASK/'src/oakink_wm').glob('*.py')),
                     TASK/'tools/run/train_oakink2_pointworld_temporal.py'):
            pin(path)
        for path in sorted((VENDOR/'ptv3').rglob('*')):
            if path.is_file() and path.suffix in ('.py','.yaml'):
                pin(path)
        train,val = MixedWindows(args.main_data,'train'),MixedWindows(args.main_data,'val')
        if tuple(d['name'] for d in train.meta['sources']) != MAIN_SOURCE_NAMES:
            raise ValueError('main loss must contain only OakInk2/GRAB/ARCTIC')
        four = json.loads((args.four_source_data/'processed/manifest.json').read_text())
        descriptors = [d for d in four['sources'] if d['name']=='contactpose']
        if len(descriptors)!=1:
            raise ValueError('one frozen ContactPose descriptor required')
        descriptor = descriptors[0]
        cp_train,cp_val = SourceWindows(descriptor,'train'),SourceWindows(descriptor,'val')
        validate_transport_source(cp_train.meta,descriptor)
        if not len(cp_train.groups[0]) or not len(cp_val.groups[0]):
            raise ValueError('moving transport windows required in train and held-out val')
        pin(Path(descriptor['root'])/'processed/manifest.json',descriptor['manifest_sha256'])
        for desc in [*train.meta['sources'],descriptor]:
            pin(Path(desc['root'])/'processed/manifest.json',desc['manifest_sha256'])
            for split in ('train','val'):
                index=desc['indices'][split];pin(index['path'],index['sha256'])
        state = torch.load(args.parent,map_location='cpu',weights_only=False)
        identity = state['identity'];model_config = state['config']
        if (state['dataset_hash'] != sha(args.main_data/'processed/manifest.json')
                or identity.get('stats_sha256') != sha(args.stats)
                or identity.get('loss_stats_sha256') != sha(args.loss_stats)
                or identity.get('arm') != 'action' or not identity.get('mixed_data')
                or identity.get('source_manifest') != train.meta):
            raise ValueError('parent main dynamics/normalization identity mismatch')
        for path,expected in identity['vendor_sources'].items():
            vendor_path = VENDOR/path
            if sha(vendor_path)!=expected:
                raise ValueError('parent vendor code differs: '+path)
        for path,expected in identity['implementation_sources'].items():
            if sha(TASK/path)!=expected:
                raise ValueError('parent implementation differs: '+path)
        norm = json.loads(args.stats.read_text())
        loss_stats = validate_loss_statistics(json.loads(args.loss_stats.read_text()),
                                              state['dataset_hash'],MAIN_SOURCE_NAMES)
        scales = {k:v.cuda() for k,v in loss_stats.items()}
        rng = np.random.default_rng(config['seed'])
        draws = mixed_indices(train,config['updates']*config['main_batch'],config['seed'])
        cp_draws = rng.choice(cp_train.groups[0],config['updates']*config['aux_batch'])
        cp_panel = rng.choice(cp_val.groups[0],config['aux_validation_samples'])
        panel_path = args.parent.parent/'validation_balanced.npy'
        pin(panel_path)
        panel = np.load(panel_path).copy()
        if args.engineering:
            panel = np.concatenate([offset+source.groups[0][:2] for source,offset in zip(val.sources,val.offsets[:-1])])
            cp_panel=cp_panel[:4]
        if len(panel)!=192 and not args.engineering:
            raise ValueError('reuse exact192-window parent validation panel')
        for name,values in [('main_draw',draws),('transport_draw',cp_draws),('main_panel',panel),('transport_panel',cp_panel)]:
            np.save(out/(name+'.npy'),values);pin(out/(name+'.npy'))
        def pin_dataset(dataset,indices):
            if isinstance(dataset,MixedWindows):
                for source,offset,end in zip(dataset.sources,dataset.offsets[:-1],dataset.offsets[1:]):
                    selected=indices[(indices>=offset)&(indices<end)]-offset
                    pin_dataset(source,selected)
                return
            for index in np.unique(dataset.rows[indices,0]):
                folder=dataset.root/'processed/sequences'/dataset.sequences[int(index)]
                for name in ('meta.json','hand.npy','hand_valid.npy','poses.npy','pose_valid.npy',
                             'program.npy','near.npy','frame_ids.npy','centers.npy'):
                    pin(folder/name)
                if dataset.kind=='native':pin(folder/'timestamps.npy')
                for obj in json.loads((folder/'meta.json').read_text())['objects']:
                    path=dataset.root/'processed/canonical'/(obj+'.npz')
                    if str(path.absolute()) not in frozen:pin(path)
        for dataset,indices in [(train,draws),(val,panel),(cp_train,cp_draws),(cp_val,cp_panel)]:
            pin_dataset(dataset,indices)
        manifest.update(status='RUNNING',parent_step=state['step'],optimizer_reset=True,
                        sampler_shared=True,auxiliary_forward_inputs=list(HISTORY_INPUTS))
        write(out/'input_manifest.json',manifest)
        def seed(value):
            random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed(value)
        class HistoryView(torch.nn.Module):
            def __init__(self,model):super().__init__();self.model=model
            def forward(self,batch,arm):
                if arm!='history':raise ValueError('transport auxiliary must use history')
                return self.model(history_inputs(batch),'history')
        def evaluate(model):
            saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
            seed(config['seed']+10000)
            main_metrics={};values=[]
            for source,offset,end,desc in zip(val.sources,val.offsets[:-1],val.offsets[1:],val.meta['sources']):
                selected=panel[(panel>=offset)&(panel<end)]-offset
                metrics=base.evaluate(model,source,selected,'action',config['validation_microbatch'],True)
                main_metrics[desc['name']]=metrics
                values.append(metrics['model/anchor/cat0/h24/point_epe'])
            cp_metrics=base.evaluate(HistoryView(model),cp_val,cp_panel,'history',config['validation_microbatch'],True)
            random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2]);torch.cuda.set_rng_state(saved[3])
            return dict(main_macro_mm=sum(values)/len(values)*1000,
                        oakink2_mm=main_metrics['oakink2']['model/anchor/cat0/h24/point_epe']*1000,
                        transport_mm=cp_metrics['model/anchor/cat0/h24/point_epe']*1000,
                        main_metrics=main_metrics,transport_metrics=cp_metrics)
        initial_hashes=[];completed={};initial_metrics=[]
        for arm,weight in (('main-only',0.),('transport-auxiliary',config['auxiliary_weight'])):
            check();seed(config['seed'])
            model=model_from_config(norm,model_config).cuda();model.load_state_dict(state['model'],strict=True)
            h=hashlib.sha256()
            for name,tensor in sorted(model.state_dict().items()):h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
            initial_hashes.append(h.hexdigest())
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=model_config['weight_decay'])
            arm_out=out/arm;arm_out.mkdir();arm_state=dict(status='RUNNING',step=0,weight=weight,initial_model_hash=h.hexdigest())
            manifest['arms'][arm]=arm_state;write(out/'input_manifest.json',manifest)
            baseline=evaluate(model);initial_metrics.append(baseline);write(arm_out/'validation_initial.json',baseline)
            main_batches=iter(base.loader(train,draws,config['main_batch'],workers=2))
            cp_batches=iter(base.loader(cp_train,cp_draws,config['aux_batch'],workers=2))
            torch.cuda.reset_peak_memory_stats();step=0
            with (arm_out/'train.jsonl').open('x') as log:
                for step in range(1,config['updates']+1):
                    if step%10==1:check()
                    before=time.monotonic();fetch=time.monotonic()
                    batch=base.device_batch(next(main_batches));aux=base.device_batch(next(cp_batches))
                    input_seconds=time.monotonic()-fetch
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(batch,'action')
                    main_loss,_=physical_loss(model,prediction,batch,scales)
                    if not torch.isfinite(main_loss):raise FloatingPointError('nonfinite main loss')
                    main_loss.backward()
                    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(history_inputs(aux),'history')
                    auxiliary_loss,_=physical_loss(model,prediction,aux,scales)
                    if not torch.isfinite(auxiliary_loss):raise FloatingPointError('nonfinite auxiliary loss')
                    (weight*auxiliary_loss).backward()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),model_config['clip_grad'],error_if_nonfinite=True)
                    optimizer.step();torch.cuda.synchronize()
                    row=dict(step=step,main_loss=float(main_loss.detach()),auxiliary_loss=float(auxiliary_loss.detach()),
                             gradient_norm=float(grad),seconds=time.monotonic()-before,input_seconds=input_seconds,
                             peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                             peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                             main_batch=config['main_batch'],aux_batch=config['aux_batch'])
                    log.write(json.dumps(row)+'\n');log.flush()
                    arm_state.update(row);write(out/'progress.json',dict(status='RUNNING',arm=arm,**row))
                    if step==1 or step%20==0:print(json.dumps(dict(arm=arm,**row)),flush=True)
            check();final=evaluate(model);write(arm_out/'validation_final.json',final)
            part=arm_out/'final.pt.part'
            torch.save(dict(checkpoint_kind='pointworld.transport-auxiliary.v1',model=model.state_dict(),
                            optimizer=optimizer.state_dict(),step=step,config=config,arm=arm,
                            identity=manifest,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                            numpy_rng=np.random.get_state(),python_rng=random.getstate()),part)
            part.replace(arm_out/'final.pt')
            arm_state.update(status='COMPLETED',validation=final);completed[arm]=final
            write(out/'input_manifest.json',manifest)
            del main_batches,cp_batches,optimizer,model;torch.cuda.empty_cache()
        check(full=True)
        if len(set(initial_hashes))!=1:raise AssertionError('matched initial weights differ')
        if initial_metrics[0] != initial_metrics[1]:raise AssertionError('matched initial validation differs')
        manifest.update(status='COMPLETED',elapsed_seconds=time.time()-started)
        verdict='ENGINEERING_PASS' if args.engineering else classify_auxiliary(completed['main-only'],completed['transport-auxiliary'])
        result=dict(status='COMPLETED',verdict=verdict,engineering_only=args.engineering,
                    initial_weights_identical=True,updates_per_arm=config['updates'],arms=completed,
                    initial_validation_identical=True,
                    elapsed_seconds=time.time()-started,training_allowed=False,
                    limitation='incremental transport auxiliary Probe; not hand-action dynamics or robot policy utility')
        write(out/'result.json',result);write(out/'progress.json',dict(status='COMPLETED',verdict=verdict))
        print(json.dumps({k:v for k,v in result.items() if k!='arms'}),flush=True)
    except BaseException as error:
        if isinstance(error,TimeoutError) and 'model' in locals() and 'optimizer' in locals() and 'arm_out' in locals():
            stopped=arm_out/'stopped.pt'
            if not stopped.exists():
                torch.save(dict(checkpoint_kind='pointworld.transport-auxiliary.v1',model=model.state_dict(),
                                optimizer=optimizer.state_dict(),step=step,config=config,arm=arm,identity=manifest),stopped)
        manifest.update(status='TIMED_OUT' if isinstance(error,TimeoutError) else 'FAILED',error=repr(error),elapsed_seconds=time.time()-started)
        write(out/'progress.json',{k:v for k,v in manifest.items() if k not in ('sources','arms')})
        raise
    finally:
        write(out/'input_manifest.json',manifest)


if __name__=='__main__':main()
