"""Bounded one-GPU WAM-v2 + ShadowHand native control finetuning."""
import argparse,json,os,sys,time,subprocess,shutil
from pathlib import Path
TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=Path)
    p.add_argument('--gpu',type=int,default=6);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--seed',type=int,default=210)
    p.add_argument('--max-seconds',type=int,default=2400);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    from shadow_finetune import paths
    if a.output.exists() or paths.ROOT/'outputs/shadow-finetune' not in a.output.resolve().parents:raise ValueError('fresh Task output')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    import numpy as np,torch
    from shadow_finetune.data import ShadowWindows,collate
    from shadow_finetune.model import ShadowPolicy,loss
    from ptv3_shared_wam.loading import to_device
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    out=a.output.resolve();out.mkdir(parents=True);scratch=paths.ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(scratch),TORCH_EXTENSIONS_DIR=str(paths.ROOT/'tmp/shared-wam-v2-extensions'))
    parent=torch.load(a.parent or paths.PARENT,map_location='cpu',weights_only=False)
    if parent.get('schema')!='ref2dex.ptv3-shared-wam.v2' or parent.get('stage')!='pretrain':raise ValueError('v2 pretrain parent')
    config=parent['config'];stats=parent['stats'];train=ShadowWindows(a.data,'train');dev=ShadowWindows(a.data,'dev')
    model=ShadowPolicy(stats,config);model.forecaster.load_state_dict(parent['model'],strict=True)
    with torch.no_grad():
        for k,v in train.normalization().items():getattr(model.actor,k).copy_(torch.as_tensor(v))
    model.cuda();optimizer=torch.optim.AdamW(model.groups(),weight_decay=.01,fused=True)
    rng=np.random.default_rng(a.seed);validation=np.random.default_rng(212).choice(len(dev),min(64,len(dev)),replace=False)
    manifest=dict(schema=paths.SCHEMA,status='RUNNING',engineering_only=a.smoke,data=str(a.data.resolve()),
        parent=str((a.parent or paths.PARENT).resolve()),parent_step=parent['step'],config=config,stats=stats,
        parameters=sum(p.numel() for p in model.parameters()),seed=a.seed,updates=4 if a.smoke else a.steps,
        batch_size=a.batch_size,gpu=a.gpu,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=paths.ROOT,text=True).strip(),
        train_windows=len(train),dev_windows=len(dev),control_hz=60,actor_state='full current native300',
        goal='decision-known task goal position, current orientation',action='canonical pre-transform 24D commands, eight60Hz controls; execute first')
    def write(name,payload):
        tmp=out/(name+'.part');tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');tmp.replace(out/name)
    write('manifest.json',manifest);np.save(out/'validation.npy',validation)
    curves=[];best=float('inf');started=time.monotonic()
    try:
        for step in range(1,manifest['updates']+1):
            batch=to_device(collate([train[i] for i in train.draw(rng,a.batch_size)]),'cuda')
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(batch);total,terms=loss(pred,batch,config,model)
            if not torch.isfinite(total):raise FloatingPointError('nonfinite loss')
            total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            optimizer.step()
            if step==1 or step%20==0 or step==manifest['updates']:
                torch.cuda.synchronize();elapsed=time.monotonic()-started
                row=dict(step=step,loss=float(total),terms={k:float(v) for k,v in terms.items()},elapsed_s=elapsed,
                    eta_s=elapsed/step*(manifest['updates']-step),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                write('progress.json',row)
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            evaluate=step%250==0 or step==manifest['updates']
            if evaluate:
                model.eval();metrics={};count=0
                with torch.no_grad():
                    for offset in range(0,len(validation),16):
                        ids=validation[offset:offset+16];b=to_device(collate([dev[int(i)] for i in ids]),'cuda')
                        pred=model(b);val,terms=loss(pred,b,config,model)
                        report=dict(total=float(val),**{k:float(v) for k,v in terms.items()},
                            command_first_mae=float((pred['action'][:,0]-b['action_target'][:,0]).abs().mean()),
                            command_root_mae=float((pred['action'][:,0,:6]-b['action_target'][:,0,:6]).abs().mean()),
                            command_finger_mae=float((pred['action'][:,0,6:]-b['action_target'][:,0,6:]).abs().mean()))
                        for k,v in report.items():metrics[k]=metrics.get(k,0.)+v*len(ids)
                        count+=len(ids)
                report=dict(step=step,count=count,**{k:v/count for k,v in metrics.items()});curves.append(report);write('curve.json',curves)
                if report['action']<best:
                    best=report['action'];save=True
                else:save=False
                # Preserve each250-step stage in this small<=2000-update Probe.
                packet=dict(manifest,step=step,model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),rng=rng.bit_generator.state)
                target=out/('step%d.pt'%step);tmp=target.with_suffix('.part');torch.save(packet,tmp);tmp.replace(target)
                for alias in (['latest.pt','best.pt'] if save else ['latest.pt']):
                    link=out/(alias+'.part');link.symlink_to(target.name);link.replace(out/alias)
                if sum(p.stat().st_size for p in out.glob('step*.pt'))>12*10**9:raise RuntimeError('12GB checkpoint budget')
            if time.monotonic()-started>a.max_seconds:raise TimeoutError('bounded train deadline')
            if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('20GiB floor')
        manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started);write('manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error));write('manifest.json',manifest);raise

if __name__=='__main__':main()
