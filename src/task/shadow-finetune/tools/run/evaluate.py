"""Matched teacher/student native simulator replay using the pinned, tested collector."""
import argparse,importlib.util,json,os,random,subprocess,sys,time
from pathlib import Path
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]


def longest_true(mask):
    best=run=0
    for x in mask:
        run=run+1 if x else 0;best=max(best,run)
    return best


def worker(a):
    # Importing the pinned collector executes no torch imports; worker imports IsaacGym first.
    source=ROOT/'src/task/video-data-preparation/tools/run/replay_unidexgrasp.py'
    text=source.read_text()
    old='    records = []\n    begin = time.monotonic()'
    new='''    # Preserve initialization RNG across teacher/student module construction.
    from shadow_finetune.execution import Controller
    _rng=(torch.get_rng_state(),torch.cuda.get_rng_state(),np.random.get_state(),random.getstate())
    shadow_controller=Controller(args.checkpoint,env_objects,actor,args.cache)
    torch.set_rng_state(_rng[0]);torch.cuda.set_rng_state(_rng[1]);np.random.set_state(_rng[2]);random.setstate(_rng[3])
    records = []
    begin = time.monotonic()'''
    if text.count(old)!=1:raise ValueError('pinned initialization hook changed')
    text=text.replace(old,new)
    old='                    action = actor(task.obs_buf.clamp(-5,5)).clamp(-1,1)'
    if text.count(old)!=1:raise ValueError('pinned action hook changed')
    text=text.replace(old,'                    action = shadow_controller.act(snapshot(),task.obs_buf,tick)')
    marker="    result = dict(status='COMPLETE',episodes=records"
    if text.count(marker)!=1:raise ValueError('pinned completion hook changed')
    text=text.replace(marker,"    write_json(args.output/'controller-audit.json',dict(student=shadow_controller.model is not None,invalid_envs=sorted(shadow_controller.invalid)))\n"+marker)
    dest=a.output/'runtime_replay.py';dest.write_text(text)
    spec=importlib.util.spec_from_file_location('shadow_runtime_replay',dest);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.insert(0,str(TASK/'src'))
    a.vendor=ROOT/'third_party/UniDexGrasp2';a.subset=ROOT/'outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1'
    a.assets=None;a.scale=.08;a.episodes=1;a.gpu_seconds=300
    # The patched worker's first dependency import loads IsaacGym before Controller/torch.
    if a.role=="teacher":a.checkpoint=None
    module.worker(a)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--object-manifest',type=Path,required=True);p.add_argument('--gpu',type=int,default=6)
    p.add_argument('--envs',type=int,default=16);p.add_argument('--seed',type=int,default=213)
    p.add_argument('--role',choices=('teacher','student'),default='student');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:worker(a);return
    import numpy as np
    if a.output.exists() or ROOT/'outputs/shadow-finetune' not in a.output.resolve().parents:raise ValueError('fresh evaluation output')
    a.output.mkdir(parents=True);reports={};initial={}
    base=[sys.executable,str(Path(__file__).resolve()),'--cache',str(a.cache.resolve()),'--object-manifest',str(a.object_manifest.resolve()),
          '--gpu',str(a.gpu),'--envs',str(a.envs),'--seed',str(a.seed),'--worker']
    for role in ('teacher','student'):
        out=a.output/role;out.mkdir();(out/'tmp').mkdir()
        cmd=base+['--checkpoint',str(a.checkpoint.resolve()),'--role',role,'--output',str(out.resolve())]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
            TMPDIR=str(out.resolve()/'tmp'),TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/shadow-isaac-extensions'),MAX_JOBS='2')
        env['LD_LIBRARY_PATH']=str(Path(sys.executable).resolve().parents[1]/'lib')+':'+env.get('LD_LIBRARY_PATH','')
        with (out/'worker.log').open('x') as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if result.returncode:raise RuntimeError(role+' native worker failed; see '+str(out/'worker.log'))
        with np.load(out/'episode-00.npz',allow_pickle=False) as z:
            initial[role]={k:z[k][0].copy() for k in ('q','object','hand')}
            distance=np.linalg.norm(z['object'][...,:3]-z['goal'],axis=-1);lift=z['object'][...,2].max(0)-z['object'][0,:,2]
            held=np.array([longest_true(distance[:,i]<=.05) for i in range(a.envs)])
            reports[role]=dict(episodes=a.envs,official_success=int((z['success'][-1]>0).sum()),
                final_within5cm=int((distance[-1]<=.05).sum()),held60_and_lift10cm=int(((held>=60)&(lift>=.1)).sum()),
                longest_goal_hold_frames=held.tolist(),max_lift_m=lift.tolist(),final_goal_distance_m=distance[-1].tolist())
    initial_error={k:float(np.max(np.abs(initial['teacher'][k]-initial['student'][k]))) for k in initial['teacher']}
    if max(initial_error.values())>1e-6:raise ValueError('teacher/student initial states differ: '+str(initial_error))
    report=dict(status='COMPLETED',seed=a.seed,roles=reports,initial_state_max_error=initial_error,
        metric='official sticky5cm flag, terminal5cm, and observed60 consecutive60Hz states within5cm with>=10cm lift',
        scope='small single-seed Probe; no stable-contact certification or data-size causal claim')
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
