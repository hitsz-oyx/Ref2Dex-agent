"""Fixed measured-H comparison of warm start and two frozen PPO means; no physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/consequence-evaluator/src')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--intervention',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.resolve().parents:
        raise ValueError('fresh task-owned output required')
    started=time.monotonic()
    manifest=json.loads((args.evaluation/'manifest.json').read_text())
    if manifest['status']!='COMPLETED' or manifest['matmul_precision']!='highest':
        raise ValueError('completed strict-precision evaluation required')
    with np.load(args.evaluation/'plans.npz') as f:
        plans={key:f[key] for key in f.files}
    ids=np.flatnonzero(np.asarray(manifest['roles'])=='warm_start')
    if len(ids)!=4:raise ValueError('allfour warm-start measured histories required')
    before=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    util,memory=map(int,before.split(','))
    if util>10 or memory>512:raise ValueError('GPU not idle: '+before)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    from trajectory_policy.actor import load_actor
    from trajectory_policy.dense_decoder import DenseDecoder
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    warm=next(Path(p) for p in manifest['input_sha256'] if p.endswith('/best.pt'))
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    files=[args.evaluation/'manifest.json',args.evaluation/'plans.npz',warm,urdf,Path(__file__)]
    for folder in (args.control,args.intervention):
        m=json.loads((folder/'manifest.json').read_text())
        if m['status']!='COMPLETED' or sha(folder/'final.pt')!=m['final_checkpoint_sha256']:
            raise ValueError('frozen PPO checkpoint drift')
        actor_source=TASK/'src/trajectory_policy/actor.py'
        if sha(actor_source)!=m['input_sha256'][str(actor_source)]:
            raise ValueError('actor implementation differs from frozen run')
        files.extend([folder/'manifest.json',folder/'final.pt'])
    decoder=DenseDecoder(urdf,'cuda')
    models={name:load_actor(torch.load(path,map_location='cuda',weights_only=False),'cuda').eval()
        for name,path in (('warm',warm),('lambda95',args.control/'final.pt'),('lambda1',args.intervention/'final.pt'))}
    data={};stats={}
    with torch.no_grad():
        for name,model in models.items():
            means=[];hands=[];native=[]
            for j in range(len(plans['tick'])):
                h=torch.as_tensor(plans['history'][j,ids],device='cuda')
                c=model(h).cpu().numpy()
                d=decoder.decode(c,plans['q'][j,ids,0],plans['query_obj'][j,ids],1/30)
                means.append(c);hands.append(d['hand'].cpu().numpy());native.append(d['q'].cpu().numpy())
            data[name]=dict(c=np.stack(means),hand=np.stack(hands),q=np.stack(native))
            if name=='warm':
                if abs(data[name]['c']-plans['c'][:,ids]).max()>2e-5:
                    raise ValueError('fixed-H warm-start replay differs')
            else:
                delta=data[name]['q'][:,:,1:9]-data['warm']['q'][:,:,1:9]
                hand=data[name]['hand'][:,:,:8]-data['warm']['hand'][:,:,:8]
                rms=np.sqrt(np.mean(hand**2,axis=(2,3,4)))*1000
                xyz=np.sqrt(np.mean(delta[...,:3]**2,axis=(2,3)))*1000
                early=plans['tick']<=40
                stats[name]=dict(initial_hand_mean_change_mm=float(rms[0].max()),
                    initial_xyz_mean_change_mm=float(xyz[0].max()),precontact_hand_rms_change_max_mm=float(rms[early].max()),
                    precontact_xyz_rms_change_max_mm=float(xyz[early].max()),
                    precontact_finger_change_max_rad=float(abs(delta[early][:,:,:,[6,8,10,12,14,15]]).max()),
                    hand_mean_change_mm_by_query=rms.tolist(),xyz_mean_change_mm_by_query=xyz.tolist(),
                    xyz_noise_std_mm=float(model.log_std.detach().exp().reshape(24,12)[:,:3].mean()*10))
    args.output.mkdir(parents=True)
    files += list((TASK/'src/trajectory_policy').glob('*.py'))
    report=dict(status='UNCLEAR',stats=stats,ticks=plans['tick'].tolist(),elapsed_s=time.monotonic()-started,
        claim='Fixed failed-warm-start measured H isolates model mean movement; not a simulation, OOD success test or causal attribution')
    (args.output/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',physical_gpu=args.gpu,input_sha256={str(p.resolve()):sha(p) for p in files}),indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in s.items() if not k.endswith('_by_query')} for name,s in stats.items()},indent=2),flush=True)


if __name__=='__main__':
    def deadline(signum,frame):
        raise TimeoutError('fixed-state mean diagnostic exceeded30s')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(30)
    main()
