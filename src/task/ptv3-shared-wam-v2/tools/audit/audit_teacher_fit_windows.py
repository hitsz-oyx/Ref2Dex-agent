"""GPU evaluation of final student weights on all cached windows and fixed panels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[5]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('assigned GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    started=time.monotonic();reports=[];prefix=[]
    for arm in ['full_relative','current_relative']:
        packets=[torch.load(a.run/(arm+'-'+s)/'step12000.pt',map_location='cpu',weights_only=False) for s in ['constant','cosine_tail']]
        assert all(torch.equal(packets[0]['model'][k],packets[1]['model'][k]) for k in packets[0]['model'])
        prefix.append(dict(arm=arm,step=12000,weights_exactly_equal=True))
        del packets
        for schedule in ['constant','cosine_tail']:
            folder=a.run/(arm+'-'+schedule);packet=torch.load(folder/'step24000.pt',map_location='cpu',weights_only=False)
            layers=[]
            for i in [0,2,4,6,8]:
                w=packet['model'][str(i)+'.weight'];layers.append(torch.nn.Linear(w.shape[1],w.shape[0]))
                if i!=8:layers.append(torch.nn.ReLU())
            model=torch.nn.Sequential(*layers).cuda().eval();model.load_state_dict(packet['model'],strict=True)
            xm,xs,ym,ys=[packet[k].cuda() for k in ['xmean','xscale','ymean','yscale']]
            report=dict(arm=arm,schedule=schedule,step=24000,splits={})
            for split in ['train','val']:
                x=np.load(a.cache/(split+('-current.npy' if arm=='current_relative' else '-full.npy')),mmap_mode='r')
                q=np.load(a.cache/(split+'-q.npy'),mmap_mode='r');pd=np.load(a.cache/(split+'-pd.npy'),mmap_mode='r');tick=np.load(a.cache/(split+'-tick.npy'),mmap_mode='r')
                errors=[]
                with torch.inference_mode():
                    for start in range(0,len(x),2048):
                        batch=torch.tensor(np.array(x[start:start+2048]),device='cuda')
                        if arm=='current_relative':batch=torch.nn.functional.pad(batch,(0,721))
                        batch=((batch-xm)/xs).clamp(-5,5)
                        if arm=='current_relative':batch[:,721:]=0
                        prediction=model(batch)*ys+ym
                        delta=(prediction[:,:3]+torch.tensor(np.array(q[start:start+2048,:3]),device='cuda')-torch.tensor(np.array(pd[start:start+2048,:3]),device='cuda'))
                        errors.append(torch.linalg.vector_norm(delta,dim=1).cpu().numpy()*1000)
                e=np.concatenate(errors);ids=np.load(a.cache/(split+'-panel.npy'))
                def stats(values):return dict(count=len(values),mean_mm=float(values.mean()),median_mm=float(np.median(values)),p95_mm=float(np.percentile(values,95)))
                report['splits'][split]=dict(all_windows=stats(e),startup=stats(e[tick<16]),later=stats(e[tick>=16]),fixed_panel=stats(e[ids]))
                expected=json.loads((folder/'result.json').read_text())['final']['panels'][split]
                assert abs(float(e[ids].mean())-expected['xyz_mean_mm'])<.01
                destination=a.run/('errors-'+arm+'-'+schedule+'-'+split+'.npy')
                if destination.exists():raise FileExistsError(destination)
                np.save(destination,e)
            reports.append(report);print(json.dumps(report),flush=True);del model
    destination=a.run/'all-window-audit.json'
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(dict(status='PASS',prefix_audit=prefix,reports=reports,elapsed_s=time.monotonic()-started,
        note='all-window weighting differs from primary sequence-uniform half-startup panel; no thresholds or memberships changed'),indent=2)+'\n')


if __name__=='__main__':main()
