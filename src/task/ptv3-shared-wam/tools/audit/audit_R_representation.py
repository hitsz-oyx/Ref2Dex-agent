"""Independent corpus/NumPy-controller replay of the three selected R models."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/audit')]
from audit_R_repair import arrays,ChunkRetargeter,independent_pd,ACTIVE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=5);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);torch.set_num_threads(4)
    import subprocess
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('idle audit GPU required')
    destination=a.run/'independent-audit.json'
    if destination.exists():raise FileExistsError(destination)
    manifest=json.loads((a.run/'manifest.json').read_text());result=json.loads((a.run/'result.json').read_text())
    if manifest['status']!='COMPLETED':raise ValueError('completed Probe required')
    with np.load(a.run/'panels.npz') as f:
        panels={name:arrays(Path(manifest['source']),split,f[key]) for name,split,key in [('TRAIN','train','TRAIN_indices'),('dev','val','dev_indices')]}
    checks={};maxima={}
    for arm in result['arms']:
        packet=torch.load(a.run/(arm+'_best.pt'),map_location='cpu',weights_only=False)
        curve=json.loads((a.run/(arm+'-curve.json')).read_text())
        selected=min(curve,key=lambda row:(row['panels']['dev']['xyz_p95_mm'],row['panels']['dev']['standardized_L1']))
        checks[arm+'_selection']=selected['step']==packet['step']==result['arms'][arm]['selected']['step']
        model=ChunkRetargeter(128).cuda().eval();model.load_state_dict(packet['model'])
        with np.load(a.run/(arm+'-selected-predictions.npz')) as saved:
            for name,panel in panels.items():
                state,tau,_,q,actual_pd,controller=panel
                with torch.no_grad():
                    pred=np.concatenate([model(torch.tensor(state[i:i+128],device='cuda'),torch.tensor(tau[i:i+128],device='cuda')).cpu().numpy() for i in range(0,len(state),128)])
                checks[arm+'/'+name+'/prediction']=bool(np.allclose(pred,saved[name],atol=2e-6,rtol=1e-5))
                target=pred[:,0].copy()
                if arm!='native':
                    target+=q[:,ACTIVE]
                    if arm=='geometry':
                        rotation=Rotation.from_euler('XYZ',q[:,3:6])*Rotation.from_rotvec(pred[:,0,3:6])
                        angles=rotation.as_euler('XYZ')
                        alternative=np.stack((angles[:,0]+np.pi,np.pi-angles[:,1],angles[:,2]+np.pi),-1)
                        candidates=np.stack((angles,alternative),1)
                        candidates+=2*np.pi*np.round((q[:,None,3:6]-candidates)/(2*np.pi))
                        target[:,3:6]=candidates[np.arange(len(q)),np.square(candidates-q[:,None,3:6]).sum(-1).argmin(-1)]
                    offset=np.asarray(controller['offset'])[ACTIVE];scale=np.asarray(controller['scale'])[ACTIVE]
                    target=(target-offset)/scale
                    target[:,:6]=(target[:,:6]*scale[:6]-q[:,:6])/scale[:6]
                    target[:,6:]=target[:,6:]*2-1
                command=np.zeros_like(q);command[:,ACTIVE]=target.clip(-1,1)
                difference=(independent_pd(command,q,controller)-actual_pd)[:,ACTIVE]
                xyz=np.linalg.norm(difference[:,:3],axis=-1)*1000
                computed=dict(xyz_mean_mm=float(xyz.mean()),xyz_p95_mm=float(np.percentile(xyz,95)),
                              rotation_p95_rad=float(np.percentile(np.abs(difference[:,3:6]).max(-1),95)),
                              finger_p95_rad=float(np.percentile(np.abs(difference[:,6:]).max(-1),95)),
                              clipping=float((np.abs(target)>1).any(-1).mean()))
                maxima[arm+'/'+name]={key:abs(value-selected['panels'][name][key]) for key,value in computed.items()}
        del model
    passed=all(checks.values()) and all(error<1e-3 for row in maxima.values() for error in row.values())
    report=dict(pass_=passed,checks=checks,maxima=maxima,device='cuda',physical_gpu=a.gpu,scope='independent source features and first PD metrics; not physics or all8 counterfactual execution')
    destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if not passed:raise SystemExit('independent metric mismatch')


if __name__=='__main__':main()
