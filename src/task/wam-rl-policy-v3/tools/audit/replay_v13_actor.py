"""Batch-check saved R outputs against its frozen checkpoint on an idle allowed GPU."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path,required=True);p.add_argument('--gpu',type=int,choices=[4,5,6],default=6);p.add_argument('--allow-cpu-if-busy',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.worker.resolve();frozen=run/'runtime-code';m=json.loads((run/'manifest.json').read_text())
    gpu=subprocess.check_output(['nvidia-smi','-i','4,5,6','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    idle=[int(line.split(',')[0]) for line in gpu.splitlines() if int(line.split(',')[1])<=64];device='cuda';fallback=None
    if a.allow_cpu_if_busy:
        if idle:a.gpu=a.gpu if a.gpu in idle else idle[0]
        else:device='cpu';fallback='all allowed physical GPUs4/5/6 occupied; engineering actor regeneration only, zero simulation/training'
    elif a.gpu not in idle:raise RuntimeError('audit GPU occupied; queue until idle')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu) if device=='cuda' else '',REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_V6_V5_CODE_ROOT=str(frozen),REF2DEX_V5_BASE_CODE_ROOT=str(frozen),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');sys.path[:0]=[str(frozen/'src/task/wam-rl-policy-v3/src'),str(frozen/'src/task/trajectory-policy/src'),str(frozen/'src/task/ptv3-shared-wam-v6/src')]
    import torch
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    torch.set_num_threads(2);model=MeasuredInspirePolicy(with_tau=True).to(device).eval();model.load_state_dict(torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)['model']);model.requires_grad_(False);records=[];started=time.monotonic();counterfactual_max=0.
    for row in m['runs']:
        with np.load(run/row['id']/'trajectory.npz') as z:obs=z['observation'][:-1].copy();expected=z['adapter_raw_action'].copy();tau=z['tau'][:-1].copy();valid=z['tau_valid'][:-1].copy()
        if len(obs)!=row['controls']:raise ValueError('physical controls/observations mismatch')
        tau_error=float(np.abs(obs[:,421:685]-tau.reshape(len(tau),264)).max());mask_match=bool(np.array_equal(obs[:,-1].astype(bool),valid));error=0.
        with torch.inference_mode():
            for lo in range(0,len(obs),256):
                x=torch.tensor(obs[lo:lo+256],dtype=torch.float32,device=device);prediction=model.actor(x);error=max(error,float(np.abs(prediction.cpu().numpy()-expected[lo:lo+256]).max()))
                # Same valid mask and non-tau fields; zero only 8x11x3 displacement.
                cf=x.clone();cf[:,421:685]=0;counterfactual_max=max(counterfactual_max,float(torch.abs(model.actor(cf)-prediction).max().item()))
        records.append(dict(id=row['id'],controls=len(obs),R_max_error=error,tau_observation_error=tau_error,valid_mask_match=mask_match,match=bool(error<2e-5 and tau_error==0 and mask_match)))
    result=dict(status='PASS' if records and all(r['match'] for r in records) and counterfactual_max>1e-6 else 'FAIL',n=len(records),controls=sum(r['controls'] for r in records),maximum_R_error=max(r['R_max_error'] for r in records),zero_tau_counterfactual_R_max_difference=counterfactual_max,gpu=a.gpu if device=='cuda' else None,device=device,fallback_reason=fallback,gpu_before=gpu,python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,checkpoint=m['base_checkpoint'],learning_updates=0,seconds=time.monotonic()-started,records=records,scope='Frozen R actor regeneration from saved observations; shape embedding and WAM predictions separately traced, not regenerated here. Counterfactual keeps valid and all non-tau fields. GPU batched GEMM tolerance2e-5 versus original single-state calls.');a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}));return int(result['status']!='PASS')
if __name__=='__main__':raise SystemExit(main())
