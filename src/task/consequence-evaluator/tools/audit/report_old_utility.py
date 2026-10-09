"""Replay frozen same-H predictions and audit real one-shot Z90 paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/cm-interaction-oracle/src')]
from consequence_evaluator.data import sha
from consequence_evaluator.old_utility import OldUtility,panel_metrics
from rolling_control import execution_z,TOLERANCE
from oracle_y_utility import stable_grasp_z,paired_bootstrap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--pw',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    run=a.run.resolve();output=run/'control-result.json';root=a.replay.resolve()
    if output.exists():raise FileExistsError(output)
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();result=json.loads((run/'result.json').read_text());hashes=dict(result['input_sha256'])
    if any(sha(path)!=digest for path,digest in hashes.items()):raise ValueError('frozen evaluator input drift')
    load=lambda path:torch.load(path,map_location='cpu',weights_only=False)
    with np.load(a.data/'windows.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    with np.load(a.pw/'future.npz',allow_pickle=False) as f:pw=f['future']
    with np.load(run/'panel-predictions.npz') as f:pred={k:f[k] for k in f.files}
    for path in (run/'result.json',run/'panel-predictions.npz',Path(__file__).resolve()):hashes[str(path)]=sha(path)
    flat=pred['panel_rows'].ravel();baseline=load(root/'reanchor-r2/panel.pt');reference=load(root/'reanchor-r2/trace.pt')
    for path in (root/'reanchor-r2/panel.pt',root/'reanchor-r2/trace.pt'):hashes[str(path)]=sha(path)
    data_meta=json.loads((a.data/'manifest.json').read_text());rows=torch.tensor(data_meta['panel_rows']);origin=data_meta['panel_query']
    z0=stable_grasp_z(baseline['height'][rows],baseline['pair'][rows],baseline['rest_height'][rows])[0].numpy().astype(int)
    counts={'baseline':int(z0.sum())};outcomes={'baseline':z0};prediction_errors={};source_fit={};checks={};metrics_replay={}
    for arm in ('C0','C1','C2'):
        ck=run/(arm+'.pt');hashes[str(ck)]=sha(ck)
        if hashes[str(ck)]!=result['checkpoint_sha256'][arm]:raise ValueError('weights changed')
        saved=load(ck);model=OldUtility(**saved['architecture']).cuda().eval();model.load_state_dict(saved['model'])
        inputs={}
        for key in ('history','action','future'):
            value=pw if key=='future' and arm=='C2' else d[key]
            mean,std=saved['statistics'][key];inputs[key]=((torch.from_numpy(value).float()-mean)/std).cuda()
        def predict(ids):
            values=[]
            with torch.inference_mode():
                for offset in range(0,len(ids),252):
                    r=ids[offset:offset+252]
                    values.append(model(inputs['history'][r],inputs['action'][r],inputs['future'][r],arm!='C0').cpu().numpy())
            return np.concatenate(values)
        q=predict(flat).reshape(25,7);error=float(np.max(np.abs(q-pred[arm])));prediction_errors[arm]=error
        if error>1e-6:raise ValueError('saved prediction replay differs')
        metric=panel_metrics(pred['target'],q);metrics_replay[arm]=metric
        if metric['choices']!=result['metrics'][arm]['choices']:raise ValueError('selector changed')
        tr=np.flatnonzero(d['split']=='train');fit=predict(tr)
        source_fit[arm]=dict(rmse=float(np.sqrt(np.mean((fit-d['label'][tr])**2))),mae=float(np.abs(fit-d['label'][tr]).mean()))
        folder=root/('evaluator-'+arm.lower()+'-one-shot');status_path=root/(folder.name+'-status.json')
        status=json.loads(status_path.read_text());hashes[str(status_path)]=sha(status_path)
        if status['status']!='COMPLETED':raise ValueError('actual path incomplete')
        for name in ('panel.pt','trace.pt'):hashes[str(folder/name)]=sha(folder/name)
        panel=load(folder/'panel.pt');trace=load(folder/'trace.pt')
        for key in ('initial_fingerprint','simulation_contract','model_fingerprint','rms_fingerprint'):
            if panel[key]!=baseline[key]:raise ValueError('actual world/actor provenance mismatch')
        if not (panel['full_world_prefix_errors']<=TOLERANCE).all():raise ValueError('actual prefix contract failed')
        if panel['clipped_steps'][rows].any():raise ValueError('clipped requested candidate')
        choices=json.loads((run/('choices-'+arm+'.json')).read_text())['choices']
        hashes[str(run/('choices-'+arm+'.json'))]=sha(run/('choices-'+arm+'.json'))
        if not torch.equal(panel['rolling_choices'],torch.tensor(choices)):raise ValueError('wrong actual plan')
        if not np.array_equal(panel['actor_obs'][rows].numpy(),d['history'][pred['panel_rows'][:,0]]):
            raise ValueError('execution starts at different model H')
        for key in ('object_pose','hand_keypoints'):
            if not torch.equal(trace['progress_geometry'][key][:origin+1,rows],reference['progress_geometry'][key][:origin+1,rows]):
                raise ValueError('actual current geometry prefix differs')
        outcomes[arm]=execution_z(trace,rows,origin,baseline['rest_height'][rows])[0].numpy().astype(int)
        counts[arm]=int(outcomes[arm].sum());checks[arm]=dict(prefix_pass=True,model_H_exact=True,geometry_prefix_exact=True,
            requested_plan_exact=True,clipped_steps=0,full90_nonterminal=True,elapsed_s=status['elapsed_s'])
    comparisons={}
    for left,right in (('C0','baseline'),('C1','baseline'),('C2','baseline'),('C1','C0'),('C2','C1')):
        x,y=outcomes[left],outcomes[right]
        comparisons[left+'_vs_'+right]=dict(rescued=int(((x==1)&(y==0)).sum()),harmed=int(((x==0)&(y==1)).sum()),
            gain=paired_bootstrap(x-y,seed=265))
    if any(sha(path)!=digest for path,digest in hashes.items()):raise ValueError('audit drift')
    report=dict(status='UNCLEAR',ranking_screen='PROMISING' if result['gt_information_gate'] else 'UNCLEAR',
        scope='frozen selector actual one-shot Z90,25exposed s3 anchors; not rolling/full task/RL',counts=counts,
        comparisons=comparisons,checks=checks,outcomes={k:v.tolist() for k,v in outcomes.items()},
        saved_prediction_max_error=prediction_errors,source_train_fit=source_fit,metrics_replay=metrics_replay,
        C2_oracle_observed_hand=True,deployable_planner=False,full_task_gate1=False,
        input_sha256=hashes,elapsed_s=time.monotonic()-start)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('counts','comparisons','checks','saved_prediction_max_error','source_train_fit','elapsed_s')},indent=2))


if __name__=='__main__':main()
