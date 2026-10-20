"""Train Scheduled686 BC from a Current421 parent and measured tau cache."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("REF2DEX_RUN_ROOT", TASK.parents[2])).resolve()


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2) + "\n"); tmp.replace(path)


def load(manifest, split):
    import sys
    sys.path.insert(0, str(TASK / "src"))
    from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus
    corpus = TeacherRolloutCorpus(manifest, split)
    measured=[]; task=[]; action=[]; object_id=[]; points=[]; sequences=[]; active_masks=[]
    for i in range(len(corpus)):
        e=corpus.episode(i); keep=np.asarray(e["active"],bool)
        if not keep.any(): continue
        oid=len(points); points.append(e["object_points"])
        measured.append(e["measured"][keep]); task.append(e["task"][keep]); action.append(e["action"][keep])
        object_id.append(np.full(int(keep.sum()),oid,np.int64)); sequences.append(e["sequence"]); active_masks.append(keep)
    return dict(measured=np.concatenate(measured),task=np.concatenate(task),action=np.concatenate(action),
                object_id=np.concatenate(object_id),points=np.stack(points),sequences=sequences,active_masks=active_masks)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--tau-cache',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--gpu',type=int,default=6)
    p.add_argument('--steps',type=int,default=50000); p.add_argument('--batch',type=int,default=2048); p.add_argument('--seed',type=int,default=247); p.add_argument('--seconds',type=int,default=28800)
    a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if out.exists() and any(x.name not in {'train.log','pid'} for x in out.iterdir()): raise ValueError('fresh scheduled output required')
    if a.steps!=50000 or not 60<=a.seconds<=28800: raise ValueError('bounded scheduled recipe required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(ROOT/'tmp'/out.name),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    (ROOT/'tmp'/out.name).mkdir(parents=True,exist_ok=True)
    import sys; sys.path.insert(0,str(TASK/'src'))
    import torch
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v3.measured_execution import TAU_LAYOUT_SCHEMA, GOAL_CONTRACT
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    train=load(a.manifest.resolve(),'train'); dev=load(a.manifest.resolve(),'dev')
    layout=json.loads((a.tau_cache/'layout.json').read_text())
    if layout.get('schema') != TAU_LAYOUT_SCHEMA or layout.get('goal_contract') != GOAL_CONTRACT:
        raise ValueError('rebuild tau cache with the shared measured TaskController goal')
    completions = [json.loads(path.read_text()) for path in sorted(a.tau_cache.glob('completion-*.json'))]
    if (not completions or len(completions) != completions[0]['world']
            or {row['rank'] for row in completions} != set(range(len(completions)))
            or sum(row['records_done'] for row in completions) != len(layout['records'])
            or any(row['status'] != 'COMPLETED' or row.get('goal_contract') != GOAL_CONTRACT
                   or row.get('source_manifest') != str(a.manifest.resolve()) for row in completions)
            or len({row['wam_sha256'] for row in completions}) != 1):
        raise ValueError('complete matching teacher-goal tau shards required')
    def check(split,data):
        rows=[r for r in layout['records'] if r['split']==split]
        if [r['sequence'] for r in rows]!=data['sequences']: raise ValueError(split+' sequence order changed')
        total=sum(r['commands'] for r in rows)
        tau=np.load(a.tau_cache/f'{split}-tau.npy',mmap_mode='r'); valid=np.load(a.tau_cache/f'{split}-valid.npy',mmap_mode='r')
        if tau.shape!=(total,264) or valid.shape!=(total,): raise ValueError(split+' tau shape mismatch')
        compressed_tau=[]; cursor=0
        compressed_valid=[]
        for sequence, mask in zip(data['sequences'], data['active_masks']):
            row=next(r for r in rows if r['sequence']==sequence)
            segment=slice(row['lo'],row['hi'])
            if len(mask) != int(row['commands']): raise ValueError(split+' active mask length mismatch for '+sequence)
            compressed_tau.append(np.asarray(tau[segment])[mask])
            compressed_valid.append(np.asarray(valid[segment])[mask])
            cursor += int(mask.sum())
        if cursor != len(data['action']): raise ValueError(split+' compressed row count mismatch')
        return np.concatenate(compressed_tau),np.concatenate(compressed_valid)
    train_tau,train_valid=check('train',train); dev_tau,dev_valid=check('dev',dev)
    if not train_valid.any() or not dev_valid.any(): raise ValueError('scheduled admission has no valid rows')
    device=torch.device('cuda')
    def tensors(d,tau,valid):
        ids=np.flatnonzero(valid)
        return dict(measured=torch.as_tensor(d['measured'][ids],device=device),task=torch.as_tensor(d['task'][ids],device=device),
                    action=torch.as_tensor(d['action'][ids],device=device),object_id=torch.as_tensor(d['object_id'][ids],device=device),
                    points=torch.as_tensor(d['points'],device=device),tau=torch.as_tensor(np.asarray(tau[ids]),device=device).reshape(-1,8,11,3),
                    valid=torch.as_tensor(np.asarray(valid[ids]),device=device))
    tr=tensors(train,train_tau,train_valid); dv=tensors(dev,dev_tau,dev_valid)
    parent=torch.load(a.parent.resolve(),map_location='cpu',weights_only=False)
    if parent.get('schema')!='ref2dex.inspire-measured-current-bc.v1': raise ValueError('Current421 parent required')
    model=MeasuredInspirePolicy(with_tau=True).to(device)
    model.initialize_current(parent)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5); rng=np.random.default_rng(a.seed+1)
    commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest=dict(schema='ref2dex.inspire-measured-scheduled-bc.v2',status='RUNNING',experiment_id='P-20261020-inspire-scheduled686-bc',run_id=out.name,git_commit=commit,
        source_manifest=str(a.manifest.resolve()),tau_cache=str(a.tau_cache.resolve()),goal_contract=GOAL_CONTRACT,wam=completions[0]['wam'],parent=str(a.parent.resolve()),input_dim=686,action_dim=12,steps=a.steps,batch=a.batch,seed=a.seed,
        train_rows=int(len(tr['action'])),dev_rows=int(len(dv['action'])),tau_admission='workspace_valid and valid=true',warmstart='Current421 best; tau/valid first-layer columns zero')
    write(out/'manifest.json',manifest); started=time.monotonic(); best=float('inf'); curve=[]
    @torch.no_grad()
    def evaluate():
        model.eval(); pred=model(dv['measured'],dv['task'],dv['points'][dv['object_id']],tau=dv['tau'],tau_valid=dv['valid']); e=(pred-dv['action']).abs(); model.train(); return dict(command_mae=float(e.mean()),root_mae=float(e[:,:6].mean()),finger_mae=float(e[:,6:].mean()),shape_train=model.shape_report(tr['points']),shape_dev=model.shape_report(dv['points']))
    write(out/'initial-dev.json',evaluate())
    for step in range(1,a.steps+1):
        if time.monotonic()-started>a.seconds or (out/'STOP').exists(): raise TimeoutError('owned scheduled stop')
        ids=torch.as_tensor(rng.integers(0,len(tr['action']),size=a.batch),device=device,dtype=torch.long)
        pred=model(tr['measured'][ids],tr['task'][ids],tr['points'][tr['object_id'][ids]],tau=tr['tau'][ids],tau_valid=tr['valid'][ids]); loss=((pred-tr['action'][ids])/model.actor.y_scale).abs().mean()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite scheduled loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==1 or step%1000==0 or step==a.steps:
            metrics=evaluate(); elapsed=time.monotonic()-started; row=dict(step=step,loss=float(loss),dev=metrics,elapsed_s=elapsed,eta_s=elapsed/step*(a.steps-step),gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True).strip()); curve.append(row); write(out/'curve.json',curve); write(out/'progress.json',row)
            packet=dict(schema=manifest['schema'],step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),config=manifest,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state); torch.save(packet,out/'latest.pt.part');(out/'latest.pt.part').replace(out/'latest.pt')
            if metrics['command_mae']<best: best=metrics['command_mae']; torch.save(packet,out/'best.pt.part');(out/'best.pt.part').replace(out/'best.pt')
            print(json.dumps(row),flush=True)
    manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started,best_dev_command_mae=best);write(out/'manifest.json',manifest)

if __name__=='__main__': main()
