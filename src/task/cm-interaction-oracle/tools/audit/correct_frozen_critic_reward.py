#!/usr/bin/env python3
"""Restore source_e260's exact 2/10/5 shaping from immutable candidate physics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src'),str(ROOT/'src/task/cm-interaction-oracle/tools/run')]
from critic_ranking import bootstrapped_score,selection_summary,FrozenCritic
from run_rolling_gt_y import CHECKPOINT,sha
from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge
from src.task.CmResidual.dexplore_approach import ApproachConfig,sampled_surface_gap,potential_approach_reward
from src.task.CmResidual.dexplore_grasp_reward import held_lift_reward,contact_lift_progress_reward


def source_components(panel,rows,raw,dones,bridge):
    config=ApproachConfig();dev=bridge.device
    q=torch.cat((panel['history'][rows,-1,:18,None].transpose(1,2),panel['native_q'][rows,:8]),1).to(dev)
    physical=torch.cat((panel['before'][rows,None],panel['trajectory'][rows,:8]),1).to(dev)
    gaps=[]
    # Preserve source's bridge.current contract including its identity hand root.
    for step in range(9):
        geometry=bridge.current(q[:,step],physical[:,step,:13])
        gaps.append(sampled_surface_gap(geometry.hand_points,geometry.object_points,config))
    gap=torch.stack(gaps,1);phys=physical[:,1:]
    contact=(phys[:,:,48:63].reshape(-1,8,5,3).norm(dim=-1)>.1).any(-1)
    contact &= phys[:,:,63:66].norm(dim=-1)>.1
    done=dones.to(dev)
    before_z=physical[:,:-1,2];after_z=phys[:,:,2]
    approach=2*potential_approach_reward(gap[:,:-1].reshape(-1),gap[:,1:].reshape(-1),done.reshape(-1),gamma=.99,config=config).reshape(-1,8)
    rest=panel['rest_height'][rows,None].expand(-1,8).to(dev)
    held=10*held_lift_reward(after_z.reshape(-1),rest.reshape(-1),contact.reshape(-1),torch.ones_like(contact.reshape(-1))).reshape(-1,8)
    progress=5*contact_lift_progress_reward(before_z.reshape(-1),after_z.reshape(-1),contact.reshape(-1),torch.ones_like(contact.reshape(-1)),done.reshape(-1)).reshape(-1,8)
    native=raw.to(dev);total=((native+approach)+torch.zeros_like(native))+held;total=total+progress
    return total.cpu(),dict(gap=gap.cpu(),before_z=before_z.cpu(),after_z=after_z.cpu(),rest=rest.cpu(),contact=contact.cpu(),native=native.cpu(),approach=approach.cpu(),held=held.cpu(),progress=progress.cpu())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    hashes={}
    def read(path,expected=None):
        path=Path(path).resolve();h=sha(path)
        if expected is not None and h!=expected:raise ValueError('hash mismatch: '+str(path))
        value=torch.load(path,map_location='cpu',weights_only=False) if path.suffix in ('.pt','.pth') else json.loads(path.read_text())
        hashes[str(path)]=h;return value
    source_manifest=read(args.source_run/'manifest.json')
    if source_manifest['status']!='COMPLETED':raise ValueError('completed capture required')
    result=read(args.source_run/'result.json');audit=read(args.source_run/'replay.json')
    if audit['status']!='PASS':raise ValueError('checkpoint replay PASS required')
    config_path=ROOT/'outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/config.json'
    config=read(config_path)
    if [config[k] for k in ('approach_reward_coef','held_lift_reward_coef','lift_progress_reward_coef','grasp_link_reward_coef','min_grasp_links')]!=[2.,10.,5.,0.,0]:
        raise ValueError('source reward config differs')
    checkpoint=read(CHECKPOINT,'16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f')
    critic=FrozenCritic(checkpoint).to('cuda:0')
    assets=ROOT/'third_party/DExplore/dexplore/data/assets'
    bridge=DExploreCmv2GeometryBridge(hand_urdf=assets/'inspire_hand_new/inspire_hand_right.urdf',object_urdf=assets/'mjcf/airplane.urdf',device=torch.device('cuda:0'),seed=42)
    code=[Path(__file__),ROOT/'src/task/cm-interaction-oracle/src/critic_ranking.py']
    code += [ROOT/'src/task/CmResidual'/x for x in ('dexplore_approach_agent.py','dexplore_approach.py','dexplore_grasp_reward.py','dexplore_cm_geometry.py','surface_execution.py','v118_planner.py')]
    for path in code:hashes[str(path.resolve())]=sha(path)
    # Geometry samples derive from pinned visual assets already audited by ref13.
    sampling=read(ROOT/'outputs/cm-interaction-oracle/oracle-y-utility-sync-s263-s264/paired_actual_flow.pt')
    for path,h in sampling['asset_sha256'].items():
        if sha(Path(path))!=h:raise ValueError('geometry assets drift')
        hashes[str(Path(path).resolve())]=h
    source_npz=args.source_run/'scores.npz';hashes[str(source_npz.resolve())]=sha(source_npz)
    old=np.load(source_npz);rewards=[];values=[];components=[];max_value_error=0.
    with torch.no_grad():
        for seed in (263,264):
            for group in (0,1):
                gr=[];gv=[];gc=[]
                for k in range(7):
                    if time.monotonic()-start>100:raise TimeoutError('120second correction budget')
                    path=args.source_run/f's{seed}-g{group}-k{k}'/'critic.pt'
                    packet=read(path,result['input_sha256'][str(path.resolve())])
                    panel=read(packet['original_candidate_path'],packet['original_candidate_sha256'])
                    total,c=source_components(panel,packet['rows'],packet['raw_rewards8'],packet['dones8'],bridge)
                    predicted=critic(packet['observations8'].to('cuda:0')).cpu()
                    max_value_error=max(max_value_error,float((predicted-packet['live_values8']).abs().max()))
                    gr.append(total);gv.append(predicted);gc.append(c)
                rewards.append(torch.stack(gr,1));values.append(torch.stack(gv,1));components.append(gc)
    r,v=[torch.cat(x).numpy() for x in (rewards,values)]
    q,rp,vp=bootstrapped_score(r,v,np.zeros_like(r,dtype=bool));yscore=old['y'][...,7]+.25*old['y'][...,3]-old['y'][...,6]
    arms={a:selection_summary(s,old['z'],yscore) for a,s in [('Critic_Q8',q),('Value8_only',v),('Reward8_only',rp),('GT_Y',yscore)]}
    primary=arms['Critic_Q8'];passed=primary['successes']>=arms['GT_Y']['successes'] and primary['rescued']>=1 and primary['harmed']==0
    corrected=dict(status='PROMISING' if passed else 'UNPROMISING',anchors=32,groups=4,records=result['records'],candidate_names=result['candidate_names'],
        arms=arms,candidate_z_upper=result['candidate_z_upper'],source_reward=dict(native=True,approach=2,held_lift=10,lift_progress=5,grasp_link=0,seed=42),
        native_score_result_sha256=sha(args.source_run/'result.json'),V_only_unchanged=arms['Value8_only']==result['arms']['Value8_only'],
        scope='Repaired original PPO reward+V on identical one-shot candidate panels; no new simulation or mixed/rolling policy')
    np.savez_compressed(args.run_dir/'scores.npz',y=old['y'],z=old['z'],rewards8=r,value8=v,score=q,reward_component=rp,bootstrap_component=vp)
    torch.save(dict(components=components),args.run_dir/'components.pt')
    (args.run_dir/'result.json').write_text(json.dumps(corrected,indent=2,allow_nan=False)+'\n')
    if max_value_error>1e-5:raise ValueError('critic replay mismatch')
    for path,h in hashes.items():
        if sha(Path(path))!=h:raise ValueError('correction input drift')
    manifest=dict(status='COMPLETED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_run=str(args.source_run.resolve()),run_id=args.run_dir.name,input_sha256=hashes,elapsed_seconds=time.monotonic()-start,
        value_replay_error=max_value_error,device='cuda:0',physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        no_simulation_or_training=True,source_reward_reconstructed_from_original_formula=True)
    (args.run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status=corrected['status'],arms={a:m['successes'] for a,m in arms.items()},seconds=manifest['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
