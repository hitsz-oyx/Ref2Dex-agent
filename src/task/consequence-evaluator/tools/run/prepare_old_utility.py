"""Prepare frozen old-U32 labels from other trajectories; reserve ref13 panels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]; ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from consequence_evaluator.data import sha, object_effect, interaction_future
from consequence_evaluator.contracts import HAND_LINKS, is_within
from consequence_evaluator.old_utility import SCHEMA, teacher


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--held-panels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();start=time.monotonic();rng=np.random.default_rng(289)
    if out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator'):
        raise ValueError('fresh task-owned pack required')
    hashes={};records=[];audits=[]
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('input drift: '+str(path))
        hashes[str(path)]=digest
    for path in (Path(__file__),TASK/'src/consequence_evaluator/old_utility.py',
                 ROOT/'src/task/cm-interaction-oracle/src/rolling_y_audit.py',
                 ROOT/'src/task/cm-interaction-oracle/src/oracle_y_utility.py'):
        pin(path)
    def sample(history,plan,poses,hands,contact,rest,tick,**meta):
        if len(poses[tick:tick+33])!=33 or not contact.dtype==np.bool_:
            raise ValueError('complete32step force-pair outcome required')
        pose=poses[tick:tick+25];hand=hands[tick:tick+25]
        y,u=teacher(poses[tick,2,3],contact[tick],poses[tick+1:tick+33,2,3],contact[tick+1:tick+33],rest)
        C=np.linalg.inv(pose[0]);hh=hands[tick-3:tick+1];hf=hand[1:]
        records.append(dict(history=history,action=plan,
            future=np.concatenate((object_effect(pose)[:,:3].reshape(24,12),interaction_future(pose,hand).reshape(24,33)),-1),
            pw_object_history=C@poses[tick-3:tick+1],
            pw_hand_history=np.einsum('ij,tkj->tki',C[:3,:3],hh)+C[:3,3],
            pw_hand_future=np.einsum('ij,tkj->tki',C[:3,:3],hf)+C[:3,3],
            y=y,label=u,tick=tick,**meta))
    for folder in a.source:
        folder=folder.resolve();m=json.loads((folder/'manifest.json').read_text());pin(folder/'manifest.json')
        if (m['schema']!='ref2dex.consequence-value.episodes.v1' or m['status']!='COMPLETED'
                or m['fps']!=30 or m['contact_semantics']!='native_hand_and_object_net_force_proxy'):
            raise ValueError('completed raw native contact source required')
        for path,digest in m['sources'].items():pin(path,digest)
        for r in m['episodes']:
            if time.monotonic()-start>120:raise TimeoutError('pack preparation120s')
            for key,digest in (('path','sha256'),('diagnostics','diagnostics_sha256')):pin(folder/r[key],r[digest])
            with np.load(folder/r['path'],allow_pickle=False) as f:d={k:f[k] for k in f.files}
            with np.load(folder/r['diagnostics'],allow_pickle=False) as f:diag={k:f[k] for k in f.files}
            steps=len(d['action']);ticks=set(range(8,steps-32+1,8))
            if 3<=r['perturbation_tick']<=steps-32:ticks.add(r['perturbation_tick'])
            eligible=np.array([t for t in sorted(ticks) if d['plan_known'][t]],np.int64)
            active=eligible[np.abs(d['residual_plan'][eligible]).max((1,2))>1e-7]
            if len(active)>16:raise ValueError('active-window cap exceeds preregistered sample design')
            rest=np.setdiff1d(eligible,active);selected=np.sort(np.r_[active,rng.choice(rest,16-len(active),replace=False)])
            if not diag['contact_valid'][1:].all() or not np.isclose(diag['initial_height'],diag['reference_object_pose'][0,2,3],atol=1e-5):
                raise ValueError('native full-reference rest/contact identity mismatch')
            for tick in selected:
                sample(d['history'][tick],d['residual_plan'][tick],d['object_pose'],d['hand_keypoints'],diag['contact'],
                    float(diag['reference_object_pose'][0,2,3]),int(tick),episode=r['episode'],split=r['split'],
                    split_group=r['split_group'],panel=-1,candidate=-1)
            audits.append(dict(episode=r['episode'],split=r['split'],windows=len(selected),active=len(active)))
    root=a.held_panels.resolve();meta=json.loads((root/'initial-scores.json').read_text());pin(root/'initial-scores.json')
    rows=meta['rows'];query=int(meta['query']);first=None;first_geometry=None
    for k in range(7):
        folder=root/('initial-k'+str(k))
        for name in ('panel.pt','trace.pt'):pin(folder/name,meta['input_sha256'][str(folder/name)])
        panel=torch.load(folder/'panel.pt',map_location='cpu',weights_only=False)
        trace=torch.load(folder/'trace.pt',map_location='cpu',weights_only=False);g=trace['progress_geometry']
        if first is None:first=panel;first_geometry=g
        for key in ('initial_fingerprint','simulation_contract','model_fingerprint','rms_fingerprint'):
            if panel[key]!=first[key]:raise ValueError('held fork identity mismatch')
        for key in ('before','history','actor_obs','hand_root'):
            if not torch.equal(panel[key][rows],first[key][rows]):raise ValueError('same-H identity mismatch')
        for key in ('object_pose','hand_keypoints'):
            if not torch.equal(g[key][:query+1,rows],first_geometry[key][:query+1,rows]):raise ValueError('held geometry prefix mismatch')
        if (panel['candidate']!=k or panel['post_window']!=32 or panel['clipped_steps'][rows].any()
                or not panel['valid_steps'][rows].all() or not np.isclose(panel['control_dt'],1/30)
                or not (panel['full_world_prefix_errors']<=torch.tensor([1e-4]*5+[1e-5,0.])).all()
                or tuple(g['hand_links'])!=HAND_LINKS):raise ValueError('held candidate contract failure')
        for j,row in enumerate(rows):
            if int(panel['triggers'][row])!=query or trace['done'][query:query+32,row].any():raise ValueError('incomplete held query')
            contact=np.r_[trace['physical'][0,row,71].numpy()>.5,
                          trace['after_physical'][:,row,71].numpy()>.5]
            poses=g['object_pose'][:,row].numpy();hands=g['hand_keypoints'][:,row].numpy()
            # physical[t] is before action[t]. The saved geometry is states0..end.
            _,u=teacher(panel['before'][row,2].numpy(),panel['before'][row,71].numpy()>.5,
                        panel['height'][row].numpy(),panel['pair'][row].numpy(),panel['rest_height'][row].numpy())
            plan=np.zeros((24,18),np.float32);plan[:8]=panel['delta'][k].numpy()
            sample(panel['actor_obs'][row].numpy(),plan,poses,hands,contact,float(panel['rest_height'][row]),query,
                episode='ref13_env'+str(row),split='panel',split_group='ref13_recovery_seed263_group0',panel=j,candidate=k)
            if not np.isclose(records[-1]['label'],u,atol=1e-7) or not np.isclose(u,meta['old_scores'][j][k],atol=1e-7):
                raise ValueError('frozen old teacher mismatch')
    keys=records[0].keys();arrays={key:np.stack([r[key] for r in records]) for key in keys}
    counts={split:int((arrays['split']==split).sum()) for split in ('train','val','test','panel')}
    for group in np.unique(arrays['split_group']):
        if len(np.unique(arrays['split'][arrays['split_group']==group]))!=1:raise ValueError('group split leakage')
    for key in ('history','action','future','pw_object_history','pw_hand_history','pw_hand_future','label'):
        if not np.isfinite(arrays[key]).all():raise ValueError('nonfinite prepared input')
    if any(sha(path)!=digest for path,digest in hashes.items()):raise ValueError('preparation drift')
    out.mkdir(parents=True);np.savez_compressed(out/'windows.npz',**arrays)
    manifest=dict(schema=SCHEMA,status='COMPLETED',label='frozen short_y32: Y7+.25Y3-Y6',horizon=24,label_horizon=32,
        source_kind='official generator for supervision; held self-trained actor panels',counts=counts,
        source_episodes=audits,panel_rows=rows,panel_query=query,training_seed=261,val_seed=262,
        model_input_whitelist=['history','action','future'],PW_semantics='observed hand oracle; object prediction only',
        input_sha256=hashes,windows_sha256=sha(out/'windows.npz'),elapsed_s=time.monotonic()-start,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(counts=counts,elapsed_s=manifest['elapsed_s']),indent=2))


if __name__=='__main__':main()
