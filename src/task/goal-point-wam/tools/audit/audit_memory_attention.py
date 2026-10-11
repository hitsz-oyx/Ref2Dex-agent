#!/usr/bin/env python3
"""Replay the 2026-10-11 frozen six-train-window attention diagnostic.

Run from the repository root with GPU visibility set by the caller. Inputs and
output are fixed to this experiment; this is neither fitting nor a held-out
metric. Conditioning attention mass does not demonstrate information use.
"""
import sys,json
from pathlib import Path
import torch
sys.path.insert(0,'src/task/goal-point-wam/tools/run')
from train_three_domain import GoalPointWAM,move_batch
from oakink_wm.multisource import MAIN_SOURCE_NAMES
torch.set_num_threads(2)
stats=json.load(open('outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json'))
b=move_batch(torch.load('outputs/goal-point-wam/fixed-train-learnability-20261011-r1/fixed_train_batch.pt',map_location='cpu',weights_only=False),torch.device('cuda'))
report={'diagnosis_only':True,'split':'train','models':[]}
for path in ['outputs/goal-point-wam/ref1-alignment-20261011-r1/best.pt','outputs/goal-point-wam/three-domain-lr1e3-20261011-r1/best.pt']:
 s=torch.load(path,map_location='cpu',weights_only=False);m=GoalPointWAM(stats,s['config']).cuda().eval()
 with torch.no_grad():
  scene=m.scene(b);goal,gv=m.goal(b['start_hand'],b['goal_hand'],torch.cat((b['start_hand_valid'],b['goal_hand_valid']),1));action,av=m.action(b['action'].reshape(-1,24,2,11,9),b['action_valid'].reshape(-1,24,2,11))
  norms={k:float(v.norm(dim=-1)[mask].mean()) for k,v,mask in [('scene',scene['scene_tokens'],scene['scene_mask']),('object',scene['object_tokens'],scene['object_mask']),('goal',goal,gv),('action',action,av)]}
  captures=[];handles=[]
  for branch in ['trajectory','world']:
   for idx,layer in enumerate(getattr(m,branch).decoder.layers):
    def hook(mod,args,kwargs,branch=branch,idx=idx):captures.append((branch,idx,mod,[x.detach() for x in args[:3]],kwargs))
    handles.append(layer.multihead_attn.register_forward_pre_hook(hook,with_kwargs=True))
  m.trajectory(scene['scene_tokens'],scene['scene_mask'],goal,gv)
  m.world(scene['scene_tokens'],scene['scene_mask'],scene['object_tokens'],scene['object_mask'],action,av)
  for h in handles:h.remove()
  attn=[]
  for branch,idx,module,args,kw in captures:
   _,w=module(*args,key_padding_mask=kw.get('key_padding_mask'),need_weights=True,average_attn_weights=True)
   mass=w[...,256:].sum(-1).mean(-1)
   attn.append(dict(branch=branch,layer=idx,conditioning_attention_mass=float(mass.mean()),source_mass={name:float(mass[b['source_id']==sid].mean()) for sid,name in enumerate(MAIN_SOURCE_NAMES)}))
  row=dict(checkpoint=path,norms=norms,attention=attn);report['models'].append(row);print(json.dumps(row),flush=True)
 del m,s,scene,goal,action,captures,handles
 torch.cuda.empty_cache()
Path('outputs/goal-point-wam/fixed-train-learnability-20261011-r2/memory_attention.json').write_text(json.dumps(report,indent=2)+'\n')
