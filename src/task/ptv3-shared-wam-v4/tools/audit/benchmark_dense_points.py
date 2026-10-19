"""Isolated dense-input timing prototype; production model/contracts stay frozen.

Uses real MANO/CAD fixtures and the existing PTv3/trunk/heads/objective. No KNN,
no data/model checkpoint writes, no scientific model-quality measurement.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

CODE_ROOT=Path(os.environ.get('REF2DEX_BENCH_CODE_ROOT',Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(CODE_ROOT / 'src/task/ptv3-shared-wam-v4/src'))
from ptv3_shared_wam_v4 import paths
from ptv3_shared_wam_v4.model import SharedWAMV4, mlp
from ptv3_shared_wam_v4.objectives import forecast_loss
from ptv3_shared_wam.model import clean
from ptv3_shared_wam_v2.geometry import JointGeometry
from oakink_wm.model import mean_groups
from oakink_wm.pointworld import fixed_workspace_grid


class DenseInputGeometry(JointGeometry):
    """Add surface points to PTv3, retaining sparse hand query readback."""
    def __init__(self, config):
        super().__init__(config)
        self.surface_hand_encoder = mlp(12, self.width)
        scales = [config['position_scale_m']]*3 + [config['velocity_scale_mps']]*3
        scales += [config['acceleration_scale_mps2']]*3 + [config['position_scale_m']]*3
        self.register_buffer('surface_scale', torch.tensor(scales))

    def forward(self, batch, hand_encoding, hand_valid):
        scene_valid = batch['point_valid'].bool() & (batch['features'][...,14] < .5)
        scene_xyz = clean(batch['xyz'], scene_valid)
        scene_features = self.scene_proj(clean(batch['features'], scene_valid)/self.feature_scale)
        query_xyz = clean(batch['start_hand'], batch['start_hand_valid']).flatten(1,2)
        dense_valid = batch['dense_hand_valid'].flatten(1)
        dense_xyz = clean(batch['dense_hand'], batch['dense_hand_valid']).flatten(1,2)
        dense_x = torch.cat((batch['dense_hand'],batch['dense_history']),-1)
        dense_features = self.surface_hand_encoder(clean(dense_x,batch['dense_hand_valid'])/self.surface_scale).flatten(1,2)
        xyz = torch.cat((scene_xyz, query_xyz, dense_xyz),1)
        valid = torch.cat((scene_valid,hand_valid,dense_valid),1)
        features = torch.cat((scene_features,hand_encoding,dense_features),1)
        count = scene_valid.shape[1]
        is_hand = torch.zeros_like(valid);is_hand[:,count:]=True
        ids = torch.arange(len(xyz),device=xyz.device)[:,None].expand_as(valid)
        with torch.autocast('cuda',enabled=False):
            grid = fixed_workspace_grid(xyz,valid,self.voxel_origin_m,self.voxel_size_m)
            unique,inverse = torch.unique(torch.cat((ids[valid][:,None],grid[valid]),-1),dim=0,return_inverse=True)
            hand_rows=is_hand[valid]
            sm=mean_groups(features.float()[valid][~hand_rows],inverse[~hand_rows],len(unique))
            hm=mean_groups(features.float()[valid][hand_rows],inverse[hand_rows],len(unique))
            encoded=self.backbone(dict(coord=mean_groups(xyz[valid],inverse,len(unique)),
                feat=self.voxel_fusion(torch.cat((sm,hm),-1)),grid_coord=unique[:,1:].int(),
                batch=unique[:,0],grid_size=self.voxel_size_m))
            packed=encoded.feat.new_zeros((*valid.shape,self.width));packed[valid]=encoded.feat[inverse]
        scene,mask=self._sample_tokens(torch.cat((packed[:,:count],scene_xyz),-1),scene_valid)
        return dict(tokens=scene[...,:self.width],xyz=scene[...,self.width:],mask=mask,
                    hand=packed[:,count:count+hand_encoding.shape[1]],voxel_count=len(unique))


class TimingWAM(SharedWAMV4):
    """Generalize only benchmark tensor cardinalities; retain all computation."""
    def __init__(self,stats,config,hand_points=11,dense_input=False):
        geometry=DenseInputGeometry(config) if dense_input else JointGeometry(config)
        super().__init__(stats,config,geometry=geometry)
        self.hand_points=hand_points
        if hand_points!=11:self.keypoint_id=nn.Embedding(hand_points,config['point_width'])

    def encode_hand(self,batch):
        valid=batch['start_hand_valid'].bool()
        x=torch.cat((clean(batch['start_hand'],valid),clean(batch['start_hand_history'],valid)),-1)
        identity=self.hand_id.weight[:x.shape[1],None]+self.keypoint_id.weight[None]
        hand=self.hand_encoder(x/self.hand_scale)+identity[None]
        return torch.where(valid[...,None],hand,0.).flatten(1,2),valid.flatten(1)

    def encode(self,batch):
        hand,hvalid=self.encode_hand(batch)
        scene=self.geometry(batch,hand,hvalid)
        scene_tokens=self.scene_project(scene['tokens'])+self.position(clean(scene['xyz'],scene['mask'])/self.position_scale)
        explicit=hand if self.explicit_hand else torch.zeros_like(hand)
        hand_tokens=self.hand_project(torch.cat((scene['hand'],explicit),-1))
        gv=batch['object_goal_valid'].bool()
        goal=torch.cat((clean(batch['start_object_points'],gv),clean(batch['goal_object_delta'],gv)),-1)/self.position_scale
        tokens=torch.cat((scene_tokens+self.token_type.weight[0],hand_tokens+self.token_type.weight[1],
                          self.goal(goal)+self.token_type.weight[2]),1)
        valid=torch.cat((scene['mask'],hvalid,gv),1)
        tokens=torch.where(valid[...,None],tokens,0.)
        if self.config.get('checkpoint_interaction',True) and self.training and torch.is_grad_enabled():
            for layer in self.interaction.layers:
                tokens=checkpoint(layer,tokens,src_key_padding_mask=~valid,use_reentrant=False,preserve_rng_state=False)
        else:tokens=self.interaction(tokens,src_key_padding_mask=~valid)
        return torch.where(valid[...,None],tokens,0.),valid,scene

    def forward(self,batch):
        shared,valid,geometry=self.encode(batch)
        offset=geometry['tokens'].shape[1]
        hands,points_per_hand=batch['start_hand'].shape[1:3]
        hand_tokens=shared[:,offset:offset+hands*points_per_hand]
        hz=self.hand_future(hand_tokens).reshape(len(shared),24,hands,points_per_hand,3)
        points=batch['scene_points'];ov=batch['object_valid'].bool()
        pv=ov[...,None].expand(*points.shape[:-1])
        query=self.scene_query(clean(points,pv)/self.position_scale).flatten(1,2)
        context=self.scene_context(shared)
        update,_=self.scene_readout(query,context,context,key_padding_mask=~valid,need_weights=False)
        latent=self.scene_norm(query+update).reshape(len(points),points.shape[1],points.shape[2],-1)
        sz=self.scene_future(latent.flatten(1,2)).reshape(len(points),24,*points.shape[1:3],3).transpose(1,2)
        ip=batch['interaction_points'];inn=batch['interaction_normals']
        qv=batch.get('interaction_query_valid',torch.ones_like(ip[...,0],dtype=torch.bool))
        q=self.interaction_query(torch.cat((clean(ip,qv),clean(inn,qv)),-1))
        oc=self._masked_mean(latent,pv,dim=2).unsqueeze(2).unsqueeze(3)
        hc=self._masked_mean(hand_tokens,batch['start_hand_valid'].flatten(1)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        gc_=self._masked_mean(shared,valid).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        relation=(q.unsqueeze(2)+self.interaction_object_project(oc)+self.interaction_hand_project(hc)
                  +self.interaction_context_project(gc_)+self.interaction_time.weight[None,None,:,None])
        logits=self.interaction_head(relation)
        interaction=torch.cat((logits[...,:1].sigmoid(),logits[...,1:].softmax(-1)),-1)
        cm=(qv[:,:,None,:]&ov[:,:,None,None]).expand(-1,-1,24,-1)
        hm=batch['start_hand_valid'][:,None,...,None];sm=pv[:,:,None,:,None]
        return dict(trajectory=torch.where(hm,self.normalizer.denormalize('hand',hz),0.),
            trajectory_normalized=torch.where(hm,hz,0.),
            scene=torch.where(sm,self.normalizer.denormalize('scene',sz),0.),
            scene_normalized=torch.where(sm,sz,0.),interaction=torch.where(cm[...,None],interaction,0.),
            interaction_normalized=torch.where(cm[...,None],interaction,0.),
            interaction_logits=torch.where(cm[...,None],logits,0.),interaction_prediction_valid=cm)


def make_batch(fixture,batch_size,mode,queries,hands,device):
    rows=[]
    for i in range(4):
        prefix=f'{i}__base__'
        row={k[len(prefix):]:v.copy() for k,v in fixture.items() if k.startswith(prefix)}
        if 'goal_object_delta' not in row:
            row['goal_object_delta']=row['goal_object_points']-row['start_object_points']
        if hands==1:
            row['start_hand_valid'][1]=False;row['trajectory_valid'][:,1]=False
        if mode!='baseline':
            row['xyz']=fixture[f'{i}__dense_xyz'];row['features']=fixture[f'{i}__dense_features']
            row['point_valid']=np.ones(2048,bool)
            dense=fixture[f'{i}__dense_hand'];history=fixture[f'{i}__dense_history']
            hv=np.broadcast_to(row['start_hand_valid'].all(-1)[:,None],(2,778)).copy()
            if mode=='geometry':
                row.update(dense_hand=dense,dense_history=history,dense_hand_valid=hv)
            else:
                row['start_hand']=dense[:hands];row['start_hand_history']=history[:hands]
                row['start_hand_valid']=hv[:hands]
                row['trajectory_target']=fixture[f'{i}__dense_trajectory'][:,:hands]
                row['trajectory_valid']=np.broadcast_to(hv[None,:hands],(24,hands,778)).copy()
        if queries==2048:
            old=row['interaction_target'];old_mask=row['interaction_valid']
            row['interaction_points']=fixture[f'{i}__dense_xyz'][None]
            row['interaction_normals']=fixture[f'{i}__dense_normals'][None]
            target=np.zeros((1,24,2048,7),np.float32);mask=np.zeros((1,24,2048),bool)
            ids=np.linspace(0,511,128,dtype=np.int64)
            target[:,:,ids]=old;mask[:,:,ids]=old_mask
            row['interaction_target']=target;row['interaction_valid']=mask
        rows.append(row)
    keys=['xyz','features','point_valid','start_hand','start_hand_history','start_hand_valid',
          'scene_points','object_valid','interaction_points','interaction_normals','interaction_target',
          'interaction_valid','start_object_points','goal_object_delta','object_goal_valid',
          'trajectory_target','trajectory_valid','scene_target','scene_valid']
    if mode=='geometry':keys+=['dense_hand','dense_history','dense_hand_valid']
    return {key:torch.from_numpy(np.stack([rows[i%4][key] for i in range(batch_size)])).to(device)
            for key in keys}


def summary(values):
    return dict(mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))


def inference(model,batch,warmup,steps):
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(warmup):model(batch)
        torch.cuda.synchronize();events=[];start=time.monotonic()
        for _ in range(steps):
            a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record()
            model(batch);b.record();events.append((a,b))
        torch.cuda.synchronize()
    return dict(cuda=summary([a.elapsed_time(b) for a,b in events]),
                wall_ms=(time.monotonic()-start)*1000/steps)


def training(model,batch,config,warmup,steps):
    model.train()
    optimizer=torch.optim.AdamW(model.parameter_groups(config['geometry_lr'],config['shared_lr']),
        weight_decay=config['weight_decay'],fused=True)
    events=[];losses=[]
    for i in range(warmup+steps):
        optimizer.zero_grad(set_to_none=True)
        ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        if i==warmup:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        ev[0].record()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=model(batch);loss,_=forecast_loss(prediction,batch,config,model.normalizer)
        ev[1].record();loss.backward();ev[2].record()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'])
        optimizer.step();ev[3].record()
        losses.append(loss.detach());
        if i>=warmup:events.append(ev)
        del prediction,loss
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    if not torch.isfinite(torch.stack(losses)).all() or not torch.isfinite(norm):
        raise ValueError('nonfinite timing updates')
    report=dict(wall_step_ms=elapsed*1000/steps,samples_per_second=len(batch['xyz'])*steps/elapsed,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        cuda={name:summary([ev[i].elapsed_time(ev[i+1]) for ev in events])
              for i,name in enumerate(['forward_loss','backward','clip_optimizer'])},
        last_loss=float(losses[-1]),optimizer_updates=warmup+steps)
    del optimizer,losses
    model.zero_grad(set_to_none=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',required=True)
    p.add_argument('--mode',choices=['baseline','geometry','full'],default='baseline')
    p.add_argument('--voxel',type=float,default=.01);p.add_argument('--queries',type=int,choices=[128,2048],default=128)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--hands',type=int,choices=[1,2],default=1)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--steps',type=int,default=8)
    p.add_argument('--check-equivalence',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(176)
    free,total=torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(.1,(free-2*2**30)/total))
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=dict(parent['config']);config['voxel_m']=args.voxel
    torch.backends.cuda.matmul.allow_tf32=config['tf32'];torch.backends.cudnn.allow_tf32=config['tf32']
    fixture=dict(np.load(args.fixture,allow_pickle=False))
    model=TimingWAM(parent['stats'],config,hand_points=778 if args.mode=='full' else 11,
                    dense_input=args.mode=='geometry')
    compatible={k:v for k,v in parent['model'].items() if k in model.state_dict()
                and v.shape==model.state_dict()[k].shape}
    model.load_state_dict(compatible,strict=False);model.cuda()
    report=dict(case=args.case,mode=args.mode,voxel_m=args.voxel,queries=args.queries,
        batch=args.batch,active_hands=args.hands,hand_vertices_per_active_side=778 if args.mode!='baseline' else 11,
        object_surface_points=512 if args.mode=='baseline' else 2048,
        shared_token_count=256+(778*args.hands if args.mode=='full' else 22)+32,
        checkpoint=str(args.checkpoint),parent_step=parent['step'],
        environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0)),
        scope='GPU-resident real geometry timing; no data-loader/DDP/validation/checkpoint cost')
    try:
        if args.check_equivalence:
            original=SharedWAMV4(parent['stats'],config).cuda().eval()
            original.load_state_dict(parent['model'],strict=True)
            small=make_batch(fixture,1,'baseline',128,2,'cuda');model.eval()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                reference=original(small);actual=model(small)
            errors={k:float((reference[k].float()-actual[k].float()).abs().max())
                    for k in reference if reference[k].dtype.is_floating_point}
            if max(errors.values())>1e-6:raise AssertionError(errors)
            report['baseline_equivalence_max_abs']=errors
            del original,small,reference,actual;gc.collect();torch.cuda.empty_cache()
        del parent,compatible;gc.collect()
        small=make_batch(fixture,1,args.mode,args.queries,args.hands,'cuda')
        report['inference_batch1']=inference(model,small,args.warmup,args.steps)
        del small
        batch=make_batch(fixture,args.batch,args.mode,args.queries,args.hands,'cuda')
        report['inference_training_batch']=inference(model,batch,args.warmup,args.steps)
        report['training']=training(model,batch,config,args.warmup,args.steps)
        report['status']='COMPLETED'
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='OOM',error=str(error),free_before_mib=free/2**20)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
