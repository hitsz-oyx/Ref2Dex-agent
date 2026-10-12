"""Explicit current semantic hand readout before scene-token subsampling."""
import torch

from .model import GoalPointWAM


def gather_hand_tokens(scene, batch):
    packed=scene['point_features']
    valid=batch['point_valid'].bool() & (batch['scene_object']<0) & (batch['features'][...,14]>.5)
    rows,points=valid.nonzero(as_tuple=True)
    hand=batch['features'][rows,points,15].round().long()
    key=(batch['features'][rows,points,16]*10).round().long()
    if ((hand<0)|(hand>=2)|(key<0)|(key>=11)).any():
        raise ValueError('current hand semantic metadata outside2x11')
    slots=rows*22+hand*11+key
    count=torch.zeros(len(packed)*22,dtype=torch.long,device=packed.device)
    count.scatter_add_(0,slots,torch.ones_like(slots))
    if (count>1).any():raise ValueError('duplicate current hand semantic point')
    mask=count.reshape(-1,2,11).bool()
    if not torch.equal(mask,batch['start_hand_valid'].bool()):
        raise ValueError('packed observed hand differs from raw current-hand mask')
    tokens=packed.new_zeros((len(packed),2,11,packed.shape[-1]))
    tokens[rows,hand,key]=packed[rows,points]
    return tokens,mask


class HandAnchoredWAM(GoalPointWAM):
    """Keep all historical parameter keys and raw-hand query residual."""
    def actor_from_scene(self, scene, batch):
        goal,mask=self.goal(batch['start_object_points'],batch['goal_object_points'],batch['object_goal_valid'])
        hand,_=gather_hand_tokens(scene,batch)
        raw=self.observed_hand_condition(batch)
        if raw is not None:hand=hand+raw
        return self.trajectory(scene['scene_tokens'],scene['scene_mask'],goal,mask,hand)

    def forward(self,batch):
        scene=self.scene(batch,return_points=True)
        trajectory=self.actor_from_scene(scene,batch)
        # Teacher-forced world interface stays goal-blind and unchanged.
        action,mask=self.action(batch['action'].reshape(-1,24,2,11,9),
                                batch['action_valid'].reshape(-1,24,2,11))
        world=self.world(scene['scene_tokens'],scene['scene_mask'],scene['object_tokens'],
                         scene['object_mask'],action,mask)
        return dict(trajectory=trajectory,translation=world['translation'],rotation=world['rotation'])
