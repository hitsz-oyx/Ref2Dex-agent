"""Per-node geometry and native motor consequences, with a known H10 law head."""
import math
import torch
from torch import nn
from .native_pd_selector import physical_inputs, native_pd_targets

NODE_DOFS = ((0,1,2,3,4,5),(6,7),(8,9),(10,11),(12,13),(14,15,16,17))
KEY_NODES = (0,3,6,9,12,15)
BINARY = (43,44,47,48,49,50)
FEATURES = ('history','physical','node_state','node_action','law')


def relative_geometry(observation):
    """Use current link origins and velocities; cancel the hand heading frame."""
    shape = observation.shape[:-1]
    x,z = observation[...,649:652],observation[...,652:655]
    rot = torch.stack((x,torch.cross(z,x,dim=-1),z),-1)
    pos = observation[...,406:454].reshape(*shape,16,3)[...,list(KEY_NODES),:]
    vel = observation[...,156:204].reshape(*shape,16,3)[...,list(KEY_NODES),:]
    pos = pos-observation[...,646:649].unsqueeze(-2)
    vel = vel-observation[...,655:658].unsqueeze(-2)
    return (torch.einsum('...ij,...kj->...ki',rot.transpose(-1,-2),pos),
            torch.einsum('...ij,...kj->...ki',rot.transpose(-1,-2),vel))


def node_channels(value):
    out = value.new_zeros(*value.shape[:-1],6,6)
    for node,channels in enumerate(NODE_DOFS):
        out[...,node,:len(channels)] = value[...,list(channels)]
    return out


def node_inputs(state,observation,targets,weights):
    pos,vel = relative_geometry(observation)
    identity = torch.eye(6,device=state.device).expand(*state.shape[:-1],6,6)
    node_state = torch.cat((pos,vel,node_channels(state[...,:18]),
                            node_channels(state[...,18:36]),identity),-1)
    node_action = torch.cat((node_channels(targets-state[...,:18]),weights),-1)
    return node_state,node_action


def transitions(record):
    if record['schema'] != 'ref2dex.contact_geometry_source.v1' or record['future_done'].any():
        raise ValueError('complete actual geometry records required')
    n = len(record['state'])
    pre = torch.cat((record['state'][:,None],record['future_state'][:,:-1]),1)
    post = record['future_state']
    pre_bits = torch.cat((record['history'][:,-1:,49:51],record['future_contact'][:,:-1].float()),1)
    histories = [record['history']]
    for step in range(1,10):
        row = torch.cat((pre[:,step],pre_bits[:,step],record['actual_action'][:,step-1]),-1)
        histories.append(torch.cat((histories[-1][:,1:],row[:,None]),1))
    history = torch.stack(histories,1)
    if not torch.equal(history[:,:,-1,:49],pre):
        raise ValueError('all prehistory/state alignment')
    hand = torch.cat((record['initial_hand_force'][:,None],record['future_hand_force'][:,:-1]),1)
    obj = torch.cat((record['initial_object_force'][:,None],record['future_object_force'][:,:-1]),1)
    clearance = torch.cat((record['initial_clearance'][:,None],record['future_clearance'][:,:-1]),1)
    mass = record['mass_kg'][:,None].expand(-1,10).reshape(-1)
    rest = record['rest_z'][:,None].expand(-1,10).reshape(-1)
    physical,_ = physical_inputs(history.flatten(0,1),hand.flatten(0,1),obj.flatten(0,1),mass,
                                 record['gravity_magnitude'],clearance.reshape(-1),rest)
    chosen = record['candidate_weights'][torch.arange(n),record['assignment']]
    weights = chosen[:,None].expand(-1,10,-1,-1)
    node_state,node_action = node_inputs(pre,record['native_observation'],record['actual_pd_targets'],weights)
    law = (record['assignment'] != 0).float()[:,None,None].expand(-1,10,-1)
    before_pos,before_vel = relative_geometry(record['native_observation'])
    after_pos,after_vel = relative_geometry(record['future_native_observation'])
    node_target = torch.cat(((after_pos-before_pos)/.005,(after_vel-before_vel)/.1),-1).flatten(-2)
    next_target = torch.cat(((post[...,43:46]-pre[...,43:46])/.1,
                             (post[...,36:39]-pre[...,36:39])/.005,
                             (record['future_clearance']-clearance)[...,None]/.002,
                             record['future_contact'].float()),-1)
    height = (record['state'][:,38]-record['rest_z']).clamp_min(0)
    last_height = (post[:,-3:,38].amin(-1)-record['rest_z']).clamp_min(0)
    final_hand = record['future_contact'][:,-3:,0].all(-1)
    final_obj = record['future_contact'][:,-3:,1].all(-1)
    end_clear = (record['future_clearance'][:,-3:]>=.002).all(-1)
    support = final_hand & final_obj & end_clear
    clear = record['future_clearance']>=.002
    was_clear = torch.cat(((record['initial_clearance']>=.002)[:,None],clear[:,:-1]),1).cummax(-1).values
    loss_event = (was_clear & ~clear).any(-1)
    macro = torch.stack(((post[:,-3:,38].amin(-1)-record['state'][:,38])/.01,
                         (record['future_clearance'][:,-3:].amin(-1)-record['initial_clearance'])/.002,
                         final_hand.float(),final_obj.float(),loss_event.float(),
                         (support & (last_height>=.03)).float(),last_height*support/.01),-1)
    target = torch.cat((node_target,next_target,macro[:,None].expand(-1,10,-1)), -1)
    # State-only kinematic prior. No actual future command or observation enters it.
    dt = record['control_dt_seconds']
    q = pre[...,39:43]
    x,y,z,w = q.unbind(-1)
    rot = torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                       2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                       2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(n,10,3,3)
    omega = torch.einsum('ntij,ntj->nti',rot.transpose(-1,-2),pre[...,46:49])
    relative_cv = (before_vel-torch.cross(omega[:,:,None].expand_as(before_pos),before_pos,dim=-1))*dt
    node_prior = torch.cat((relative_cv/.005,torch.zeros_like(before_vel)),-1).flatten(-2)
    logits = (pre_bits*2-1)*math.log(9)
    next_prior = torch.cat((torch.zeros_like(pre[...,43:46]),pre[...,43:46]*dt/.005,
                            pre[...,45,None]*dt/.002,logits),-1)
    dz = torch.minimum(pre[...,45]*dt*8,pre[...,45]*dt*10)
    current_height = (pre[...,38]-record['rest_z'][:,None]).clamp_min(0)
    cv_clear = clearance+dz>=.002
    cv_support = pre_bits.bool().all(-1)&cv_clear
    cv_height = (current_height+dz).clamp_min(0)*cv_support
    cv_loss = (clearance>=.002)&(clearance+torch.minimum(pre[...,45]*dt,pre[...,45]*dt*10)<.002)
    macro_prior = torch.cat((dz[...,None]/.01,dz[...,None]/.002,logits,
                             ((cv_loss.float()*2-1)*math.log(9))[...,None],
                             (((cv_support&(cv_height>=.03)).float()*2-1)*math.log(9))[...,None],
                             cv_height[...,None]/.01),-1)
    prior = torch.cat((node_prior,next_prior,macro_prior),-1)
    return dict(history=history.flatten(0,1),physical=physical,node_state=node_state.flatten(0,1),
                node_action=node_action.flatten(0,1),law=law.flatten(0,1),target=target.flatten(0,1),
                prior=prior.flatten(0,1),first=(torch.arange(10)[None].expand(n,-1)==0).reshape(-1),
                early=(~record['outcome']['initially_clear'])[:,None].expand(-1,10).reshape(-1))


class GeometryConsequenceModel(nn.Module):
    def __init__(self,physical_dim,mode='cm'):
        super().__init__()
        if mode not in ('cm','state_only','shuffled','direct_score'):
            raise ValueError('unknown mode')
        self.mode = mode
        self.history = nn.GRU(69,64,batch_first=True)
        self.physical = nn.Sequential(nn.Linear(physical_dim,64),nn.SiLU(),nn.LayerNorm(64))
        self.nodes = nn.Sequential(nn.Linear(36,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.LayerNorm(64))
        self.trunk = nn.Sequential(nn.Linear(193,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
        self.node_head = nn.Sequential(nn.Linear(192,128),nn.SiLU(),nn.Linear(128,6))
        self.global_head = nn.Linear(128,16)

    def forward(self,history,physical,node_state,node_action,law,prior):
        if self.mode == 'state_only':
            node_action,law = torch.zeros_like(node_action),torch.zeros_like(law)
        _,hidden = self.history(history)
        nodes = self.nodes(torch.cat((node_state,node_action),-1))
        global_state = self.trunk(torch.cat((hidden[-1],self.physical(physical),nodes.mean(-2),law),-1))
        local = self.node_head(torch.cat((nodes,global_state[:,None].expand(-1,6,-1)),-1)).flatten(-2)
        global_output = self.global_head(global_state)
        if self.mode == 'direct_score':
            correction = torch.zeros_like(prior)
            correction[:,51] = global_output[:,-1]
            return prior+correction
        return prior+torch.cat((local,global_output),-1)


def loss(pred,target,first,mode):
    f = torch.nn.functional
    if mode == 'direct_score':
        if not first.any():
            raise ValueError('direct-score batch needs initial programme states')
        return f.smooth_l1_loss(pred[first,51],target[first,51])
    localp,localt = pred[:,:36].reshape(-1,6,6),target[:,:36].reshape(-1,6,6)
    terms = [f.smooth_l1_loss(localp[...,:3],localt[...,:3]),
             f.smooth_l1_loss(localp[...,3:],localt[...,3:]),
             f.smooth_l1_loss(pred[:,36:39],target[:,36:39]),
             f.smooth_l1_loss(pred[:,39:42],target[:,39:42]),
             f.smooth_l1_loss(pred[:,42],target[:,42]),
             f.binary_cross_entropy_with_logits(pred[:,43:45],target[:,43:45])]
    if not first.any():
        raise ValueError('physics batch needs initial programme states')
    terms += [f.smooth_l1_loss(pred[first,45:47],target[first,45:47]),
              f.binary_cross_entropy_with_logits(pred[first,47:51],target[first,47:51]),
              f.smooth_l1_loss(pred[first,51],target[first,51])]
    return torch.stack(terms).mean()


def normalize(data,norm):
    return {k:((data[k]-norm[k+'_mean'])/norm[k+'_std']).clamp(-8,8) for k in FEATURES}


def candidate_features(record,data):
    """Generate pre-only programme features for all saved, executable proposals."""
    n = len(record['state'])
    state = record['state'][:,None].expand(-1,8,-1)
    observation = record['native_observation'][:,0,None].expand(-1,8,-1)
    targets = native_pd_targets(record['candidate_actions'],state[...,:18],record['pd_offset'],record['pd_scale'])
    nodes,actions = node_inputs(state,observation,targets,record['candidate_weights'])
    law = torch.tensor([0,1,1,1,1,1,1,1],device=state.device,dtype=state.dtype)[None,:,None].expand(n,-1,-1)
    out = {k:data[k][::10,None].expand(-1,8,*data[k].shape[1:]).flatten(0,1)
           for k in ('history','physical','prior')}
    out.update(node_state=nodes.flatten(0,1),node_action=actions.flatten(0,1),law=law.flatten(0,1))
    return out
