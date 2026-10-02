"""Current-input H10 joint support, conditional height and physical consequences.

Experimental prototype. The three event bits are force-presence proxies and
whole-mesh clearance, not identified hand/object contact pairs.
"""
import torch
from torch import nn
from .contact_geometry_consequence import node_channels, node_inputs, relative_geometry
from .native_pd_selector import physical_inputs, native_pd_targets

FEATURES = ('history', 'physical', 'native', 'node_context', 'node_action', 'law')
MODES = ('cm', 'state_only', 'shuffled', 'direct_score')
EVENT_BITS = ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))


def current_features(history, native, hand_force, object_force, mass, gravity,
                     clearance, rest, bank, cup_action, candidate_action, weights,
                     law, offset, scale):
    """No future state, force, expert bank or label argument is accepted."""
    state = history[:,-1,:49]
    q = state[:,:18]
    physical, _ = physical_inputs(history, hand_force, object_force, mass, gravity, clearance, rest)
    cup_pd = native_pd_targets(cup_action, q, offset, scale)
    candidate_pd = native_pd_targets(candidate_action, q, offset, scale)
    bank_pd = native_pd_targets(bank, q[:,None], offset, scale)
    ref = torch.zeros_like(weights); ref[:,:,1] = 1
    nodes, _ = node_inputs(state, native, cup_pd, ref)
    bank_difference = node_channels(bank_pd-cup_pd[:,None]).transpose(1,2).flatten(-2)
    context = torch.cat((nodes, node_channels(cup_pd-q), bank_difference), -1)
    action = torch.cat((node_channels(candidate_pd-cup_pd), weights-ref), -1)
    return dict(history=history, physical=physical, native=native,
                node_context=context, node_action=action, law=law.reshape(-1,1).float())


def record_features(record, arm=None):
    n = len(record['state'])
    ids = torch.arange(n,device=record['state'].device)
    chosen = record['assignment'] if arm is None else torch.full_like(record['assignment'], arm)
    return current_features(record['history'], record['native_observation'][:,0],
        record['initial_hand_force'], record['initial_object_force'], record['mass_kg'],
        record['gravity_magnitude'], record['initial_clearance'], record['rest_z'],
        record['expert_bank'][:,0], record['candidate_actions'][:,1],
        record['candidate_actions'][ids,chosen], record['candidate_weights'][ids,chosen],
        chosen != 0, record['pd_offset'], record['pd_scale'])


def factual_labels(record):
    if record['schema'] not in ('ref2dex.contact_geometry_source.v1','ref2dex.optimized_contact_source.v1'):
        raise ValueError('audited executable H10 source required')
    if record['future_done'].any() or record['future_state'].shape[1:] != (10,49):
        raise ValueError('complete actual H10 windows required')
    hand = record['future_contact'][:,-3:,0].all(-1)
    obj = record['future_contact'][:,-3:,1].all(-1)
    clear = (record['future_clearance'][:,-3:] >= .002).all(-1)
    event = hand.long()*4+obj.long()*2+clear.long()
    support = event == 7
    height = (record['future_state'][:,-3:,38].amin(-1)-record['rest_z']).clamp_min(0)/.01
    clearance = record['future_clearance'] >= .002
    past = torch.cat(((record['initial_clearance']>=.002)[:,None],clearance[:,:-1]),-1).cummax(-1).values
    loss = (past & ~clearance).any(-1)
    before_pos, before_vel = relative_geometry(record['native_observation'][:,0])
    after_pos, after_vel = relative_geometry(record['future_native_observation'][:,0])
    local = torch.cat(((after_pos-before_pos)/.005,(after_vel-before_vel)/.1),-1).flatten(-2)
    post = record['future_state'][:,0]
    global_target = torch.cat(((post[:,43:46]-record['state'][:,43:46])/.1,
        (post[:,36:39]-record['state'][:,36:39])/.005,
        (record['future_clearance'][:,-3:].amin(-1)-record['initial_clearance'])[:,None]/.002),-1)
    return dict(event=event,support=support,height=height,supported_height=height*support,
        conditional_lift=height>=3.,lift=support&(height>=3.),loss=loss,
        physical=torch.cat((local,global_target),-1))


def normalization(data, fit):
    norm = {}
    for key in FEATURES:
        x = data[key][fit]
        dims = (0,1) if key in ('history','node_context','node_action') else 0
        norm[key+'_mean'] = x.mean(dims)
        norm[key+'_std'] = x.std(dims,unbiased=False).clamp_min(.001)
    return norm


def normalize(data,norm):
    return {k:((data[k]-norm[k+'_mean'])/norm[k+'_std']).clamp(-8,8) for k in FEATURES}


class StructuredContactConsequence(nn.Module):
    def __init__(self, physical_dim, mode='cm'):
        super().__init__()
        if mode not in MODES:
            raise ValueError('unknown consequence mode')
        self.mode = mode
        self.history = nn.GRU(69,64,batch_first=True)
        self.physical = nn.Sequential(nn.Linear(physical_dim,64),nn.SiLU(),nn.LayerNorm(64))
        self.native = nn.Sequential(nn.Linear(1442,64),nn.SiLU(),nn.LayerNorm(64))
        self.nodes = nn.Sequential(nn.Linear(78,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.LayerNorm(64))
        self.trunk = nn.Sequential(nn.Linear(257,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
        self.node_head = nn.Sequential(nn.Linear(192,128),nn.SiLU(),nn.Linear(128,6))
        self.global_head = nn.Linear(128,19)

    def forward(self,history,physical,native,node_context,node_action,law):
        if self.mode == 'state_only':
            node_action,law = torch.zeros_like(node_action),torch.zeros_like(law)
        _,hidden = self.history(history)
        nodes = self.nodes(torch.cat((node_context,node_action),-1))
        context = self.trunk(torch.cat((hidden[-1],self.physical(physical),
            self.native(native),nodes.mean(-2),law),-1))
        local = self.node_head(torch.cat((nodes,context[:,None].expand(-1,6,-1)),-1)).flatten(-2)
        output = self.global_head(context)
        event_logits = output[:,:8]
        events = event_logits.softmax(-1)
        conditional_height = nn.functional.softplus(output[:,8])
        support = events[:,7]
        conditional_lift_logit = output[:,9]
        lift = support*conditional_lift_logit.sigmoid()
        height = support*conditional_height
        if self.mode == 'direct_score':
            height = nn.functional.softplus(output[:,11])
        return dict(event_logits=event_logits,event_probability=events,
            support_probability=support,lift_probability=lift,
            conditional_height=conditional_height,conditional_lift_logit=conditional_lift_logit,
            loss_logit=output[:,10],loss_probability=output[:,10].sigmoid(),
            supported_height=height,physical=torch.cat((local,output[:,12:19]),-1))


def objective(pred,labels,mode):
    f = nn.functional
    if mode == 'direct_score':
        return f.smooth_l1_loss(pred['supported_height'],labels['supported_height'])
    support = labels['support']
    terms = [f.cross_entropy(pred['event_logits'],labels['event']),
        f.smooth_l1_loss(pred['supported_height'],labels['supported_height']),
        f.binary_cross_entropy_with_logits(pred['loss_logit'],labels['loss'].float()),
        f.smooth_l1_loss(pred['physical'][:,:36],labels['physical'][:,:36]),
        f.smooth_l1_loss(pred['physical'][:,36:42],labels['physical'][:,36:42]),
        f.smooth_l1_loss(pred['physical'][:,42],labels['physical'][:,42])]
    if support.any():
        terms += [f.smooth_l1_loss(pred['conditional_height'][support],labels['height'][support]),
            f.binary_cross_entropy_with_logits(pred['conditional_lift_logit'][support],labels['conditional_lift'][support].float())]
    return torch.stack(terms).mean()


def event_marginals(probabilities):
    bits = probabilities.new_tensor(EVENT_BITS)
    return probabilities @ bits
