"""Task physics conditional on an executable H10 feedback control law."""
import math
import torch
from torch import nn
from .executable_contact_options import INDEPENDENT
from .native_pd_selector import physical_inputs


def windows(record):
    if record['schema'] != 'ref2dex.orientation_feedback_options.v1':
        raise ValueError('original full feedback-law windows required')
    if record['future_done'].any() or record['future_state'].shape[1:] != (10, 49):
        raise ValueError('complete source windows required')
    history = record['history']
    state = record['state']
    if not torch.equal(history[:, -1, :49], state):
        raise ValueError('prehistory alignment')
    physical, _ = physical_inputs(history, record['initial_hand_force'],
                                  record['initial_object_force'], record['mass_kg'],
                                  record['gravity_magnitude'], record['initial_clearance'],
                                  record['rest_z'])
    goal = (record['actual_pd_targets'][:, 0] - state[:, :18])[:, list(INDEPENDENT)]
    choice = record['assignment']
    expert = torch.where(choice == 7, 1, torch.where(choice == 6, 4, choice))
    law = torch.cat((torch.nn.functional.one_hot(expert, 6).float(),
                     (choice >= 6).float()[:, None]), -1)
    future = record['future_state']
    clearance = record['future_clearance']
    initial = record['initial_clearance']
    height = (state[:, 38] - record['rest_z']).clamp_min(0)
    pair = record['future_contact'].all(-1)
    clear = clearance >= .002
    support = pair[:, -3:].all(-1) & clear[:, -3:].all(-1)
    last_height = (future[:, -3:, 38].amin(-1) - record['rest_z']).clamp_min(0)
    lift = support & (last_height >= .03)
    was_clear = torch.cat(((initial >= .002)[:, None], clear[:, :-1]), -1).cummax(-1).values
    loss_event = (was_clear & ~clear).any(-1)
    score = last_height * support - height
    target = torch.cat(((future[:, :, 38] - state[:, None, 38]) / .01,
                        (clearance - initial[:, None]) / .01, pair.float(),
                        support.float()[:, None], loss_event.float()[:, None],
                        lift.float()[:, None], score[:, None] / .01), -1)
    delta = state[:, 45, None] / 30 * torch.arange(1, 11, device=state.device)[None]
    current_joint = history[:, -1, 49:51].bool().all(-1)
    current_support = current_joint & (initial >= .002)
    current_lift = current_support & (height >= .03)
    bits = torch.cat((current_joint[:, None].expand(-1, 10), current_support[:, None],
                      torch.zeros_like(current_support)[:, None], current_lift[:, None]), -1)
    cv_clear = (initial[:, None] + delta[:, -3:]).amin(-1) >= .002
    cv_score = (height + delta[:, -3:].amin(-1)).clamp_min(0) * current_joint * cv_clear - height
    prior = torch.cat((delta / .01, delta / .01,
                       (bits.float() * 2 - 1) * math.log(99), cv_score[:, None] / .01), -1)
    early = ~((state[:, 38] - record['rest_z'] >= .03) & (initial >= .002))
    return dict(history=history, physical=physical, goal=goal, law=law,
                native=record['native_observation'][:, 0], prior=prior,
                target=target, early=early, persist_lift=current_lift.float())


class ContactToLiftMacroModel(nn.Module):
    def __init__(self, physical_dim, native_dim, mode='cm'):
        super().__init__()
        self.mode = mode
        self.history = nn.GRU(69, 64, batch_first=True)
        self.physical = nn.Sequential(nn.Linear(physical_dim, 64), nn.SiLU(), nn.LayerNorm(64))
        self.native = nn.Sequential(nn.Linear(native_dim, 64), nn.SiLU(), nn.LayerNorm(64))
        self.head = nn.Sequential(nn.Linear(211, 256), nn.SiLU(), nn.Linear(256, 128),
                                  nn.SiLU(), nn.Linear(128, 34))

    def forward(self, history, physical, native, goal, law, prior):
        _, hidden = self.history(history)
        if self.mode == 'state_only':
            goal = torch.zeros_like(goal)
            law = torch.zeros_like(law)
        state = torch.cat((hidden[-1], self.physical(physical), self.native(native), goal, law), -1)
        return self.head(state) + prior


def loss(prediction, target):
    f = torch.nn.functional
    return (f.smooth_l1_loss(prediction[:, :20], target[:, :20])
            + f.smooth_l1_loss(prediction[:, 33], target[:, 33])
            + f.binary_cross_entropy_with_logits(prediction[:, 20:30], target[:, 20:30])
            + f.binary_cross_entropy_with_logits(prediction[:, 30:33], target[:, 30:33]))
