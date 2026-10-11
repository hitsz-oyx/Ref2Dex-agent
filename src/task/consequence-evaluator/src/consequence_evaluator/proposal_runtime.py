"""Future-reference-free displacement prediction and train-only retrieval."""
import numpy as np
import torch
from torch import nn

from .proposal_history import MeasuredHistoryToTau, SCHEMA


class TauProposal(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        if checkpoint.get('schema') != SCHEMA or checkpoint.get('arm') != 'displacement':
            raise ValueError('measured-history displacement checkpoint required')
        state = checkpoint['model']
        self.model = MeasuredHistoryToTau(state['net.0.weight'].shape[0])
        self.model.load_state_dict(state)
        statistics = checkpoint['statistics']
        for name, value in [('history_mean', statistics['history'][0]),
                            ('history_scale', statistics['history'][1]),
                            ('target_mean', statistics['target'][0]),
                            ('target_scale', statistics['target'][1])]:
            self.register_buffer(name, torch.as_tensor(np.asarray(value), dtype=torch.float32))
        self.input_clip = float(checkpoint['input_clip'])

    def encode(self, history):
        return ((history-self.history_mean)/self.history_scale).clamp(-self.input_clip, self.input_clip)

    def forward(self, history, current_hand):
        if current_hand.shape != (len(history), 11, 3):
            raise ValueError('query current hand shape mismatch')
        return self.model(self.encode(history))*self.target_scale+self.target_mean+current_hand[:, None]


def retrieve_rows(query, bank_history, bank_episode, k=8):
    """Rank by normalized history only; future trajectories are not arguments."""
    episode = torch.as_tensor(bank_episode, device=query.device)
    unique = torch.unique(episode, sorted=True)
    if len(unique) < k or k < 1:
        raise ValueError('enough distinct train episodes required')
    distance = torch.cdist(query, bank_history).square()/query.shape[1]
    minima = []; source_rows = []
    for group in unique:
        rows = torch.nonzero(episode == group, as_tuple=True)[0]
        value, local = distance[:, rows].min(1)
        minima.append(value); source_rows.append(rows[local])
    scores = torch.stack(minima, 1); ids = torch.stack(source_rows, 1)
    values, groups = scores.topk(k, largest=False, sorted=True)
    return ids.gather(1, groups), values


def choose_generated(scores):
    """Ten declared generated candidates only; observed-GT is a comparator."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != 10 or not np.isfinite(scores).all():
        raise ValueError('ten finite generated scores required')
    return scores.argmax(1)
