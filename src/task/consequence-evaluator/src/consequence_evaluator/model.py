"""Equal-capacity E0/Eoracle with masked progress and independent branch scores."""
import torch
from torch import nn
from torch.nn import functional as F

from .data import K


class Evaluator(nn.Module):
    def __init__(self, history_dim, width=128, layers=2):
        super().__init__()
        self.history = nn.Linear(history_dim, width)
        self.action = nn.Linear(18, width)
        self.future = nn.Linear(12, width)
        self.time = nn.Parameter(torch.zeros(1, K + 1, width))
        block = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.progress = nn.Linear(width, 10)
        self.score = nn.Linear(width, 1)

    def forward(self, history, action, future, use_future=True):
        if action.shape[1:] != (K, 18) or future.shape[1:] != (K, 12):
            raise ValueError('24-step action/future contract required')
        if not use_future:
            future = torch.zeros_like(future)
        chunk = self.action(action) + self.future(future)
        tokens = torch.cat((self.history(history)[:, None], chunk), dim=1) + self.time
        encoded = self.encoder(tokens)
        return dict(progress_logits=self.progress(encoded[:, 1:]),
                    score=self.score(encoded[:, 1:].mean(1)).squeeze(-1))


def progress_loss(logits, progress, mask):
    """Robometer-inspired ten-bin soft CE, only where dense progress is known."""
    if not mask.any():
        return logits.sum() * 0
    positions = progress[mask] * 9
    lo, hi = positions.floor().long(), positions.ceil().long()
    upper = positions - lo
    logp = F.log_softmax(logits[mask], dim=-1)
    return (-logp.gather(-1, lo[:, None]).squeeze(-1) * (1 - upper)
            - logp.gather(-1, hi[:, None]).squeeze(-1) * upper).mean()


def matched_loss(chosen, rejected, chosen_labels, rejected_labels):
    """Independent scalar BT is an explicit adaptation, not Robometer's joint head."""
    rank = F.softplus(rejected['score'] - chosen['score']).mean()
    logits = torch.cat((chosen['progress_logits'], rejected['progress_logits']))
    target = torch.cat((chosen_labels['progress'], rejected_labels['progress']))
    mask = torch.cat((chosen_labels['progress_mask'], rejected_labels['progress_mask']))
    progress = progress_loss(logits, target, mask)
    return rank + progress, dict(preference=rank, progress=progress)
