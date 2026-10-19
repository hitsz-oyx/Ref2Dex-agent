"""Self-trained independent-command BC mean for the Inspire adapter."""
import torch


class InspireExecutor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Compact scratch BC used for the bounded embodiment smoke.
        self.network = torch.nn.Sequential(torch.nn.Linear(986,512),torch.nn.ReLU(),
            torch.nn.Linear(512,256),torch.nn.ReLU(),torch.nn.Linear(256,128),
            torch.nn.ReLU(),torch.nn.Linear(128,12))

    def forward(self, value):
        return self.network(value)


def command(model, value, packet):
    x = ((value-packet['input_mean'])/packet['input_std']).clamp(-10,10)
    return (model(x)*packet['output_std']+packet['output_mean']).clamp(-1,1)
