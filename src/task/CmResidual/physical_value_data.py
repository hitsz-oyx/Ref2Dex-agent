"""Episode-indexed transitions and complete-return targets for HF08."""
from __future__ import annotations
import json
from pathlib import Path
import hashlib
import torch
from src.task.CmResidual.physical_value_contract import HISTORY, validate_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Episodes:
    def __init__(self, collections, gamma=.99):
        parts = []
        self.inputs = []
        for root in collections:
            result = json.loads((Path(root) / "results.json").read_text())
            if result["run_status"] != "COMPLETED" or result["mode"] != "collect":
                raise ValueError("requires completed collection")
            for shard in result["shards"]:
                path = Path(shard["path"])
                if sha(path) != shard["sha256"]:
                    raise ValueError("transition shard drift")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                validate_rows(payload)
                if abs(payload["gamma"] - gamma) > 1e-9:
                    raise ValueError("discount differs from collection")
                parts.append(payload)
                self.inputs.append(shard)
        keys = [k for k, v in parts[0].items() if isinstance(v, torch.Tensor)]
        self.data = {k: torch.cat([p[k] for p in parts]) for k in keys}
        del parts
        order = torch.argsort(self.data["episode_id"] * (int(self.data["step"].max()) + 1) + self.data["step"], stable=True)
        self.data = {k: v[order] for k, v in self.data.items()}
        self._index_complete(gamma)

    @classmethod
    def from_data(cls, data, gamma=.99):
        obj = cls.__new__(cls)
        obj.data, obj.inputs = data, []
        obj._index_complete(gamma)
        return obj

    def _index_complete(self, gamma):
        data = self.data
        episode = data["episode_id"]
        starts = torch.cat((torch.zeros(1, dtype=torch.long), (episode[1:] != episode[:-1]).nonzero().flatten() + 1))
        ends = torch.cat((starts[1:] - 1, torch.tensor([len(episode) - 1])))
        keep = torch.zeros(len(episode), dtype=torch.bool)
        complete = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            steps = data["step"][start:end + 1]
            expected = torch.arange(end - start + 1)
            if steps[0] != 0 or not data["done"][end]:
                continue
            if not torch.equal(steps, expected) or data["done"][start:end].any():
                raise ValueError("duplicate steps or reset inside episode")
            keep[start:end + 1] = True
            complete.append((start, end))
        if not complete:
            raise ValueError("no complete episodes")
        self.excluded_rows = int((~keep).sum())
        self.data = {k: v[keep] for k, v in data.items()}
        data = self.data
        episode = data["episode_id"]
        self.starts = torch.cat((torch.zeros(1, dtype=torch.long), (episode[1:] != episode[:-1]).nonzero().flatten() + 1))
        self.ends = torch.cat((self.starts[1:] - 1, torch.tensor([len(episode) - 1])))
        returns = torch.zeros(len(episode))
        running = torch.zeros(len(self.starts))
        lengths = self.ends - self.starts + 1
        for time in range(int(lengths.max()) - 1, -1, -1):
            valid = time < lengths
            indices = self.starts[valid] + time
            running[valid] = data["reward"][indices] + gamma * running[valid]
            returns[indices] = running[valid]
        data["return"] = returns
        self.episode_ids = episode[self.starts]
        # Split assignment is a predeclared deterministic episode hash;
        # rollout trajectories cannot cross fit and holdout.
        hold_episodes = self.episode_ids.remainder(10) < 2
        hold_row = torch.repeat_interleave(hold_episodes, lengths)
        self.fit = (~hold_row).nonzero().flatten()
        self.hold = hold_row.nonzero().flatten()

    def subset(self, target_rows):
        chosen = []
        count = 0
        for start, end in zip(self.starts.tolist(), self.ends.tolist()):
            if int(self.data["episode_id"][start]) % 10 < 2:
                continue
            chosen.append(torch.arange(start, end + 1))
            count += end - start + 1
            if count >= target_rows:
                break
        if not chosen:
            raise ValueError("empty fit subset")
        return torch.cat(chosen)

    def batch(self, indices, device):
        indices = indices.cpu().long()
        history = indices[:, None] - torch.arange(HISTORY - 1, -1, -1)[None]
        safe = history.clamp_min(0)
        mask = (history >= 0) & (self.data["episode_id"][safe] == self.data["episode_id"][indices, None])
        result = {k: v[indices].to(device) for k, v in self.data.items()}
        result["history_state"] = (self.data["state"][safe] * mask[:, :, None]).to(device)
        result["history_action"] = (self.data["previous_action"][safe] * mask[:, :, None]).to(device)
        result["history_mask"] = mask[:, :, None].float().to(device)
        return result

    def future(self, indices, offset, device):
        indices = indices.cpu().long()
        shifted = (indices + offset).clamp_max(len(self.data["state"]) - 1)
        valid = (self.data["episode_id"][shifted] == self.data["episode_id"][indices]) & (
            self.data["step"][shifted] == self.data["step"][indices] + offset)
        return self.batch(shifted, device), valid.to(device)
