"""Compare recent physical history and current action for future-contact Cm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "outputs/Dexplore/agent_crossobject_train3_s179_e320"
FILES = {
    "train": RUN / "eval_s178_e320_full_train3/transitions.pt",
    "heldout": RUN / "eval_s174_e320_full_apple_heldout/transitions.pt",
}
CHECKPOINT_SHA = "e442bf481ad2f02f03d0f7bc5085e4378ffbb2547baf0db25626afd6e761f390"
INPUT_SHA = "35fffcb500f1f3db76fb8a59c940f5da0a113627bb0be9db0d68b0112fe7f516"
HISTORY = 10
FUTURE = 20
N_ENV = 64


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(which: str) -> tuple[dict, dict]:
    path = FILES[which]
    manifest = json.loads((path.parent / "run_manifest.json").read_text())
    if (manifest["run_status"] != "COMPLETED" or
            manifest["checkpoint_sha256"] != CHECKPOINT_SHA or
            manifest["input_manifest_sha256"] != INPUT_SHA or
            manifest["seed"] != (178 if which == "train" else 174)):
        raise ValueError(f"source provenance drift: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != "ref2dex.cmlite_transition.v1":
        raise ValueError("wrong transition schema")
    tensors = {name: value.reshape(-1, N_ENV, value.shape[-1])
               for name, value in payload.items() if isinstance(value, torch.Tensor)}
    length = tensors["done"].shape[0]
    if any(value.shape[0] != length for value in tensors.values()):
        raise ValueError("transition length mismatch")
    if not all(torch.isfinite(value.float()).all() for value in tensors.values()):
        raise FloatingPointError("nonfinite transitions")
    return tensors, {"path": str(path), "sha256": sha256(path),
                     "checkpoint_sha256": CHECKPOINT_SHA,
                     "input_manifest_sha256": INPUT_SHA}


def make_rows(tensors: dict, *, per_object: int, rng: torch.Generator):
    tmax = tensors["done"].shape[0]
    contact = ((tensors["hand_contact"][..., 0] > 0) &
               (tensors["object_contact"][..., 0] > 0))
    first_episode = tensors["done"][..., 0].int().cumsum(0) == 0
    valid = contact & first_episode
    eligible = torch.zeros_like(valid)
    eligible[HISTORY:tmax-FUTURE] = valid[HISTORY:tmax-FUTURE] & (
        first_episode[HISTORY+FUTURE:tmax])
    ids = tensors["data_id"][..., 0].long()
    object_ids = sorted(ids[eligible].unique().tolist())
    selected = []
    counts = {}
    for object_id in object_ids:
        indices = ((eligible) & (ids == object_id)).nonzero(as_tuple=False)
        if len(indices) < per_object:
            raise ValueError(f"insufficient object {object_id} rows: {len(indices)}")
        choice = indices[torch.randperm(len(indices), generator=rng)[:per_object]]
        selected.append(choice)
        counts[str(object_id)] = len(choice)
    chosen = torch.cat(selected)
    t, env = chosen[:, 0], chosen[:, 1]
    current_state = torch.cat([
        tensors["q"][t, env], tensors["dof_vel"][t, env],
        tensors["object_state"][t, env]], dim=-1).float()
    current_action = tensors["action"][t, env].float()
    history_t = t[:, None] - torch.arange(HISTORY - 1, -1, -1)[None, :]
    history_action_t = t[:, None] - torch.arange(HISTORY, 0, -1)[None, :]
    history_contact = contact[history_t, env[:, None]].float()
    history_z = (tensors["object_state"][history_t, env[:, None], 2] -
                 tensors["object_state"][t, env, 2][:, None]).float()
    history_action_z = tensors["action"][history_action_t, env[:, None], 2].float()
    history = torch.cat([history_contact, history_z, history_action_z], dim=-1)
    future_t = t[:, None] + torch.arange(1, FUTURE + 1)[None, :]
    future_contact = contact[future_t, env[:, None]].float().mean(-1)
    dz = (tensors["next_object_state"][t + FUTURE - 1, env, 2] -
          tensors["object_state"][t, env, 2]).float()
    supported_lift = (dz * future_contact).clamp(-0.3, 0.3) / 0.3
    if not (torch.isfinite(current_state).all() and torch.isfinite(history).all() and
            torch.isfinite(supported_lift).all()):
        raise FloatingPointError("nonfinite sampled feature or target")
    return {"state": current_state, "action": current_action,
            "history": history, "contact": future_contact,
            "lift": supported_lift, "object_id": ids[t, env],
            "env_id": env, "time": t}, counts


def input_for(rows: dict, arm: str, *, train: bool, rng: torch.Generator) -> torch.Tensor:
    action = rows["action"].clone()
    history = rows["history"].clone()
    if arm == "current_action":
        history.zero_()
    elif arm == "temporal_blind":
        action.zero_()
    elif arm == "temporal_shuffled" and train:
        for object_id in rows["object_id"].unique().tolist():
            indices = (rows["object_id"] == object_id).nonzero(as_tuple=False).flatten()
            action[indices] = action[indices[torch.randperm(len(indices), generator=rng)]]
    elif arm not in ("temporal_action", "temporal_shuffled"):
        raise ValueError(arm)
    return torch.cat([rows["state"], action, history], dim=-1)


class Head(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit(train_x: torch.Tensor, train: dict, test_x: torch.Tensor,
        test: dict) -> tuple[dict, dict]:
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(0.05)
    x = ((train_x - mean) / std).clamp(-10, 10)
    xt = ((test_x - mean) / std).clamp(-10, 10)
    torch.manual_seed(20260924)
    model = Head(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(20260924)
    for _ in range(400):
        indices = torch.randint(len(x), (512,), generator=generator)
        pred = model(x[indices])
        loss = F.mse_loss(pred[:, 0].sigmoid(), train["contact"][indices])
        loss = loss + 0.2 * F.smooth_l1_loss(pred[:, 1], train["lift"][indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite temporal Cm loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(xt)
        contact_pred = pred[:, 0].sigmoid()
        lift_pred = pred[:, 1]
        metrics = {"contact_rmse": float(F.mse_loss(contact_pred, test["contact"]).sqrt()),
                   "supported_lift_rmse_scaled": float(F.mse_loss(lift_pred, test["lift"]).sqrt()),
                   "train_final_loss": float(loss)}
    artifact = {"model": model.state_dict(), "mean": mean, "std": std}
    return metrics, artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    train_data, train_source = load("train")
    test_data, test_source = load("heldout")
    train, train_count = make_rows(train_data, per_object=1000,
                                   rng=torch.Generator().manual_seed(17824))
    test, test_count = make_rows(test_data, per_object=1000,
                                 rng=torch.Generator().manual_seed(17424))
    if train["state"].shape[0] != 3000 or test["state"].shape[0] != 1000:
        raise ValueError("unexpected train/test identity count")
    output.mkdir(parents=True)
    arms = {}
    for arm in ("current_action", "temporal_action", "temporal_blind",
                "temporal_shuffled"):
        x = input_for(train, arm, train=True,
                      rng=torch.Generator().manual_seed(20260924))
        xt = input_for(test, arm, train=False,
                       rng=torch.Generator().manual_seed(20260924))
        metrics, artifact = fit(x, train, xt, test)
        arms[arm] = metrics
        torch.save(artifact, output / f"{arm}.pt")
    temporal = arms["temporal_action"]["contact_rmse"]
    gate = (temporal <= 0.9 * arms["current_action"]["contact_rmse"] and
            temporal <= 0.9 * arms["temporal_blind"]["contact_rmse"] and
            temporal < arms["temporal_shuffled"]["contact_rmse"])
    report = {"experiment_id": "P-20260924-temporal-contact-cm",
              "train_source": train_source, "heldout_source": test_source,
              "train_samples_by_motion_id": train_count,
              "heldout_samples_by_motion_id": test_count,
              "history_steps": HISTORY, "future_steps": FUTURE,
              "first_episode_only": True, "current_contact_only": True,
              "arms": arms, "gate": gate}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arms": arms, "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
