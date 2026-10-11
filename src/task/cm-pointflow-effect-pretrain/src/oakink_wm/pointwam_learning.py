"""Current-only joint scene/hand learning contract for a bounded teacher Probe.

This small set forecaster is an explicit architecture/text-encoder substitute.
The user's untracked pointwam_core.py and the native world model are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from .pointwam_data import sha256


def load_qualified(manifest: Path, *, expected_clips: int = 12,
                   expected_fit_clips: int = 6) -> list[dict]:
    if not (1 <= expected_clips <= 52 and 2 <= expected_fit_clips < expected_clips):
        raise ValueError("cohort must declare 1..52 clips, at least two fit clips and a dev clip")
    data = json.loads(manifest.read_text())
    if data.get("status") != "COMPLETED" or not data.get("gate_pass"):
        raise ValueError("data qualification has not passed")
    clips = []
    owners = {}
    for row in data["clips"]:
        if not row["qualified"]:
            continue
        path = Path(row["pack"])
        if sha256(path) != row["pack_sha256"]:
            raise ValueError("qualified pack changed")
        with np.load(path, allow_pickle=False) as stored:
            pack = {key: stored[key].copy() for key in stored.files}
        if (str(pack["source_split"]) != "official_train"
                or str(pack["source_episode_id"]) != row["episode_id"]
                or row["partition"] not in ("fit_candidate", "dev_candidate")):
            raise ValueError("TEST/unknown source cannot be admitted to this Probe")
        if owners.setdefault(row["episode_id"], row["partition"]) != row["partition"]:
            raise ValueError("source episode crosses fit/dev")
        for kind in ("scene", "hand"):
            xyz, valid = pack[kind + "_xyz_m"], pack[kind + "_valid"]
            delta, mask = pack[kind + "_displacement_m"], pack[kind + "_target_valid"]
            if (len(xyz) != 31 or len(delta) != 30 or not np.isfinite(xyz[valid]).all()
                    or not np.array_equal(mask, valid[1:] & valid[0])
                    or not np.allclose(delta[mask], (xyz[1:] - xyz[0])[mask])):
                raise ValueError("stored trajectory/mask contract mismatch")
        if pack["hand_xyz_m"].shape != (31, 2, 10, 3) or pack["hand_order"].tolist() != ["right", "left"]:
            raise ValueError("semantic hand layout mismatch")
        if not np.allclose(pack["timestamps_s"][::3], pack["source_frame_ids"] / 30):
            raise ValueError("source clock mismatch")
        if not str(pack["language"]).strip():
            raise ValueError("empty instruction")
        clips.append(dict(pack=pack, path=str(path), sha256=row["pack_sha256"],
                          episode=row["episode_id"], partition=row["partition"]))
    if (len(clips) != expected_clips or len({c["path"] for c in clips}) != expected_clips
            or sum(c["partition"] == "fit_candidate" for c in clips) != expected_fit_clips):
        raise ValueError(f"expected the declared {expected_clips}-clip/{expected_fit_clips}-fit cohort")
    return clips


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())[:64]


def fit_contract(clips: list[dict]) -> dict:
    """Only fit labels determine normalization and only fit text determines vocab."""
    fit = [c for c in clips if c["partition"] == "fit_candidate"]
    words = sorted({w for c in fit for w in tokens(str(c["pack"]["language"]))})
    contract = dict(vocabulary={w: i + 1 for i, w in enumerate(words)}, unknown_id=0,
                    std_floor_m=.001, fit_pack_hashes=[c["sha256"] for c in fit])
    for kind in ("scene", "hand"):
        count = np.zeros((30, 1), np.float64)
        total = np.zeros((30, 3), np.float64); square = total.copy()
        for c in fit:
            pack = c["pack"]
            target = pack[kind + "_displacement_m"].reshape(30, -1, 3).astype(np.float64)
            valid = pack[kind + "_target_valid"].reshape(30, -1)
            if kind == "scene":
                target = np.clip(target, -.5, .5)
            target = np.where(valid[..., None], target, 0)
            count += valid.sum(1, keepdims=True)
            total += target.sum(1); square += (target ** 2).sum(1)
        if (count == 0).any():
            raise ValueError("missing fit supervision at a horizon step")
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean ** 2, 0)).clip(.001)
        contract[kind + "_mean_m"] = mean.tolist()
        contract[kind + "_std_m"] = std.tolist()
    return contract


def current_inputs(pack: dict, ids: np.ndarray, contract: dict) -> dict:
    """This function cannot inspect future coordinates, masks or trajectories."""
    hand = pack["hand_xyz_m"][0].reshape(20, 3)
    hvalid = pack["hand_valid"][0].reshape(20)
    if not hvalid.any() or not np.isfinite(hand[hvalid]).all():
        raise ValueError("no finite current hand anchor")
    anchor = hand[hvalid].mean(0)
    scene = pack["scene_xyz_m"][0, ids]
    svalid = pack["scene_valid"][0, ids]
    return dict(scene_xyz=np.where(svalid[:, None], (scene - anchor) / .5, 0).astype(np.float32),
                scene_rgb=pack["scene_rgb"][ids].astype(np.float32) / 255,
                scene_valid=svalid, hand_xyz=np.where(hvalid[:, None], (hand - anchor) / .5, 0).astype(np.float32),
                hand_valid=hvalid,
                language_ids=np.asarray([contract["vocabulary"].get(w, 0) for w in tokens(str(pack["language"]))] or [0]))


def supervision(pack: dict, ids: np.ndarray, contract: dict) -> dict:
    out = {}
    for kind in ("scene", "hand"):
        target = pack[kind + "_displacement_m"].reshape(30, -1, 3)
        valid = pack[kind + "_target_valid"].reshape(30, -1)
        if kind == "scene":
            target, valid = target[:, ids], valid[:, ids]
            target = np.clip(target, -.5, .5)
        mean = np.asarray(contract[kind + "_mean_m"], np.float32)[:, None]
        std = np.asarray(contract[kind + "_std_m"], np.float32)[:, None]
        out[kind + "_target"] = np.where(valid[..., None], (target - mean) / std, 0).astype(np.float32)
        out[kind + "_mask"] = valid
    xyz = pack["scene_xyz_m"][:, ids]
    good = pack["scene_valid"][1:, ids] & pack["scene_valid"][:-1, ids]
    delta = np.where(good, np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1), 0)
    out["scene_weight"] = (1 / (1 + np.exp(-np.clip(5 * (delta / .005 - 1), -60, 60)))
                           * good * out["scene_mask"]).astype(np.float32)
    return out


def batch(clips: list[dict], ids: list[np.ndarray], contract: dict, device: str) -> tuple[dict, dict]:
    inputs = [current_inputs(c["pack"], i, contract) for c, i in zip(clips, ids)]
    targets = [supervision(c["pack"], i, contract) for c, i in zip(clips, ids)]
    max_words = max(len(x["language_ids"]) for x in inputs)
    for x in inputs:
        length = len(x["language_ids"])
        x["language_mask"] = np.arange(max_words) < length
        x["language_ids"] = np.pad(x["language_ids"], (0, max_words - length))
    convert = lambda rows: {key: torch.as_tensor(np.stack([x[key] for x in rows]), device=device) for key in rows[0]}
    return convert(inputs), convert(targets)


class JointForecaster(nn.Module):
    """Small RGB/geometry set encoder with explicit hand IDs and two trajectory heads."""
    def __init__(self, words: int, hidden: int = 128):
        super().__init__()
        mlp = lambda n, m: nn.Sequential(nn.Linear(n, hidden), nn.GELU(), nn.Linear(hidden, m))
        self.scene = mlp(6, hidden); self.hand = mlp(3, hidden)
        self.hand_ids = nn.Embedding(20, hidden); self.text = nn.Embedding(words, hidden)
        self.context = mlp(3 * hidden, hidden)
        self.scene_head = mlp(2 * hidden, 90); self.hand_head = mlp(2 * hidden, 90)
        for head in (self.scene_head, self.hand_head):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    @staticmethod
    def pool(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (value * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def forward(self, scene_xyz, scene_rgb, scene_valid, hand_xyz, hand_valid,
                language_ids, language_mask) -> dict:
        scene = self.scene(torch.cat((scene_xyz, scene_rgb), -1))
        hand = self.hand(hand_xyz) + self.hand_ids.weight[None]
        text = self.text(language_ids)
        context = self.context(torch.cat((self.pool(scene, scene_valid), self.pool(hand, hand_valid),
                                          self.pool(text, language_mask)), -1))
        out = {}
        for name, value, head in (("scene", scene, self.scene_head), ("hand", hand, self.hand_head)):
            delta = head(torch.cat((value, context[:, None].expand_as(value)), -1))
            out[name] = delta.reshape(len(delta), -1, 30, 3).transpose(1, 2)
        return out


def joint_loss(prediction: dict, target: dict) -> tuple[torch.Tensor, dict]:
    scene_error = (prediction["scene"] - target["scene_target"]).square().mean(-1)
    weight = target["scene_weight"]
    if weight.sum() <= 0:
        raise ValueError("no adjacent valid scene supervision")
    scene = (scene_error * weight).sum() / weight.sum()
    hand_error = (prediction["hand"] - target["hand_target"]).square().mean(-1)
    hvalid = target["hand_mask"]
    counts = hvalid.sum((1, 2))
    if (counts == 0).any():
        raise ValueError("no hand supervision in a batch element")
    hand = ((hand_error * hvalid).sum((1, 2)) / counts).mean()
    return scene + hand, dict(scene=scene, hand=hand)


def metric_prediction(normalized: torch.Tensor, kind: str, contract: dict) -> np.ndarray:
    value = normalized.detach().cpu().numpy()
    return (value * np.asarray(contract[kind + "_std_m"])[None, :, None]
            + np.asarray(contract[kind + "_mean_m"])[None, :, None]).astype(np.float32)
